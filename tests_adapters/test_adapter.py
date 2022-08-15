from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Union

import torch
from datasets.commands.dummy_data import MockDownloadManager
import datasets


from transformers import (
    AutoModel,
    GlueDataset,
    GlueDataTrainingArguments,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
)
from transformers.testing_utils import torch_device


def make_config(config_class, **kwargs):
    return staticmethod(lambda: config_class(**kwargs))


class AdapterTestBase:
    # If not overriden by subclass, AutoModel should be used.
    model_class = AutoModel
    # Default shape of inputs to use
    default_input_samples_shape = (3, 64)

    def get_model(self):
        if self.model_class == AutoModel:
            model = AutoModel.from_config(self.config())
        else:
            model = self.model_class(self.config())
        model.to(torch_device)
        return model

    def get_input_samples(self, shape=None, vocab_size=5000, config=None):
        shape = shape or self.default_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(random.randint(0, vocab_size - 1))
        input_ids = torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()
        # this is needed e.g. for BART
        if config and config.eos_token_id is not None and config.eos_token_id < vocab_size:
            input_ids[input_ids == config.eos_token_id] = random.randint(0, config.eos_token_id - 1)
            input_ids[:, -1] = config.eos_token_id
        in_data = {"input_ids": input_ids}

        if config and config.is_encoder_decoder:
            in_data["decoder_input_ids"] = input_ids.clone()
        return in_data

    def add_head(self, model, name, **kwargs):
        model.add_classification_head(name, **kwargs)
        return model.heads[name].config["num_labels"]

    def dataset(self, tokenizer=None):
        # setup tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        return GlueDataset(data_args, tokenizer=tokenizer, mode="train")

    def data_collator():
        return None


class VisionAdapterTestBase(AdapterTestBase):
    default_input_samples_shape = (3, 3, 224, 224)

    def get_input_samples(self, shape=None, config=None):
        shape = shape or self.default_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.random())
        pixel_values = torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()
        in_data = {"pixel_values": pixel_values}

        return in_data

    def add_head(self, model, name, **kwargs):
        if "num_labels" not in kwargs:
            kwargs["num_labels"] = 10
        model.add_image_classification_head(name, **kwargs)
        return model.heads[name].config["num_labels"]

    def dataset(self, feature_extractor=None):
        if feature_extractor is None:
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.feature_extractor_name)

        def transform(example_batch):
            inputs = feature_extractor([x for x in example_batch["img"]], return_tensors="pt")

            inputs["labels"] = example_batch["label"]
            return inputs

        dataset_builder = datasets.load_dataset_builder("cifar10")

        mock_dl_manager = MockDownloadManager("cifar10", dataset_builder.config, datasets.Version("1.0.0"))
        dataset_builder.download_and_prepare(dl_manager=mock_dl_manager, ignore_verifications=True)

        dataset = dataset_builder.as_dataset(split="train")
        dataset = dataset.with_transform(transform)

        return dataset


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


class SpeechAdapterTestBase(AdapterTestBase):
    default_input_samples_shape = (3, 4096)

    def get_input_samples(self, shape=None, config=None):
        shape = shape or self.default_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.random())
        input_values = torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()
        in_data = {"input_values": input_values}

        return in_data

    def add_head(self, model, name, **kwargs):

        if "num_labels" not in kwargs:
            tokenizer = AutoTokenizer.from_pretrained(self.feature_extractor_name)
            kwargs["num_labels"] = len(tokenizer.get_vocab().keys())
            kwargs["pad_token_id"] = tokenizer.pad_token_id
        model.add_speech_recognition_head(name, **kwargs)
        return model.heads[name].config["num_labels"]

    def dataset(self, processor=None):
        if processor is None:
            processor = AutoProcessor.from_pretrained(self.feature_extractor_name)

        def prepare_dataset(batch):
            # load audio
            sample = batch["audio"]

            inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"])
            batch["input_values"] = inputs.input_values[0]
            batch["input_length"] = len(batch["input_values"])

            # encode targets
            with processor.as_target_processor():
                batch["labels"] = processor(batch["text"]).input_ids
            return batch

        dataset = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        dataset.cast_column("audio", datasets.features.Audio(sampling_rate=processor.feature_extractor.sampling_rate))
        dataset = dataset.map(prepare_dataset)
        return dataset

    def data_collator(self, processor=None):
        if processor is None:
            processor = AutoProcessor.from_pretrained(self.feature_extractor_name)
        return DataCollatorCTCWithPadding(processor)
