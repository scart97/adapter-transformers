from typing import Optional

import torch

from ....models.wav2vec2.modeling_wav2vec2 import (
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    WAV_2_VEC_2_START_DOCSTRING,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...context import AdapterSetup
from ...heads import ModelWithFlexibleHeadsAdaptersMixin, SpeechRecognitionHead


@add_start_docstrings(
    """Wav2Vec Model with the option to add multiple flexible heads on top.""",
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2AdapterModel(ModelWithFlexibleHeadsAdaptersMixin, Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        self._init_head_modules()

        # Initialize weights and apply final processing
        # self.post_init()

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # retrieve loss input_lengths from attention_mask
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            head_outputs = self.forward_head(
                outputs,
                head_name=head,
                return_dict=return_dict,
                input_lengths=input_lengths,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model
            return outputs

    head_types = {
        "speech_recognition": SpeechRecognitionHead,
    }

    def add_speech_recognition_head(
        self,
        head_name,
        num_labels=2,
        layers=1,
        activation_function=None,
        overwrite_ok=False,
        id2label=None,
        pad_token_id=0,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
    ):
        """
        Adds an speech recognition head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 1.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """

        head = SpeechRecognitionHead(
            self,
            head_name,
            num_labels=num_labels,
            layers=layers,
            activation_function=activation_function,
            id2label=id2label,
            pad_token_id=pad_token_id,
            ctc_loss_reduction=ctc_loss_reduction,
            ctc_zero_infinity=ctc_zero_infinity,
        )
        self.add_prediction_head(head, overwrite_ok)
