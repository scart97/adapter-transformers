import unittest

from tests.models.wav2vec2.test_modeling_wav2vec2 import Wav2Vec2Config, Wav2Vec2ModelTest
from transformers import Wav2Vec2AdapterModel
from transformers.testing_utils import require_torch

from .methods import BottleneckAdapterTestMixin, CompacterTestMixin, LoRATestMixin, PrefixTuningTestMixin
from .test_adapter import SpeechAdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class Wav2Vec2AdapterModelTest(AdapterModelTesterMixin, Wav2Vec2ModelTest):
    all_model_classes = (Wav2Vec2AdapterModel,)


class Wav2Vec2AdapterTestBase(SpeechAdapterTestBase):
    config_class = Wav2Vec2Config
    config = make_config(
        Wav2Vec2Config,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = "facebook/wav2vec2-base-960h"


@require_torch
class Wav2Vec2AdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    # AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    # ParallelAdapterInferenceTestMixin,
    # ParallelTrainingMixin,
    Wav2Vec2AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class Wav2Vec2ClassConversionTest(
    ModelClassConversionTestMixin,
    Wav2Vec2AdapterTestBase,
    unittest.TestCase,
):
    pass
