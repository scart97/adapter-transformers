from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import (
    ModelAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)


class Wav2VecEncoderLayerAdaptersMixin:
    """Adds adapters to the Wav2VecEncoderLayer module of Wav2Vec."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class Wav2VecModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the Wav2Vec module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layers):
            yield i, layer


class Wav2VecModelWithHeadsAdaptersMixin(ModelWithHeadsAdaptersMixin):
    pass
