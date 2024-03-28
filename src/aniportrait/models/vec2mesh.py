import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model, Wav2Vec2Config

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin

from aniportrait.utils import get_mask_from_lengths
from aniportrait.audio_models import PretrainedWav2Vec2Model

__all__ = ["Vec2MeshModel"]

class Vec2MeshModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        out_dim: int=1404,
        latent_dim: int=512,
        hidden_size: int=768,
        zero_init: bool=True,
    ):  
        super().__init__()
        self.register_to_config(
            out_dim=out_dim,
            latent_dim=latent_dim,
            hidden_size=hidden_size
        )
        self.in_fn = nn.Linear(hidden_size, latent_dim)
        self.out_fn = nn.Linear(latent_dim, out_dim)
        if zero_init:
            nn.init.constant_(self.out_fn.weight, 0)
            nn.init.constant_(self.out_fn.bias, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.in_fn(hidden_states)
        hidden_states = self.out_fn(hidden_states)
        return hidden_states
