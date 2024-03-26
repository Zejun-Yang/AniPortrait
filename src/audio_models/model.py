import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config

from .torch_utils import get_mask_from_lengths
from .wav2vec2 import Wav2Vec2Model


class Audio2MeshModel(nn.Module):
    def __init__(
        self,
        config
    ):  
        super().__init__()
        out_dim = config['out_dim']
        latent_dim = config['latent_dim']
        model_path = config['model_path']
        only_last_fetures = config['only_last_fetures']
        from_pretrained = config['from_pretrained']

        self._only_last_features = only_last_fetures

        self.audio_encoder_config = Wav2Vec2Config.from_pretrained(model_path, local_files_only=True)
        if from_pretrained:
            self.audio_encoder = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True)
        else:
            self.audio_encoder = Wav2Vec2Model(self.audio_encoder_config)
        self.audio_encoder.feature_extractor._freeze_parameters()

        hidden_size = self.audio_encoder_config.hidden_size

        self.in_fn = nn.Linear(hidden_size, latent_dim)
       
        self.out_fn = nn.Linear(latent_dim, out_dim)
        nn.init.constant_(self.out_fn.weight, 0)
        nn.init.constant_(self.out_fn.bias, 0)

    def forward(self, audio, label, audio_len=None):
        attention_mask = ~get_mask_from_lengths(audio_len) if audio_len else None
        
        seq_len = label.shape[1]

        embeddings = self.audio_encoder(audio, seq_len=seq_len, output_hidden_states=True,
                                        attention_mask=attention_mask)

        if self._only_last_features:
            hidden_states = embeddings.last_hidden_state
        else:
            hidden_states = sum(embeddings.hidden_states) / len(embeddings.hidden_states)

        layer_in = self.in_fn(hidden_states)
        out = self.out_fn(layer_in)

        return out, None

    def infer(self, input_value, seq_len):
        embeddings = self.audio_encoder(input_value, seq_len=seq_len, output_hidden_states=True)

        if self._only_last_features:
            hidden_states = embeddings.last_hidden_state
        else:
            hidden_states = sum(embeddings.hidden_states) / len(embeddings.hidden_states)

        layer_in = self.in_fn(hidden_states)
        out = self.out_fn(layer_in)

        return out
    

