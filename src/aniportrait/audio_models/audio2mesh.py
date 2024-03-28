import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa

from typing import Optional, Tuple

from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin

from aniportrait.utils import get_mask_from_lengths
from aniportrait.audio_models.wav2vec2 import PretrainedWav2Vec2Model

__all__ = ['Audio2MeshModel']

class Audio2MeshModel(ModelMixin, ConfigMixin):
    def __init__(
		self,
        # wav2vec2 config
		attention_dropout=0.1,
		bos_token_id=1,
		codevector_dim=256,
		contrastive_logits_temperature=0.1,
		conv_bias=False,
		conv_dim=[512, 512, 512, 512, 512, 512, 512],
		conv_kernel=[10, 3, 3, 3, 3, 2, 2],
		conv_stride=[5, 2, 2, 2, 2, 2, 2],
		ctc_loss_reduction="sum",
		ctc_zero_infinity=False,
		diversity_loss_weight=0.1,
		do_stable_layer_norm=False,
		eos_token_id=2,
		feat_extract_activation="gelu",
		feat_extract_dropout=0.0,
		feat_extract_norm="group",
		feat_proj_dropout=0.1,
		feat_quantizer_dropout=0.0,
		final_dropout=0.1,
		gradient_checkpointing=False,
		hidden_act="gelu",
		hidden_dropout=0.1,
		hidden_dropout_prob=0.1,
		hidden_size=768,
		initializer_range=0.02,
		intermediate_size=3072,
		layer_norm_eps=1e-05,
		layerdrop=0.1,
		mask_feature_length=10,
		mask_feature_prob=0.0,
		mask_time_length=10,
		mask_time_prob=0.05,
		model_type="wav2vec2",
		num_attention_heads=12,
		num_codevector_groups=2,
		num_codevectors_per_group=320,
		num_conv_pos_embedding_groups=16,
		num_conv_pos_embeddings=128,
		num_feat_extract_layers=7,
		num_hidden_layers=12,
		num_negatives=100,
		pad_token_id=0,
		proj_codevector_dim=256,
		vocab_size=32,
        # wav2vec extractor config
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        padding_side="right",
        do_normalize=True,
        return_attention_mask=False,
        # audio2mesh config
        out_dim=1404,
        latent_dim=512,
        use_final_features=True,
        zero_init=True
    ) -> None:
        super().__init__()
        config = PretrainedWav2Vec2Model.config_class(
            attention_dropout=attention_dropout,
            bos_token_id=bos_token_id,
            codevector_dim=codevector_dim,
            contrastive_logits_temperature=contrastive_logits_temperature,
            conv_bias=conv_bias,
            conv_dim=conv_dim,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            ctc_loss_reduction=ctc_loss_reduction,
            ctc_zero_infinity=ctc_zero_infinity,
            diversity_loss_weight=diversity_loss_weight,
            do_stable_layer_norm=do_stable_layer_norm,
            eos_token_id=eos_token_id,
            feat_extract_activation=feat_extract_activation,
            feat_extract_dropout=feat_extract_dropout,
            feat_extract_norm=feat_extract_norm,
            feat_proj_dropout=feat_proj_dropout,
            feat_quantizer_dropout=feat_quantizer_dropout,
            final_dropout=final_dropout,
            gradient_checkpointing=gradient_checkpointing,
            hidden_act=hidden_act,
            hidden_dropout=hidden_dropout,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size,
            initializer_range=initializer_range,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            layerdrop=layerdrop,
            mask_feature_length=mask_feature_length,
            mask_feature_prob=mask_feature_prob,
            mask_time_length=mask_time_length,
            mask_time_prob=mask_time_prob,
            model_type=model_type,
            num_attention_heads=num_attention_heads,
            num_codevector_groups=num_codevector_groups,
            num_codevectors_per_group=num_codevectors_per_group,
            num_conv_pos_embedding_groups=num_conv_pos_embedding_groups,
            num_conv_pos_embeddings=num_conv_pos_embeddings,
            num_feat_extract_layers=num_feat_extract_layers,
            num_hidden_layers=num_hidden_layers,
            num_negatives=num_negatives,
            pad_token_id=pad_token_id,
            proj_codevector_dim=proj_codevector_dim,
            vocab_size=vocab_size
        )
        self.audio_encoder = PretrainedWav2Vec2Model(config)
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            padding_side=padding_side,
            do_normalize=do_normalize,
            return_attention_mask=return_attention_mask
        )
        self.register_to_config(
            out_dim=out_dim,
            latent_dim=latent_dim,
            use_final_features=use_final_features,
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            padding_side=padding_side,
            do_normalize=do_normalize,
            return_attention_mask=return_attention_mask,
        )
        self.in_fn = nn.Linear(hidden_size, latent_dim)
        self.out_fn = nn.Linear(latent_dim, out_dim)
        if zero_init:
            nn.init.constant_(self.in_fn.weight, 0)
            nn.init.constant_(self.in_fn.bias, 0)

    def forward(
        self,
        audio: torch.Tensor,
        label: torch.Tensor,
        audio_len: Optional[int]=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_mask = ~get_mask_from_lengths(audio_len) if audio_len else None
        
        seq_len = label.shape[1]

        embeddings = self.audio_encoder(
            audio,
            seq_len=seq_len,
            output_hidden_states=True,
            attention_mask=attention_mask
        )

        if self.config.use_final_features:
            hidden_states = embeddings.last_hidden_state
        else:
            hidden_states = sum(embeddings.hidden_states) / len(embeddings.hidden_states)

        layer_in = self.in_fn(hidden_states)
        out = self.out_fn(layer_in)

        return out, None

    @torch.no_grad()
    def infer(
        self,
        input_value: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Infer the model
        """
        embeddings = self.audio_encoder(input_value, seq_len=seq_len, output_hidden_states=True)

        if self.config.use_final_features:
            hidden_states = embeddings.last_hidden_state
        else:
            hidden_states = sum(embeddings.hidden_states) / len(embeddings.hidden_states)

        layer_in = self.in_fn(hidden_states)
        out = self.out_fn(layer_in)

        return out

    @torch.no_grad()
    def infer_from_path(
        self,
        audio_path: str,
        sampling_rate: int=16000,
        fps: int=30
    ) -> torch.Tensor:
        """
        Infer the model from an audio file
        """
        audio_features, sequence_length = self.get_audio_features(
            audio_path,
            sampling_rate=sampling_rate,
            fps=fps
        )
        return self.infer(audio_features, sequence_length)

    def get_audio_features(
        self,
        audio_path: str,
        sampling_rate: int=16000,
        fps: int=30,
    ) -> Tuple[torch.Tensor, int]:
        """
        Get audio embeddings from an audio file.
        """
        speech_array, sampling_rate = librosa.load(
            audio_path,
            sr=sampling_rate
        )
        audio_features = np.squeeze(
            self.feature_extractor(
                speech_array,
                sampling_rate=sampling_rate,
            ).input_values
        )
        sequence_length = math.ceil(len(audio_features)/sampling_rate*fps)
        audio_features = torch.from_numpy(audio_features).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        return audio_features, sequence_length
