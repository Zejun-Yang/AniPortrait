from __future__ import annotations

from typing import Any, Optional, Tuple, List, Union

import math
import torch
import librosa

from PIL import Image

from diffusers import DiffusionPipeline

from aniportrait.utils import (
    LandmarkExtractor,
    FaceMeshVisualizer,
    load_state_dict,
    iterate_state_dict
)
from aniportrait.models import PretrainedVec2MeshModel
from aniportrait.audio_models import PretrainedWav2Vec2Model

class Audio2PosePipeline(DiffusionPipeline):
    """
    Pipeline for converting audio to mesh.
    """
    def __init__(
        self,
        encoder: PretrainedWav2Vec2Model,
        mesher: Vec2MeshModel,
        use_last_audio_features: bool=True,
    ) -> None:
        super().__init__()
        self.register_modules(
            encoder=encoder,
            mesher=mesher,
        )
        self.register_to_config(
            use_last_audio_features=use_last_audio_features,
        )

    @classmethod
    def from_pretrained(
        cls,
        mesher_path: str,
        encoder_path: Optional[str]="facebook/wav2vec2-base-960h",
        *args: Any,
        **kwargs: Any,
    ) -> Audio2MeshPipeline:
        """
        Load a pretrained model.
        """
        encoder = PretrainedWav2Vec2Model.from_pretrained(
            encoder_path,
            *args,
            **kwargs
        )
        kwargs["hidden_size"] = encoder.config.hidden_size
        mesher = Vec2MeshModel()
        audio_encoder_state_dict = {}
        for key, value in iterate_state_dict(mesher_path):
            module, _, name = key.partition(".")
            if module == "audio_encoder":
                audio_encoder_state_dict[name] = value
        encoder.load_state_dict(audio_encoder_state_dict)
        encoder.save_pretrained("./temp_encoder")
        return cls(encoder=encoder, mesher=mesher)

    def get_audio_embeddings(
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
        audio_features = self.encoder.feature_extractor(
            speech_array,
            sampling_rate=sampling_rate,
        ).input_values
        sequence_length = math.ceil(audio_features.shape[1]/sampling_rate*fps)
        embeddings = self.encoder(audio_features, sequence_length, output_hidden_states=True)
        if self.config.use_last_audio_features:
            return embeddings.last_hidden_state, sequence_length
        else:
            return sum(embeddings.hidden_states) / len(embeddings.hidden_states), sequence_length

    @torch.no_grad()
    def __call__(
        self,
        audio_path: str,
        sampling_rate: int=16000,
        fps: int=30,
    ) -> torch.Tensor:
        """
        Convert audio to a pose sequence
        """
        # Get audio embeddings
        sample, sequence_length = self.get_audio_embeddings(
            audio_path,
            sampling_rate,
            fps
        )
        # Predict mesh points
        sample = self.mesher(sample, sequence_length)
        return sample
