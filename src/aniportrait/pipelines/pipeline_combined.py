# Inspired by https://github.com/huggingface/diffusers/blob/main/examples/community/stable_diffusion_mega.py
from __future__ import annotations
import os
import json
import torch
import numpy as np

from PIL import Image
from typing import Union, List, Dict, Any, Optional, Callable

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DiffusionPipeline
)

from huggingface_hub import hf_hub_download
from transformers import CLIPVisionModelWithProjection

from aniportrait.utils import PoseHelper, get_data_dir

from aniportrait.audio_models.audio2mesh import Audio2MeshModel
from aniportrait.models.unet_2d_condition import UNet2DConditionModel
from aniportrait.models.unet_3d_condition import UNet3DConditionModel
from aniportrait.models.pose_guider import PoseGuiderModel

from aniportrait.pipelines.pipeline_pose2img import Pose2ImagePipeline, Pose2ImagePipelineOutput
from aniportrait.pipelines.pipeline_pose2vid import Pose2VideoPipeline, Pose2VideoPipelineOutput
from aniportrait.pipelines.pipeline_pose2vid_long import Pose2LongVideoPipeline

__all__ = ["AniPortraitPipeline"]

class AniPortraitPipeline(DiffusionPipeline):
    """
    Combines all the pipelines into one and shares models.
    """
    vae_slicing: bool = False
    cpu_offload_gpu_id: Optional[int] = None

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuiderModel,
        audio_mesher: Audio2MeshModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler
        ]
    ) -> None:
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
            audio_mesher=audio_mesher
        )
        self.pose_helper = PoseHelper()

    @classmethod
    def from_single_file(
        cls,
        file_path_or_repository: str,
        filename: str="aniportrait.safetensors",
        config_filename: str="config.json",
        variant: Optional[str]=None,
        subfolder: Optional[str]=None,
        device: Optional[Union[str, torch.device]]=None,
        torch_dtype: Optional[torch.dtype]=None,
        cache_dir: Optional[str]=None,
    ) -> AniPortraitPipeline:
        """
        Loads the pipeline from a single file.
        """
        if variant is not None:
            filename, ext = os.path.splitext(filename)
            filename = f"{filename}.{variant}{ext}"

        if device is None:
            device = "cpu"
        else:
            device = str(device)

        if os.path.isdir(file_path_or_repository):
            model_dir = file_path_or_repository
            if subfolder:
                model_dir = os.path.join(model_dir, subfolder)
            file_path = os.path.join(model_dir, filename)
            config_path = os.path.join(model_dir, config_filename)
        elif os.path.isfile(file_path_or_repository):
            file_path = file_path_or_repository
            if os.path.isfile(config_filename):
                config_path = config_filename
            else:
                config_path = os.path.join(os.path.dirname(file_path), config_filename)
                if not os.path.exists(config_path) and subfolder:
                    config_path = os.path.join(os.path.dirname(file_path), subfolder, config_filename)
        elif re.search(r"^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$", file_path_or_repository):
            file_path = hf_hub_download(
                file_path_or_repository,
                filename,
                subfolder=subfolder,
                cache_dir=cache_dir,
            )
            try:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                )
            except:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    cache_dir=cache_dir,
                )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File {config_path} not found.")

        with open(config_path, "r") as f:
            aniportrait_config = json.load(f)

        # Create the scheduler
        scheduler = DDIMScheduler(**aniportrait_config["scheduler"])

        # Create the models
        with (init_empty_weights() if is_accelerate_available() else nullcontext()):
            # UNets
            reference_unet = UNet2DConditionModel.from_config(aniportrait_config["reference_unet"])
            denoising_unet = UNet3DConditionModel.from_config(aniportrait_config["denoising_unet"])

            # VAE
            vae = AutoencoderKL.from_config(aniportrait_config["vae"])

            # Image encoder
            image_encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(**aniportrait_config["image_encoder"]))

            # Guidance encoders
            if use_depth_guidance:
                guidance_encoder_depth = GuidanceEncoder(**aniportrait_config["guidance_encoder"])
            else:
                guidance_encoder_depth = None

            if use_normal_guidance:
                guidance_encoder_normal = GuidanceEncoder(**aniportrait_config["guidance_encoder"])
            else:
                guidance_encoder_normal = None

            if use_semantic_map_guidance:
                guidance_encoder_semantic_map = GuidanceEncoder(**aniportrait_config["guidance_encoder"])
            else:
                guidance_encoder_semantic_map = None

            if use_dwpose_guidance:
                guidance_encoder_dwpose = GuidanceEncoder(**aniportrait_config["guidance_encoder"])
            else:
                guidance_encoder_dwpose = None

        # Load the weights
        logger.debug("Models created, loading weights...")
        state_dicts = {}
        for key, value in iterate_state_dict(file_path):
            try:
                module, _, key = key.partition(".")
                if is_accelerate_available():
                    if module == "reference_unet":
                        set_module_tensor_to_device(reference_unet, key, device=device, value=value)
                    elif module == "denoising_unet":
                        set_module_tensor_to_device(denoising_unet, key, device=device, value=value)
                    elif module == "vae":
                        set_module_tensor_to_device(vae, key, device=device, value=value)
                    elif module == "image_encoder":
                        set_module_tensor_to_device(image_encoder, key, device=device, value=value)
                    elif module == "guidance_encoder_depth" and guidance_encoder_depth is not None:
                        set_module_tensor_to_device(guidance_encoder_depth, key, device=device, value=value)
                    elif module == "guidance_encoder_normal" and guidance_encoder_normal is not None:
                        set_module_tensor_to_device(guidance_encoder_normal, key, device=device, value=value)
                    elif module == "guidance_encoder_semantic_map" and guidance_encoder_semantic_map is not None:
                        set_module_tensor_to_device(guidance_encoder_semantic_map, key, device=device, value=value)
                    elif module == "guidance_encoder_dwpose" and guidance_encoder_dwpose is not None:
                        set_module_tensor_to_device(guidance_encoder_dwpose, key, device=device, value=value)
                    else:
                        raise ValueError(f"Unknown module: {module}")
                else:
                    if module not in state_dicts:
                        state_dicts[module] = {}
                    state_dicts[module][key] = value
            except (AttributeError, KeyError, ValueError) as ex:
                logger.warning(f"Skipping module {module} key {key} due to {type(ex)}: {ex}")
        if not is_accelerate_available():
            try:
                reference_unet.load_state_dict(state_dicts["reference_unet"])
                denoising_unet.load_state_dict(state_dicts["denoising_unet"])
                vae.load_state_dict(state_dicts["vae"])
                image_encoder.load_state_dict(state_dicts["image_encoder"], strict=False)
                if guidance_encoder_depth is not None:
                    guidance_encoder_depth.load_state_dict(state_dicts["guidance_encoder_depth"])
                if guidance_encoder_normal is not None:
                    guidance_encoder_normal.load_state_dict(state_dicts["guidance_encoder_normal"])
                if guidance_encoder_semantic_map is not None:
                    guidance_encoder_semantic_map.load_state_dict(state_dicts["guidance_encoder_semantic_map"])
                if guidance_encoder_dwpose is not None:
                    guidance_encoder_dwpose.load_state_dict(state_dicts["guidance_encoder_dwpose"])
                del state_dicts
                gc.collect()
            except KeyError as ex:
                raise RuntimeError(f"File did not provide a state dict for {ex}")

        # Create the pipeline
        pipeline = cls(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            guidance_encoder_depth=guidance_encoder_depth,
            guidance_encoder_normal=guidance_encoder_normal,
            guidance_encoder_semantic_map=guidance_encoder_semantic_map,
            guidance_encoder_dwpose=guidance_encoder_dwpose,
            scheduler=scheduler,
        )

        if torch_dtype is not None:
            pipeline.to(torch_dtype)

        pipeline.to(device)

        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("XFormers is not available, falling back to PyTorch attention")
        return pipeline

    def enable_vae_slicing(self) -> None:
        """
        Enables VAE slicing.
        """
        self.vae_slicing = True

    def disable_vae_slicing(self) -> None:
        """
        Disables VAE slicing.
        """
        self.vae_slicing = False

    def enable_sequential_cpu_offload(self, gpu_id: int=0) -> None:
        """
        Offloads the models to the CPU sequentially.
        """
        self.cpu_offload_gpu_id = gpu_id

    def disable_sequential_cpu_offload(self):
        """
        Disables the sequential CPU offload.
        """
        self.cpu_offload_gpu_id = None

    def get_default_face_landmarks(self) -> np.ndarray:
        """
        Gets the default set of face landmarks.
        """
        return np.load(os.path.join(get_data_dir(), "face_landmarks.npy"))

    def get_default_translation_matrix(self) -> np.ndarray:
        """
        Gets the default translation matrix.
        """
        return np.load(os.path.join(get_data_dir(), "translation_matrix.npy"))

    def get_default_pose_sequence(self) -> np.ndarray:
        """
        Gets the default pose sequence.
        """
        return np.load(os.path.join(get_data_dir(), "pose_sequence.npy"))

    @torch.no_grad()
    def img2pose(
        self,
        ref_image: Image.Image,
        width: Optional[int]=None,
        height: Optional[int]=None
    ) -> Image.Image:
        """
        Generates a pose image from a reference image.
        """
        return self.pose_helper.image_to_pose(ref_image, width=width, height=height)

    @torch.no_grad()
    def vid2pose(
        self,
        ref_images: List[Image.Image],
        width: Optional[int]=None,
        height: Optional[int]=None
    ) -> List[Image.Image]:
        """
        Generates a list of pose images from a list of reference images.
        """
        return [
            self.img2pose(ref_image, width=width, height=height)
            for ref_image in ref_images
        ]

    @torch.no_grad()
    def audio2pose(
        self,
        audio_path: str,
        fps: int=30,
        reference_image: Optional[Image.Image]=None,
        pose_reference_images: Optional[List[Image.Image]]=None,
        width: Optional[int]=None,
        height: Optional[int]=None
    ) -> List[Image.Image]:
        """
        Generates a pose image from an audio clip.
        """
        if reference_image is not None:
            image_width, image_height = reference_image.size
            if width is None:
                width = image_width
            if height is None:
                height = image_height
            landmarks = self.pose_helper.get_landmarks(reference_image)
            if not landmarks:
                raise ValueError("No face landmarks found in the reference image.")
            face_landmarks = landmarks["lmks3d"]
            translation_matrix = landmarks["trans_mat"]
        else:
            face_landmarks = self.get_default_face_landmarks()
            translation_matrix = self.get_default_translation_matrix()

        if pose_reference_images is not None:
            pose_sequence = self.pose_helper.images_to_pose_sequence(pose_reference_images)
        else:
            pose_sequence = self.get_default_pose_sequence()

        if width is None:
            width = 256
        if height is None:
            height = 256

        prediction = self.audio_mesher.infer_from_path(audio_path, fps=fps)
        prediction = prediction.squeeze().detach().cpu().numpy()
        prediction = prediction.reshape(prediction.shape[0], -1, 3)

        return self.pose_helper.pose_sequence_to_images(
            prediction + face_landmarks,
            pose_sequence,
            translation_matrix,
            width,
            height
        )

    @torch.no_grad()
    def pose2img(
        self,
        ref_image: Image.Image,
        pose_image: Image.Image,
        ref_pose_image: Image.Image,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        eta: float=0.0,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates an image from a reference image and pose image.
        """
        pipeline = Pose2ImagePipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler
        )
        if self.vae_slicing:
            pipeline.enable_vae_slicing()
        if self.cpu_offload_gpu_id is not None:
            pipeline.enable_sequential_cpu_offload(self.cpu_offload_gpu_id)

        return pipeline(
            ref_image=ref_image,
            pose_image=pose_image,
            ref_pose_image=ref_pose_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generation=generation,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs
        )

    @torch.no_grad()
    def pose2vid(
        self,
        ref_image: Image.Image,
        pose_images: List[Image.Image],
        ref_pose_image: Image.Image,
        width: int,
        height: int,
        video_length: int,
        num_inference_steps: int,
        guidance_scale: float,
        eta: float=0.0,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates a video from a reference image and a list of pose images.
        """
        pipeline = Pose2VideoPipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler
        )
        if self.vae_slicing:
            pipeline.enable_vae_slicing()
        if self.cpu_offload_gpu_id is not None:
            pipeline.enable_sequential_cpu_offload(self.cpu_offload_gpu_id)

        return pipeline(
            ref_image=ref_image,
            pose_images=pose_images,
            ref_pose_image=ref_pose_image,
            width=width,
            height=height,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generation=generation,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs
        )

    @torch.no_grad()
    def pose2vid_long(
        self,
        ref_image: Image.Image,
        pose_images: List[Image.Image],
        ref_pose_image: Image.Image,
        width: int,
        height: int,
        video_length: int,
        num_inference_steps: int,
        guidance_scale: float,
        eta: float=0.0,
        generation: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        output_type: Optional[str]="pil",
        return_dict: bool=True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None,
        callback_steps: Optional[int]=None,
        context_schedule: str="uniform",
        context_frames: int=16,
        context_overlap: int=4,
        context_batch_size: int=1,
        interpolation_factor: int=1,
        **kwargs: Any
    ) -> Pose2VideoPipelineOutput:
        """
        Generates a long video from a reference image and a list of pose images.
        """
        pipeline = Pose2LongVideoPipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler
        )
        if self.vae_slicing:
            pipeline.enable_vae_slicing()
        if self.cpu_offload_gpu_id is not None:
            pipeline.enable_sequential_cpu_offload(self.cpu_offload_gpu_id)

        return pipeline(
            ref_image=ref_image,
            pose_images=pose_images,
            ref_pose_image=ref_pose_image,
            width=width,
            height=height,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generation=generation,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            context_schedule=context_schedule,
            context_frames=context_frames,
            context_overlap=context_overlap,
            context_batch_size=context_batch_size,
            interpolation_factor=interpolation_factor,
            **kwargs
        )
