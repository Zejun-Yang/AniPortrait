import argparse
import os
import ffmpeg
import random
from datetime import datetime
from pathlib import Path
from typing import List
import subprocess
import av
import numpy as np
import cv2
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

from src.audio_models.model import Audio2MeshModel
from src.audio_models.pose_model import Audio2PoseModel
from src.utils.audio_util import prepare_audio_feature
from src.utils.mp_utils  import LMKExtractor
from src.utils.draw_util import FaceMeshVisualizer
from src.utils.pose_util import project_points, smooth_pose_seq
from src.utils.frame_interpolation import init_frame_interpolation_model, batch_images_interpolation_tool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/prompts/animation_audio.yaml')
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("-acc", "--accelerate", action='store_true')
    parser.add_argument("--fi_step", type=int, default=3)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
        
    audio_infer_config = OmegaConf.load(config.audio_inference_config)
    # prepare model
    a2m_model = Audio2MeshModel(audio_infer_config['a2m_model'])
    a2m_model.load_state_dict(torch.load(audio_infer_config['pretrained_model']['a2m_ckpt']), strict=False)
    a2m_model.cuda().eval()

    a2p_model = Audio2PoseModel(audio_infer_config['a2p_model'])
    a2p_model.load_state_dict(torch.load(audio_infer_config['pretrained_model']['a2p_ckpt']), strict=False)
    a2p_model.cuda().eval()

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")


    pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(device="cuda", dtype=weight_dtype) # not use cross attention

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)


    lmk_extractor = LMKExtractor()
    vis = FaceMeshVisualizer(forehead_edge=False)
    
    if args.accelerate:
        frame_inter_model = init_frame_interpolation_model()

    for ref_image_path in config["test_cases"].keys():
        # Each ref_image may correspond to multiple actions
        for audio_path in config["test_cases"][ref_image_path]:
            ref_name = Path(ref_image_path).stem
            audio_name = Path(audio_path).stem

            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
            ref_image_np = cv2.resize(ref_image_np, (args.W, args.H))
            
            face_result = lmk_extractor(ref_image_np)
            assert face_result is not None, "No face detected."
            lmks = face_result['lmks'].astype(np.float32)
            ref_pose = vis.draw_landmarks((ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True)
            
            sample = prepare_audio_feature(audio_path, wav2vec_model_path=audio_infer_config['a2m_model']['model_path'])
            sample['audio_feature'] = torch.from_numpy(sample['audio_feature']).float().cuda()
            sample['audio_feature'] = sample['audio_feature'].unsqueeze(0)

            # inference
            pred = a2m_model.infer(sample['audio_feature'], sample['seq_len'])
            pred = pred.squeeze().detach().cpu().numpy()
            pred = pred.reshape(pred.shape[0], -1, 3)
            pred = pred + face_result['lmks3d']
            
            if 'pose_temp' in config and config['pose_temp'] is not None:
                pose_seq = np.load(config['pose_temp'])
                mirrored_pose_seq = np.concatenate((pose_seq, pose_seq[-2:0:-1]), axis=0)
                pose_seq = np.tile(mirrored_pose_seq, (sample['seq_len'] // len(mirrored_pose_seq) + 1, 1))[:sample['seq_len']]
            else:
                id_seed = random.randint(0, 99)
                id_seed = torch.LongTensor([id_seed]).cuda()

                # Currently, only inference up to a maximum length of 10 seconds is supported.
                chunk_duration = 5 # 5 seconds
                sr = 16000
                fps = 30
                chunk_size = sr * chunk_duration 

                audio_chunks = list(sample['audio_feature'].split(chunk_size, dim=1))
                seq_len_list = [chunk_duration*fps] * (len(audio_chunks) - 1) + [sample['seq_len'] % (chunk_duration*fps)] # 30 fps 

                audio_chunks[-2] = torch.cat((audio_chunks[-2], audio_chunks[-1]), dim=1)
                seq_len_list[-2] = seq_len_list[-2] + seq_len_list[-1]
                del audio_chunks[-1]
                del seq_len_list[-1]
    
                pose_seq = []
                for audio, seq_len in zip(audio_chunks, seq_len_list):
                    pose_seq_chunk = a2p_model.infer(audio, seq_len, id_seed)
                    pose_seq_chunk = pose_seq_chunk.squeeze().detach().cpu().numpy()
                    pose_seq_chunk[:, :3] *= 0.5
                    pose_seq.append(pose_seq_chunk)
                
                pose_seq = np.concatenate(pose_seq, 0)
                pose_seq = smooth_pose_seq(pose_seq, 7)

            # project 3D mesh to 2D landmark
            projected_vertices = project_points(pred, face_result['trans_mat'], pose_seq, [height, width])

            pose_images = []
            for i, verts in enumerate(projected_vertices):
                lmk_img = vis.draw_landmarks((width, height), verts, normed=False)
                pose_images.append(lmk_img)

            pose_list = []
            pose_tensor_list = []
            print(f"pose video has {len(pose_images)} frames, with {args.fps} fps")
            pose_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor()]
            )
            args_L = len(pose_images) if args.L is None else args.L
            for pose_image_np in pose_images[: args_L]:
                pose_image_pil = Image.fromarray(cv2.cvtColor(pose_image_np, cv2.COLOR_BGR2RGB))
                pose_tensor_list.append(pose_transform(pose_image_pil))
            sub_step = args.fi_step if args.accelerate else 1
            for pose_image_np in pose_images[: args_L: sub_step]:
                pose_image_np = cv2.resize(pose_image_np,  (width, height))
                pose_list.append(pose_image_np)
            
            pose_list = np.array(pose_list)
            
            video_length = len(pose_list)

            pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
            pose_tensor = pose_tensor.transpose(0, 1)
            pose_tensor = pose_tensor.unsqueeze(0)

            video = pipe(
                ref_image_pil,
                pose_list,
                ref_pose,
                width,
                height,
                video_length,
                args.steps,
                args.cfg,
                generator=generator,
            ).videos
            
            if args.accelerate:
                video = batch_images_interpolation_tool(video, frame_inter_model, inter_frames=args.fi_step-1)

            ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
            ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
                0
            )  # (1, c, 1, h, w)
            ref_image_tensor = repeat(
                ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video.shape[2]
            )
            
            video = torch.cat([ref_image_tensor, pose_tensor[:,:,:video.shape[2]], video], dim=0)
            save_path = f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_noaudio.mp4"
            save_videos_grid(
                video,
                save_path,
                n_rows=3,
                fps=args.fps,
            )
            
            stream = ffmpeg.input(save_path)
            audio = ffmpeg.input(audio_path)
            ffmpeg.output(stream.video, audio.audio, save_path.replace('_noaudio.mp4', '.mp4'), vcodec='copy', acodec='aac', shortest=None).run()
            os.remove(save_path)

if __name__ == "__main__":
    main()
    
