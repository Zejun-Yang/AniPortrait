import argparse
import os
import ffmpeg
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

from src.utils.mp_utils  import LMKExtractor
from src.utils.draw_util import FaceMeshVisualizer
from src.utils.pose_util import project_points_with_trans, matrix_to_euler_and_translation, euler_and_translation_to_matrix, smooth_pose_seq
from src.utils.frame_interpolation import init_frame_interpolation_model, batch_images_interpolation_tool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/prompts/animation_facereenac.yaml')
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--fps", type=int)
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
        for source_video_path in config["test_cases"][ref_image_path]:
            ref_name = Path(ref_image_path).stem
            pose_name = Path(source_video_path).stem

            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
            ref_image_np = cv2.resize(ref_image_np, (args.H, args.W))
            
            face_result = lmk_extractor(ref_image_np)
            assert face_result is not None, "Can not detect a face in the reference image."
            lmks = face_result['lmks'].astype(np.float32)
            ref_pose = vis.draw_landmarks((ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True)
            
            

            source_images = read_frames(source_video_path)
            src_fps = get_fps(source_video_path)
            print(f"source video has {len(source_images)} frames, with {src_fps} fps")
            pose_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor()]
            )
            
            step = 1
            if src_fps == 60:
                src_fps = 30
                step = 2
            
            pose_trans_list = []
            verts_list = []
            bs_list = []
            src_tensor_list = []
            args_L = len(source_images) if args.L is None else args.L*step
            for src_image_pil in source_images[: args_L: step]:
                src_tensor_list.append(pose_transform(src_image_pil))
            sub_step = step*args.fi_step if args.accelerate else step
            for src_image_pil in source_images[: args_L: sub_step]:
                src_img_np = cv2.cvtColor(np.array(src_image_pil), cv2.COLOR_RGB2BGR)
                frame_height, frame_width, _ = src_img_np.shape
                src_img_result = lmk_extractor(src_img_np)
                if src_img_result is None:
                    break
                pose_trans_list.append(src_img_result['trans_mat'])
                verts_list.append(src_img_result['lmks3d'])
                bs_list.append(src_img_result['bs'])

            trans_mat_arr = np.array(pose_trans_list)
            verts_arr = np.array(verts_list)
            bs_arr = np.array(bs_list)
            min_bs_idx = np.argmin(bs_arr.sum(1))
            
            # compute delta pose
            pose_arr = np.zeros([trans_mat_arr.shape[0], 6])

            for i in range(pose_arr.shape[0]):
                euler_angles, translation_vector = matrix_to_euler_and_translation(trans_mat_arr[i]) # real pose of source
                pose_arr[i, :3] =  euler_angles
                pose_arr[i, 3:6] =  translation_vector
            
            init_tran_vec = face_result['trans_mat'][:3, 3] # init translation of tgt
            pose_arr[:, 3:6] = pose_arr[:, 3:6] - pose_arr[0, 3:6] + init_tran_vec # (relative translation of source) + (init translation of tgt)

            pose_arr_smooth = smooth_pose_seq(pose_arr, window_size=3)
            pose_mat_smooth = [euler_and_translation_to_matrix(pose_arr_smooth[i][:3], pose_arr_smooth[i][3:6]) for i in range(pose_arr_smooth.shape[0])]    
            pose_mat_smooth = np.array(pose_mat_smooth)   

            # face retarget
            verts_arr = verts_arr - verts_arr[min_bs_idx] + face_result['lmks3d']
            # project 3D mesh to 2D landmark
            projected_vertices = project_points_with_trans(verts_arr, pose_mat_smooth, [frame_height, frame_width])

            pose_list = []
            for i, verts in enumerate(projected_vertices):
                lmk_img = vis.draw_landmarks((frame_width, frame_height), verts, normed=False)
                pose_image_np = cv2.resize(lmk_img,  (width, height))
                pose_list.append(pose_image_np)
            
            pose_list = np.array(pose_list)

            video_length = len(pose_list)

            src_tensor = torch.stack(src_tensor_list, dim=0)  # (f, c, h, w)
            src_tensor = src_tensor.transpose(0, 1)
            src_tensor = src_tensor.unsqueeze(0)

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

            video = torch.cat([ref_image_tensor, video, src_tensor[:,:,:video.shape[2]]], dim=0)
            save_path = f"{save_dir}/{ref_name}_{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_noaudio.mp4"
            save_videos_grid(
                video,
                save_path,
                n_rows=3,
                fps=src_fps if args.fps is None else args.fps,
            )
            
            audio_output = 'audio_from_video.aac'
            # extract audio
            ffmpeg.input(source_video_path).output(audio_output, acodec='copy').run()
            # merge audio and video
            stream = ffmpeg.input(save_path)
            audio = ffmpeg.input(audio_output)
            ffmpeg.output(stream.video, audio.audio, save_path.replace('_noaudio.mp4', '.mp4'), vcodec='copy', acodec='aac', shortest=None).run()
            
            os.remove(save_path)
            os.remove(audio_output)

if __name__ == "__main__":
    main()
