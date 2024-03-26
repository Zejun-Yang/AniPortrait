import os, io, csv, math, random, pdb
import cv2
import numpy as np
import json
from PIL import Image
from einops import rearrange

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPImageProcessor
import torch.distributed as dist


from src.utils.draw_util import FaceMeshVisualizer

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)



class FaceDatasetValid(Dataset):
    def __init__(
            self,
            json_path,
            extra_json_path=None,
            sample_size=[512, 512], sample_stride=4, sample_n_frames=16,
            is_image=False,
            sample_stride_aug=False
    ):
        zero_rank_print(f"loading annotations from {json_path} ...")
        self.data_dic_name_list, self.data_dic = self.get_data(json_path, extra_json_path)
        
        self.length = len(self.data_dic_name_list)
        zero_rank_print(f"data scale: {self.length}")

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        
        self.sample_stride_aug = sample_stride_aug

        self.sample_size = sample_size
        self.resize = transforms.Resize((sample_size[0], sample_size[1]))


        self.pixel_transforms = transforms.Compose([
            transforms.Resize([sample_size[1], sample_size[0]]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.visualizer = FaceMeshVisualizer(forehead_edge=False)
        self.clip_image_processor = CLIPImageProcessor()
        self.is_image = is_image

    def get_data(self, json_name, extra_json_name, augment_num=1):
        zero_rank_print(f"start loading data: {json_name}")
        with open(json_name,'r') as f:
            data_dic = json.load(f)

        data_dic_name_list = []
        for augment_index in range(augment_num):
            for video_name in data_dic.keys():
                data_dic_name_list.append(video_name)

        invalid_video_name_list = []
        for video_name in data_dic_name_list:
            video_clip_num = len(data_dic[video_name]['clip_data_list'])
            if video_clip_num < 1:
                invalid_video_name_list.append(video_name)
        for name in invalid_video_name_list:
            data_dic_name_list.remove(name)


        if extra_json_name is not None:
            zero_rank_print(f"start loading data: {extra_json_name}")
            with open(extra_json_name,'r') as f:
                extra_data_dic = json.load(f)
            data_dic.update(extra_data_dic)
            for augment_index in range(3*augment_num):
                for video_name in extra_data_dic.keys():
                    data_dic_name_list.append(video_name)
        random.shuffle(data_dic_name_list)
        zero_rank_print("finish loading")
        return data_dic_name_list, data_dic

    def __len__(self):
        return len(self.data_dic_name_list)
    
    def get_batch_wo_pose(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])

        source_anchor = random.sample(range(video_clip_num), 1)[0]
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_mesh2d_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['lmks_list']

        video_length = len(source_image_path_list)
        
        if self.sample_stride_aug:
            tmp_sample_stride = self.sample_stride if random.random() > 0.5 else 4
        else:
            tmp_sample_stride = self.sample_stride

        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * tmp_sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        ref_img_idx = random.randint(0, video_length - 1)

        ref_img = cv2.imread(source_image_path_list[ref_img_idx])
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = self.contrast_normalization(ref_img)
        
        ref_mesh2d_clip = np.load(source_mesh2d_path_list[ref_img_idx]).astype(float)
        ref_pose_image = self.visualizer.draw_landmarks(self.sample_size, ref_mesh2d_clip, normed=True)

        images = [cv2.imread(source_image_path_list[idx]) for idx in batch_index]
        images = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for  bgr_image in images]
        image_np = np.array([self.contrast_normalization(img) for img in images])

        pixel_values = torch.from_numpy(image_np).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
    
        mesh2d_clip = np.array([np.load(source_mesh2d_path_list[idx]).astype(float) for idx in batch_index])

        pixel_values_pose = []
        for frame_id in range(mesh2d_clip.shape[0]):
            normed_mesh2d = mesh2d_clip[frame_id]
            
            pose_image = self.visualizer.draw_landmarks(self.sample_size, normed_mesh2d, normed=True)
            pixel_values_pose.append(pose_image)
        pixel_values_pose = np.array(pixel_values_pose)

        if self.is_image:
            pixel_values = pixel_values[0]
            pixel_values_pose = pixel_values_pose[0]
            image_np = image_np[0]
        
        return ref_img, pixel_values_pose, image_np, ref_pose_image

    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        # convert input image to float32
        image = image.astype(np.float32)

        # normalize the image
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound

        # convert to uint8
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image

    def __getitem__(self, idx):
        ref_img, pixel_values_pose, tar_gt, pixel_values_ref_pose = self.get_batch_wo_pose(idx)

        sample = dict(
            pixel_values_pose=pixel_values_pose,
            ref_img=ref_img,
            tar_gt=tar_gt,
            pixel_values_ref_pose=pixel_values_ref_pose,
            )
        
        return sample
    


class FaceDataset(Dataset):
    def __init__(
            self,
            json_path,
            extra_json_path=None,
            sample_size=[512, 512], sample_stride=4, sample_n_frames=16,
            is_image=False, 
            sample_stride_aug=False
    ):
        zero_rank_print(f"loading annotations from {json_path} ...")
        self.data_dic_name_list, self.data_dic = self.get_data(json_path, extra_json_path)
        
        self.length = len(self.data_dic_name_list)
        zero_rank_print(f"data scale: {self.length}")

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        
        self.sample_stride_aug = sample_stride_aug

        self.sample_size = sample_size
        self.resize = transforms.Resize((sample_size[0], sample_size[1]))


        self.pixel_transforms = transforms.Compose([
            transforms.Resize([sample_size[1], sample_size[0]]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.visualizer = FaceMeshVisualizer(forehead_edge=False)
        self.clip_image_processor = CLIPImageProcessor()
        self.is_image = is_image

    def get_data(self, json_name, extra_json_name, augment_num=1):
        zero_rank_print(f"start loading data: {json_name}")
        with open(json_name,'r') as f:
            data_dic = json.load(f)

        data_dic_name_list = []
        for augment_index in range(augment_num):
            for video_name in data_dic.keys():
                data_dic_name_list.append(video_name)

        invalid_video_name_list = []
        for video_name in data_dic_name_list:
            video_clip_num = len(data_dic[video_name]['clip_data_list'])
            if video_clip_num < 1:
                invalid_video_name_list.append(video_name)
        for name in invalid_video_name_list:
            data_dic_name_list.remove(name)


        if extra_json_name is not None:
            zero_rank_print(f"start loading data: {extra_json_name}")
            with open(extra_json_name,'r') as f:
                extra_data_dic = json.load(f)
            data_dic.update(extra_data_dic)
            for augment_index in range(3*augment_num):
                for video_name in extra_data_dic.keys():
                    data_dic_name_list.append(video_name)
        random.shuffle(data_dic_name_list)
        zero_rank_print("finish loading")
        return data_dic_name_list, data_dic

    def __len__(self):
        return len(self.data_dic_name_list)

    
    def get_batch_wo_pose(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])

        source_anchor = random.sample(range(video_clip_num), 1)[0]
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_mesh2d_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['lmks_list']

        video_length = len(source_image_path_list)
        
        if self.sample_stride_aug:
            tmp_sample_stride = self.sample_stride if random.random() > 0.5 else 4
        else:
            tmp_sample_stride = self.sample_stride

        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * tmp_sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]
        
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = cv2.imread(source_image_path_list[ref_img_idx])
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = self.contrast_normalization(ref_img)
        ref_img_pil = Image.fromarray(ref_img)
        
        clip_ref_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values
        
        pixel_values_ref_img = torch.from_numpy(ref_img).permute(2, 0, 1).contiguous()
        pixel_values_ref_img = pixel_values_ref_img / 255.
        
        ref_mesh2d_clip = np.load(source_mesh2d_path_list[ref_img_idx]).astype(float)
        ref_pose_image = self.visualizer.draw_landmarks(self.sample_size, ref_mesh2d_clip, normed=True)
        pixel_values_ref_pose = torch.from_numpy(ref_pose_image).permute(2, 0, 1).contiguous()
        pixel_values_ref_pose = pixel_values_ref_pose / 255.

        images = [cv2.imread(source_image_path_list[idx]) for idx in batch_index]
        images = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for  bgr_image in images]
        image_np = np.array([self.contrast_normalization(img) for img in images])

        pixel_values = torch.from_numpy(image_np).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
    
        mesh2d_clip = np.array([np.load(source_mesh2d_path_list[idx]).astype(float) for idx in batch_index])

        pixel_values_pose = []
        for frame_id in range(mesh2d_clip.shape[0]):
            normed_mesh2d = mesh2d_clip[frame_id]
            
            pose_image = self.visualizer.draw_landmarks(self.sample_size, normed_mesh2d, normed=True)

            pixel_values_pose.append(pose_image)
            
        pixel_values_pose = np.array(pixel_values_pose)
        pixel_values_pose = torch.from_numpy(pixel_values_pose).permute(0, 3, 1, 2).contiguous()
        pixel_values_pose = pixel_values_pose / 255.
        
        if self.is_image:
            pixel_values = pixel_values[0]
            pixel_values_pose = pixel_values_pose[0]
        
        return pixel_values, pixel_values_pose, clip_ref_image, pixel_values_ref_img, pixel_values_ref_pose

    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        image = image.astype(np.float32)
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image

    def __getitem__(self, idx):
        pixel_values, pixel_values_pose, clip_ref_image, pixel_values_ref_img, pixel_values_ref_pose = self.get_batch_wo_pose(idx)

        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values_pose = self.pixel_transforms(pixel_values_pose)
        
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
        pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img)
        pixel_values_ref_img = pixel_values_ref_img.squeeze(0)
        
        pixel_values_ref_pose = pixel_values_ref_pose.unsqueeze(0)
        pixel_values_ref_pose = self.pixel_transforms(pixel_values_ref_pose)
        pixel_values_ref_pose = pixel_values_ref_pose.squeeze(0)
        
        drop_image_embeds = 1 if random.random() < 0.1 else 0
        
        sample = dict(
            pixel_values=pixel_values, 
            pixel_values_pose=pixel_values_pose,
            clip_ref_image=clip_ref_image,
            pixel_values_ref_img=pixel_values_ref_img,
            drop_image_embeds=drop_image_embeds,
            pixel_values_ref_pose=pixel_values_ref_pose,
            )
        
        return sample

def collate_fn(data): 
    pixel_values = torch.stack([example["pixel_values"] for example in data])
    pixel_values_pose = torch.stack([example["pixel_values_pose"] for example in data])
    clip_ref_image = torch.cat([example["clip_ref_image"] for example in data])
    pixel_values_ref_img = torch.stack([example["pixel_values_ref_img"] for example in data])
    drop_image_embeds = [example["drop_image_embeds"] for example in data]
    drop_image_embeds = torch.Tensor(drop_image_embeds)
    pixel_values_ref_pose = torch.stack([example["pixel_values_ref_pose"] for example in data])

    return {
        "pixel_values": pixel_values,
        "pixel_values_pose": pixel_values_pose,
        "clip_ref_image": clip_ref_image,
        "pixel_values_ref_img": pixel_values_ref_img,
        "drop_image_embeds": drop_image_embeds,
        "pixel_values_ref_pose": pixel_values_ref_pose,
    }

