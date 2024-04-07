# Adapted from https://github.com/dajes/frame-interpolation-pytorch
import os
import cv2
import numpy as np
import torch
import bisect
import shutil
import pdb
from tqdm import tqdm

def init_frame_interpolation_model():
    print("Initializing frame interpolation model")
    checkpoint_name = os.path.join("./pretrained_model/film_net_fp16.pt")

    model = torch.jit.load(checkpoint_name, map_location='cpu')
    model.eval()
    model = model.half()
    model = model.to(device="cuda")
    return model


def batch_images_interpolation_tool(input_tensor, model, inter_frames=1):

    video_tensor = []
    frame_num = input_tensor.shape[2]  # bs, channel, frame, height, width
    
    for idx in tqdm(range(frame_num-1)):
        image1 = input_tensor[:,:,idx]
        image2 = input_tensor[:,:,idx+1]

        results = [image1, image2]

        inter_frames = int(inter_frames)
        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))

        splits = torch.linspace(0, 1, inter_frames + 2)

        for _ in range(len(remains)):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            x0 = x0.half()
            x1 = x1.half()
            x0 = x0.cuda()
            x1 = x1.cuda()

            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = model(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            del remains[step]

        for sub_idx in range(len(results)-1):
            video_tensor.append(results[sub_idx].unsqueeze(2))

    video_tensor.append(input_tensor[:,:,-1].unsqueeze(2))
    video_tensor = torch.cat(video_tensor, dim=2)
    return video_tensor