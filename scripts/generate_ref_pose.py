import os
import math
import argparse

import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

from src.utils.mp_utils  import LMKExtractor
from src.utils.pose_util import smooth_pose_seq, matrix_to_euler_and_translation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_video", type=str, default='', help='path of input video')
    parser.add_argument("--save_path", type=str, default='', help='path to save pose')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    lmk_extractor = LMKExtractor()

    cap = cv2.VideoCapture(args.ref_video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  

    pbar = tqdm(range(total_frames), desc="processing ...")
    
    trans_mat_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)
        result = lmk_extractor(frame)
        if result is None:
            break
        trans_mat_list.append(result['trans_mat'].astype(np.float32))
    cap.release()
    
    total_frames = len(trans_mat_list)

    trans_mat_arr = np.array(trans_mat_list)

    # compute delta pose
    trans_mat_inv_frame_0 = np.linalg.inv(trans_mat_arr[0])
    pose_arr = np.zeros([trans_mat_arr.shape[0], 6])

    for i in range(pose_arr.shape[0]):
        pose_mat = trans_mat_inv_frame_0 @ trans_mat_arr[i]
        euler_angles, translation_vector = matrix_to_euler_and_translation(pose_mat)
        pose_arr[i, :3] =  euler_angles
        pose_arr[i, 3:6] =  translation_vector

    # interpolate to 30 fps
    new_fps = 30
    old_time = np.linspace(0, total_frames / fps, total_frames)
    new_time = np.linspace(0, total_frames / fps, int(total_frames * new_fps / fps))

    pose_arr_interp = np.zeros((len(new_time), 6))
    for i in range(6):
        interp_func = interp1d(old_time, pose_arr[:, i])
        pose_arr_interp[:, i] = interp_func(new_time)

    pose_arr_smooth = smooth_pose_seq(pose_arr_interp)
    np.save(args.save_path, pose_arr_smooth)


if __name__ == "__main__":
    main()

