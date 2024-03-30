import os
import argparse

import numpy as np
import cv2
from tqdm import tqdm
import glob
import json

from src.utils.mp_utils  import LMKExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='', help='path of dataset')
    parser.add_argument("--output_dir", type=str, default='',  help='path to save extracted annotations')
    parser.add_argument("--training_json", type=str, default='',  help='path to save training json')
    args = parser.parse_args()
    return args


def generate_training_json_mesh(video_dir, face_info_dir, res_json_path, min_clip_length=30):
    video_name_list = sorted(os.listdir(face_info_dir))
    res_data_dic = {}

    pbar = tqdm(range(len(video_name_list)))


    for video_index, video_name in enumerate(video_name_list):
        pbar.update(1)

        tem_dic = {}
        tem_tem_dic = {}
        video_clip_dir = os.path.join(video_dir, video_name)
        lmks_clip_dir = os.path.join(face_info_dir, video_name)

        video_clip_num = 1
        video_data_list = []

        frame_path_list = sorted(glob.glob(os.path.join(video_clip_dir, '*.png')))
        lmks_path_list = sorted(glob.glob(os.path.join(lmks_clip_dir, '*lmks.npy')))

        min_len = min(len(frame_path_list), len(lmks_path_list))
        frame_path_list = frame_path_list[:min_len]
        lmks_path_list = lmks_path_list[:min_len]


        if min_len < min_clip_length:
            info = 'min length: {} {}'.format(video_name, min_len)
            video_clip_num -= 1
            continue

        first_frame_basename = os.path.basename(frame_path_list[0]).split('.')[0]
        first_lmks_basename = os.path.basename(lmks_path_list[0]).split('_')[0]
        last_frame_basename = os.path.basename(frame_path_list[-1]).split('.')[0]
        last_lmks_basename = os.path.basename(lmks_path_list[-1]).split('_')[0]
    
        if (first_frame_basename != first_lmks_basename) or (last_frame_basename != last_lmks_basename):
            info = 'different length skip: {} , length {}/{}, frame/lmks'.format(video_name, len(frame_path_list), len(lmks_path_list))
            video_clip_num -= 1
            continue

        frame_name_list = [os.path.join(video_name, os.path.basename(item)) for item in frame_path_list]

        tem_tem_dic['frame_name_list'] = frame_name_list
        tem_tem_dic['frame_path_list'] = frame_path_list
        tem_tem_dic['lmks_list'] = lmks_path_list
        video_data_list.append(tem_tem_dic)

        tem_dic['video_clip_num'] = video_clip_num
        tem_dic['clip_data_list'] = video_data_list
        res_data_dic[video_name] = tem_dic

    with open(res_json_path, 'w') as f:
        json.dump(res_data_dic, f)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    folders = [f.path for f in os.scandir(args.input_dir) if f.is_dir()]
    folders.sort()

    lmk_extractor = LMKExtractor()

    pbar = tqdm(range(len(folders)), desc="processing ...")
    for folder in folders:
        pbar.update(1)
        output_subdir = os.path.join(args.output_dir, os.path.basename(folder))
        os.makedirs(output_subdir, exist_ok=True)
        for img_file in sorted(glob.glob(os.path.join(folder, "*.png"))):
            base = os.path.basename(img_file)
            lmks_output_file = os.path.join(output_subdir, os.path.splitext(base)[0] + "_lmks.npy")
            lmks3d_output_file = os.path.join(output_subdir, os.path.splitext(base)[0] + "_lmks3d.npy")
            trans_mat_output_file = os.path.join(output_subdir, os.path.splitext(base)[0] + "_trans_mat.npy")
            bs_output_file = os.path.join(output_subdir, os.path.splitext(base)[0] + "_bs.npy")
                
            img = cv2.imread(img_file)
            result = lmk_extractor(img)
    
            if result is not None:
                np.save(lmks_output_file, result['lmks'].astype(np.float32))
                np.save(lmks3d_output_file, result['lmks3d'].astype(np.float32))
                np.save(trans_mat_output_file, result['trans_mat'].astype(np.float32))
                np.save(bs_output_file, np.array(result['bs']).astype(np.float32))

    # write json
    generate_training_json_mesh(args.input_dir, args.output_dir, args.training_json, min_clip_length=30)


if __name__ == "__main__":
    main()


