import os
import ffmpeg
from PIL import Image
import cv2
from tqdm import tqdm

from src.utils.util import get_fps, read_frames, save_videos_from_pil
import numpy as np
from src.utils.draw_util import FaceMeshVisualizer
from src.utils.mp_utils  import LMKExtractor

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise ValueError(f"Path: {args.video_path} not exists")

    dir_path, video_name = (
        os.path.dirname(args.video_path),
        os.path.splitext(os.path.basename(args.video_path))[0],
    )
    out_path = os.path.join(dir_path, video_name + "_kps_noaudio.mp4")
    
    lmk_extractor = LMKExtractor()
    vis = FaceMeshVisualizer(forehead_edge=False)
    
    width = 512
    height = 512

    fps = get_fps(args.video_path)
    frames = read_frames(args.video_path)
    kps_results = []
    for i, frame_pil in enumerate(tqdm(frames)):
        image_np = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        image_np = cv2.resize(image_np, (height, width))
        face_result = lmk_extractor(image_np)
        try:
            lmks = face_result['lmks'].astype(np.float32)
            pose_img = vis.draw_landmarks((image_np.shape[1], image_np.shape[0]), lmks, normed=True)
            pose_img = Image.fromarray(cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB))
        except:
            pose_img = kps_results[-1]
            
        kps_results.append(pose_img)

    print(out_path.replace('_noaudio.mp4', '.mp4'))
    save_videos_from_pil(kps_results, out_path, fps=fps)
    
    audio_output = 'audio_from_video.aac'
    ffmpeg.input(args.video_path).output(audio_output, acodec='copy').run()
    stream = ffmpeg.input(out_path)
    audio = ffmpeg.input(audio_output)
    ffmpeg.output(stream.video, audio.audio, out_path.replace('_noaudio.mp4', '.mp4'), vcodec='copy', acodec='aac').run()
    os.remove(out_path)
    os.remove(audio_output)
