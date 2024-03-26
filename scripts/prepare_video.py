import subprocess
from src.utils.mp_utils  import LMKExtractor
from src.utils.draw_util import FaceMeshVisualizer

import os
import numpy as np
import cv2


def crop_video(input_file, output_file):
    width = 800
    height = 800
    x = 550
    y = 50

    ffmpeg_cmd = f'ffmpeg -i {input_file} -filter:v "crop={width}:{height}:{x}:{y}" -c:a copy {output_file}'
    subprocess.call(ffmpeg_cmd, shell=True)



def extract_and_draw_lmks(input_file, output_file):
    lmk_extractor = LMKExtractor()
    vis = FaceMeshVisualizer()

    cap = cv2.VideoCapture(input_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    write_ref_img = False
    for i in range(200):
        ret, frame = cap.read()

        if not ret:
            break
            
        if not write_ref_img:
            write_ref_img = True
            cv2.imwrite(os.path.join(os.path.dirname(output_file), "ref_img.jpg"), frame)
        
        result = lmk_extractor(frame)

        if result is not None:
            lmks = result['lmks'].astype(np.float32)
            lmk_img = vis.draw_landmarks((frame.shape[1], frame.shape[0]), lmks, normed=True)
            out.write(lmk_img)
        else:
            print('multiple faces in the frame')
 

if __name__ == "__main__":
    
    input_file = "./Moore-AnimateAnyone/examples/video.mp4"
    lmk_video_path = "./Moore-AnimateAnyone/examples/pose.mp4"

    # crop video
    # crop_video(input_file, output_file)

    # extract and draw lmks
    extract_and_draw_lmks(input_file, lmk_video_path)