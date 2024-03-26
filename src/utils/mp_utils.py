import os
import numpy as np
import cv2
import time
from tqdm import tqdm
import multiprocessing
import glob

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import face_landmark

CUR_DIR = os.path.dirname(__file__)


class LMKExtractor():
    def __init__(self, FPS=25):
        # Create an FaceLandmarker object.
        self.mode = mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE
        base_options = python.BaseOptions(model_asset_path=os.path.join(CUR_DIR, 'mp_models/face_landmarker_v2_with_blendshapes.task'))
        base_options.delegate = mp.tasks.BaseOptions.Delegate.CPU
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            running_mode=self.mode,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.detector = face_landmark.FaceLandmarker.create_from_options(options)
        self.last_ts = 0
        self.frame_ms = int(1000 / FPS)

        det_base_options = python.BaseOptions(model_asset_path=os.path.join(CUR_DIR, 'mp_models/blaze_face_short_range.tflite'))
        det_options = vision.FaceDetectorOptions(base_options=det_base_options)
        self.det_detector = vision.FaceDetector.create_from_options(det_options)
                

    def __call__(self, img):
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        t0 = time.time()
        if self.mode == mp.tasks.vision.FaceDetectorOptions.running_mode.VIDEO:
            det_result = self.det_detector.detect(image)
            if len(det_result.detections) != 1:
                return None
            self.last_ts += self.frame_ms
            try:
                detection_result, mesh3d = self.detector.detect_for_video(image, timestamp_ms=self.last_ts)
            except:
                return None
        elif self.mode == mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE:
            # det_result = self.det_detector.detect(image)

            # if len(det_result.detections) != 1:
            #     return None
            try:
                detection_result, mesh3d = self.detector.detect(image)
            except:
                return None
            
        
        bs_list = detection_result.face_blendshapes
        if len(bs_list) == 1:
            bs = bs_list[0]
            bs_values = []
            for index in range(len(bs)):
                bs_values.append(bs[index].score)
            bs_values = bs_values[1:] # remove neutral
            trans_mat = detection_result.facial_transformation_matrixes[0]
            face_landmarks_list = detection_result.face_landmarks
            face_landmarks = face_landmarks_list[0]
            lmks = []
            for index in range(len(face_landmarks)):
                x = face_landmarks[index].x
                y = face_landmarks[index].y
                z = face_landmarks[index].z
                lmks.append([x, y, z])
            lmks = np.array(lmks)
            
            lmks3d = np.array(mesh3d.vertex_buffer)
            lmks3d = lmks3d.reshape(-1, 5)[:, :3]
            mp_tris = np.array(mesh3d.index_buffer).reshape(-1, 3) + 1

            return {
                "lmks": lmks,
                'lmks3d': lmks3d,
                "trans_mat": trans_mat,
                'faces': mp_tris,
                "bs": bs_values
            }
        else:
            # print('multiple faces in the image: {}'.format(img_path))
            return None
        