import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

class FaceMeshVisualizer:
    def __init__(self, forehead_edge=False):
        self.mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_mesh = mp_face_mesh
        self.forehead_edge = forehead_edge

        DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
        f_thick = 2
        f_rad = 1
        right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
        right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
        right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
        left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
        left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
        left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
        head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)
        
        mouth_draw_obl = DrawingSpec(color=(10, 180, 20), thickness=f_thick, circle_radius=f_rad)
        mouth_draw_obr = DrawingSpec(color=(20, 10, 180), thickness=f_thick, circle_radius=f_rad)
        
        mouth_draw_ibl = DrawingSpec(color=(100, 100, 30), thickness=f_thick, circle_radius=f_rad)
        mouth_draw_ibr = DrawingSpec(color=(100, 150, 50), thickness=f_thick, circle_radius=f_rad)
        
        mouth_draw_otl = DrawingSpec(color=(20, 80, 100), thickness=f_thick, circle_radius=f_rad)
        mouth_draw_otr = DrawingSpec(color=(80, 100, 20), thickness=f_thick, circle_radius=f_rad)
        
        mouth_draw_itl = DrawingSpec(color=(120, 100, 200), thickness=f_thick, circle_radius=f_rad)
        mouth_draw_itr = DrawingSpec(color=(150 ,120, 100), thickness=f_thick, circle_radius=f_rad)
        
        FACEMESH_LIPS_OUTER_BOTTOM_LEFT = [(61,146),(146,91),(91,181),(181,84),(84,17)]
        FACEMESH_LIPS_OUTER_BOTTOM_RIGHT = [(17,314),(314,405),(405,321),(321,375),(375,291)]
        
        FACEMESH_LIPS_INNER_BOTTOM_LEFT = [(78,95),(95,88),(88,178),(178,87),(87,14)]
        FACEMESH_LIPS_INNER_BOTTOM_RIGHT = [(14,317),(317,402),(402,318),(318,324),(324,308)]
        
        FACEMESH_LIPS_OUTER_TOP_LEFT = [(61,185),(185,40),(40,39),(39,37),(37,0)]
        FACEMESH_LIPS_OUTER_TOP_RIGHT = [(0,267),(267,269),(269,270),(270,409),(409,291)]
        
        FACEMESH_LIPS_INNER_TOP_LEFT = [(78,191),(191,80),(80,81),(81,82),(82,13)]
        FACEMESH_LIPS_INNER_TOP_RIGHT = [(13,312),(312,311),(311,310),(310,415),(415,308)]
        
        FACEMESH_CUSTOM_FACE_OVAL = [(176, 149), (150, 136), (356, 454), (58, 132), (152, 148), (361, 288), (251, 389), (132, 93), (389, 356), (400, 377), (136, 172), (377, 152), (323, 361), (172, 58), (454, 323), (365, 379), (379, 378), (148, 176), (93, 234), (397, 365), (149, 150), (288, 397), (234, 127), (378, 400), (127, 162), (162, 21)]

        # mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
        face_connection_spec = {}
        if self.forehead_edge:
            for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
                face_connection_spec[edge] = head_draw
        else:
            for edge in FACEMESH_CUSTOM_FACE_OVAL:
                face_connection_spec[edge] = head_draw
        for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
            face_connection_spec[edge] = left_eye_draw
        for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
            face_connection_spec[edge] = left_eyebrow_draw
        # for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
        #    face_connection_spec[edge] = left_iris_draw
        for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
            face_connection_spec[edge] = right_eye_draw
        for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
            face_connection_spec[edge] = right_eyebrow_draw
        # for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
        #    face_connection_spec[edge] = right_iris_draw
        # for edge in mp_face_mesh.FACEMESH_LIPS:
        #     face_connection_spec[edge] = mouth_draw
        
        for edge in FACEMESH_LIPS_OUTER_BOTTOM_LEFT:
            face_connection_spec[edge] = mouth_draw_obl
        for edge in FACEMESH_LIPS_OUTER_BOTTOM_RIGHT:
            face_connection_spec[edge] = mouth_draw_obr
        for edge in FACEMESH_LIPS_INNER_BOTTOM_LEFT:
            face_connection_spec[edge] = mouth_draw_ibl
        for edge in FACEMESH_LIPS_INNER_BOTTOM_RIGHT:
            face_connection_spec[edge] = mouth_draw_ibr
        for edge in FACEMESH_LIPS_OUTER_TOP_LEFT:
            face_connection_spec[edge] = mouth_draw_otl
        for edge in FACEMESH_LIPS_OUTER_TOP_RIGHT:
            face_connection_spec[edge] = mouth_draw_otr
        for edge in FACEMESH_LIPS_INNER_TOP_LEFT:
            face_connection_spec[edge] = mouth_draw_itl
        for edge in FACEMESH_LIPS_INNER_TOP_RIGHT:
            face_connection_spec[edge] = mouth_draw_itr

        
        iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}
        
        self.face_connection_spec = face_connection_spec
    def draw_pupils(self, image, landmark_list, drawing_spec, halfwidth: int = 2):
        """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
        landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
        if len(image.shape) != 3:
            raise ValueError("Input image must be H,W,C.")
        image_rows, image_cols, image_channels = image.shape
        if image_channels != 3:  # BGR channels
            raise ValueError('Input image must contain three channel bgr data.')
        for idx, landmark in enumerate(landmark_list.landmark):
            if (
                    (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                    (landmark.HasField('presence') and landmark.presence < 0.5)
            ):
                continue
            if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
                continue
            image_x = int(image_cols*landmark.x)
            image_y = int(image_rows*landmark.y)
            draw_color = None
            if isinstance(drawing_spec, Mapping):
                if drawing_spec.get(idx) is None:
                    continue
                else:
                    draw_color = drawing_spec[idx].color
            elif isinstance(drawing_spec, DrawingSpec):
                draw_color = drawing_spec.color
            image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color
        


    def draw_landmarks(self, image_size, keypoints, normed=False):
        ini_size = [512, 512]
        image = np.zeros([ini_size[1], ini_size[0], 3], dtype=np.uint8)
        new_landmarks = landmark_pb2.NormalizedLandmarkList()
        for i in range(keypoints.shape[0]):
            landmark = new_landmarks.landmark.add()
            if normed:
                landmark.x = keypoints[i, 0]
                landmark.y = keypoints[i, 1]
            else:
                landmark.x = keypoints[i, 0] / image_size[0]
                landmark.y = keypoints[i, 1] / image_size[1]
            landmark.z = 1.0

        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=new_landmarks,
            connections=self.face_connection_spec.keys(),
            landmark_drawing_spec=None,
            connection_drawing_spec=self.face_connection_spec
        )
        # draw_pupils(image, face_landmarks, iris_landmark_spec, 2)
        image = cv2.resize(image, (image_size[0], image_size[1]))
        
        return image

