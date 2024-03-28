from __future__ import annotations

import numpy as np

from typing import Any, Dict, Optional, List, Tuple, Union

from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

from aniportrait.utils.mp_utils import LandmarkExtractor
from aniportrait.utils.draw_utils import FaceMeshVisualizer

__all__ = ["PoseHelper"]

class PoseHelper:
    def __init__(self):
        self.landmark_extractor = LandmarkExtractor()
        self.visualizer = FaceMeshVisualizer()
    
    def get_landmarks(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Get landmarks from the image
        """
        return self.landmark_extractor(image)

    def draw_landmarks(self, width: int, height: int, landmarks: Dict[str, Any]) -> Image.Image:
        """
        Draw landmarks on the image
        """
        return self.visualizer.draw_landmarks(
            (width, height),
            landmarks["lmks"].astype(np.float32),
            normed=True
        )

    def image_to_pose(
        self,
        image: Image.Image,
        width: Optional[int]=None,
        height: Optional[int]=None
    ) -> Optional[Image.Image]:
        """
        Get pose from the image
        """
        image_width, image_height = image.size
        if width is None:
            width = image_width
        if height is None:
            height = image_height
        landmarks = self.get_landmarks(image)
        if landmarks is None:
            return None
        return self.draw_landmarks(width, height, landmarks)

    def images_to_pose_sequence(
        self,
        images: List[Image.Image],
        fps: Optional[int]=None,
        new_fps: Optional[int]=None,
        include_images: bool=False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[Image.Image]]]:
        """
        Get pose sequence from the images
        """
        landmarks = [
            self.get_landmarks(image)
            for image in images
        ]
        # remove None values
        landmarks = [landmark for landmark in landmarks if landmark is not None]
        sequence = self.landmarks_to_pose_sequence(landmarks)
        if fps is not None and new_fps is not None:
            sequence = self.interpolate_pose_sequence(sequence, fps, new_fps)
        if include_images:
            width, height = images[0].size
            images = [
                self.draw_landmarks(width, height, landmark)
                for landmark in landmarks
            ]
            return sequence, images
        return sequence

    def pose_sequence_to_images(
        self,
        pred_pose_seq: np.ndarray,
        pose_sequence: np.ndarray,
        translation_matrix: np.ndarray,
        width: int,
        height: int,
    ) -> List[Image.Image]:
        """
        Get images from the pose sequence
        """
        sequence_length = pred_pose_seq.shape[0]
        pose_sequence = self.bounce_pose_to_sequence_length(pose_sequence, sequence_length)
        projected_vertices = self.project_points(
            pred_pose_seq,
            translation_matrix,
            pose_sequence,
            (width, height)
        )
        images = [
            self.visualizer.draw_landmarks(
                (width, height),
                projected_vertices[i],
                normed=False
            )
            for i in range(sequence_length)
        ]
        return images

    @classmethod
    def bounce_pose_to_sequence_length(
        cls,
        pose_seq: np.ndarray,
        sequence_length: int
    ) -> np.ndarray:
        """
        Repeat the pose sequence
        """
        repeat_pose_sequence = np.concatenate((pose_seq, pose_seq[-2:0:-1]), axis=0)
        repeat_pose_sequence = np.tile(repeat_pose_sequence, (sequence_length // len(repeat_pose_sequence) + 1, 1))[:sequence_length]
        return repeat_pose_sequence

    @classmethod
    def matrix_to_euler_and_translation(cls, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 4x4 transformation matrix to euler angles and translation vector
        """
        rotation_matrix = matrix[:3, :3]
        translation_vector = matrix[:3, 3]
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        return euler_angles, translation_vector

    @classmethod
    def smooth_pose_seq(cls, pose_seq: np.ndarray, window_size: int=5) -> np.ndarray:
        """
        Smooth the pose sequence
        """
        smoothed_pose_seq = np.zeros_like(pose_seq)

        for i in range(len(pose_seq)):
            start = max(0, i - window_size // 2)
            end = min(len(pose_seq), i + window_size // 2 + 1)
            smoothed_pose_seq[i] = np.mean(pose_seq[start:end], axis=0)

        return smoothed_pose_seq

    @classmethod
    def landmarks_to_pose_sequence(cls, landmarks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Get pose sequence from the images
        """
        trans_mats = np.array([
            landmark["trans_mat"].astype(np.float32)
            for landmark in landmarks
        ])

        # compute delta pose
        trans_mat_inv_frame_0 = np.linalg.inv(trans_mats[0])
        pose_arr = np.zeros([trans_mats.shape[0], 6])

        for i in range(pose_arr.shape[0]):
            pose_mat = trans_mat_inv_frame_0 @ trans_mats[i]
            euler_angles, translation_vector = cls.matrix_to_euler_and_translation(pose_mat)
            pose_arr[i, :3] = euler_angles
            pose_arr[i, 3:6] = translation_vector

        return pose_arr

    @classmethod
    def interpolate_pose_sequence(
        cls,
        pose_arr: np.ndarray,
        fps: int,
        new_fps: int=30
    ) -> np.ndarray:
        """
        Interpolate the pose sequence
        """
        total_frames = pose_arr.shape[0]
        old_time = np.linspace(0, total_frames / fps, total_frames)
        new_time = np.linspace(0, total_frames / fps, int(total_frames * new_fps / fps))
        pose_arr_interp = np.zeros((len(new_time), 6))

        for i in range(6):
            interp_func = interp1d(old_time, pose_arr[:, i])
            pose_arr_interp[:, i] = interp_func(new_time)

        return cls.smooth_pose_seq(pose_arr_interp)

    @classmethod
    def create_perspective_matrix(cls, aspect_ratio: float) -> np.ndarray:
        near = 1
        far = 10000
        perspective_matrix = np.zeros(16, dtype=np.float32)

        # Standard perspective projection matrix calculations.
        f = 1.0 / np.tan(np.pi / 180.0 * 63 / 2.0)

        denom = 1.0 / (near - far)
        perspective_matrix[0] = f / aspect_ratio
        perspective_matrix[5] = f
        perspective_matrix[10] = (near + far) * denom
        perspective_matrix[11] = -1.0
        perspective_matrix[14] = 1.0 * far * near * denom

        # If the environment's origin point location is in the top left corner,
        # then skip additional flip along Y-axis is required to render correctly.

        perspective_matrix[5] *= -1.0
        return perspective_matrix

    @classmethod
    def project_points(
        cls,
        points_3d: np.ndarray,
        transformation_matrix: np.ndarray,
        pose_vectors: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        P = cls.create_perspective_matrix(image_shape[1] / image_shape[0]).reshape(4, 4).T
        L, N, _ = points_3d.shape
        projected_points = np.zeros((L, N, 2))
        for i in range(L):
            points_3d_frame = points_3d[i]
            ones = np.ones((points_3d_frame.shape[0], 1))
            points_3d_homogeneous = np.hstack([points_3d_frame, ones])
            transformed_points = (
                points_3d_homogeneous
                @ (
                    transformation_matrix
                    @ cls.euler_and_translation_to_matrix(
                        pose_vectors[i][:3], pose_vectors[i][3:]
                    )
                ).T
                @ P
            )
            projected_points_frame = (
                transformed_points[:, :2] / transformed_points[:, 3, np.newaxis]
            )  # -1 ~ 1
            projected_points_frame[:, 0] = (
                (projected_points_frame[:, 0] + 1) * 0.5 * image_shape[1]
            )
            projected_points_frame[:, 1] = (
                (projected_points_frame[:, 1] + 1) * 0.5 * image_shape[0]
            )
            projected_points[i] = projected_points_frame
        return projected_points

    @classmethod
    def project_points_with_trans(
        cls,
        points_3d: np.ndarray,
        transformation_matrix: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        P = create_perspective_matrix(image_shape[1] / image_shape[0]).reshape(4, 4).T
        L, N, _ = points_3d.shape
        projected_points = np.zeros((L, N, 2))
        for i in range(L):
            points_3d_frame = points_3d[i]
            ones = np.ones((points_3d_frame.shape[0], 1))
            points_3d_homogeneous = np.hstack([points_3d_frame, ones])
            transformed_points = points_3d_homogeneous @ transformation_matrix[i].T @ P
            projected_points_frame = (
                transformed_points[:, :2] / transformed_points[:, 3, np.newaxis]
            )  # -1 ~ 1
            projected_points_frame[:, 0] = (
                (projected_points_frame[:, 0] + 1) * 0.5 * image_shape[1]
            )
            projected_points_frame[:, 1] = (
                (projected_points_frame[:, 1] + 1) * 0.5 * image_shape[0]
            )
            projected_points[i] = projected_points_frame
        return projected_points

    @classmethod
    def euler_and_translation_to_matrix(
        cls,
        euler_angles: np.ndarray,
        translation_vector: np.ndarray
    ) -> np.ndarray:
        rotation = R.from_euler("xyz", euler_angles, degrees=True)
        rotation_matrix = rotation.as_matrix()

        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation_vector

        return matrix
