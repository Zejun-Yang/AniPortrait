import math

import numpy as np
from scipy.spatial.transform import Rotation as R


def create_perspective_matrix(aspect_ratio):
    kDegreesToRadians = np.pi / 180.
    near = 1
    far = 10000
    perspective_matrix = np.zeros(16, dtype=np.float32)

    # Standard perspective projection matrix calculations.
    f = 1.0 / np.tan(kDegreesToRadians * 63 / 2.)

    denom = 1.0 / (near - far)
    perspective_matrix[0] = f / aspect_ratio
    perspective_matrix[5] = f
    perspective_matrix[10] = (near + far) * denom
    perspective_matrix[11] = -1.
    perspective_matrix[14] = 1. * far * near * denom

    # If the environment's origin point location is in the top left corner,
    # then skip additional flip along Y-axis is required to render correctly.

    perspective_matrix[5] *= -1.
    return perspective_matrix


def project_points(points_3d, transformation_matrix, pose_vectors, image_shape):
    P = create_perspective_matrix(image_shape[1] / image_shape[0]).reshape(4, 4).T
    L, N, _ = points_3d.shape
    projected_points = np.zeros((L, N, 2))
    for i in range(L):
        points_3d_frame = points_3d[i]
        ones = np.ones((points_3d_frame.shape[0], 1))
        points_3d_homogeneous = np.hstack([points_3d_frame, ones])  
        transformed_points = points_3d_homogeneous @ (transformation_matrix @ euler_and_translation_to_matrix(pose_vectors[i][:3], pose_vectors[i][3:])).T @ P
        projected_points_frame = transformed_points[:, :2] / transformed_points[:, 3, np.newaxis] # -1 ~ 1
        projected_points_frame[:, 0] = (projected_points_frame[:, 0] + 1) * 0.5 * image_shape[1] 
        projected_points_frame[:, 1]  = (projected_points_frame[:, 1] + 1) * 0.5 * image_shape[0]
        projected_points[i] = projected_points_frame
    return projected_points


def project_points_with_trans(points_3d, transformation_matrix, image_shape):
    P = create_perspective_matrix(image_shape[1] / image_shape[0]).reshape(4, 4).T
    L, N, _ = points_3d.shape
    projected_points = np.zeros((L, N, 2))
    for i in range(L):
        points_3d_frame = points_3d[i]
        ones = np.ones((points_3d_frame.shape[0], 1))
        points_3d_homogeneous = np.hstack([points_3d_frame, ones])  
        transformed_points = points_3d_homogeneous @ transformation_matrix[i].T @ P
        projected_points_frame = transformed_points[:, :2] / transformed_points[:, 3, np.newaxis] # -1 ~ 1
        projected_points_frame[:, 0] = (projected_points_frame[:, 0] + 1) * 0.5 * image_shape[1] 
        projected_points_frame[:, 1]  = (projected_points_frame[:, 1] + 1) * 0.5 * image_shape[0]
        projected_points[i] = projected_points_frame
    return projected_points


def euler_and_translation_to_matrix(euler_angles, translation_vector):
    rotation = R.from_euler('xyz', euler_angles, degrees=True)
    rotation_matrix = rotation.as_matrix()
    
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation_vector
    
    return matrix


def matrix_to_euler_and_translation(matrix):
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles, translation_vector