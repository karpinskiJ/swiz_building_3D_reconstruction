import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def find_keypoints(image_1, image_2):
    sift = cv2.xfeatures2d.SIFT_create()
    image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1_gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2_gray, None)

    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(descriptors_1, descriptors_2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good_matches.append(m)
    return np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]), np.float32([keypoints_2[m.trainIdx].pt for m in good_matches])

def reprojection_error(object_points, image_points, transformation_matrix, camera_matrix, is_homogeneous):
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    if is_homogeneous:
        object_points = cv2.convertPointsFromHomogeneous(object_points.T)
    projected_points, _ = cv2.projectPoints(object_points, rotation_vector, translation_vector, camera_matrix, None)
    projected_points = np.float32(projected_points[:, 0, :])
    total_error = cv2.norm(projected_points, np.float32(image_points.T) if is_homogeneous else np.float32(image_points), cv2.NORM_L2)
    return total_error / len(projected_points), object_points

def PnP(object_points, image_points, camera_matrix, distortion_coeffs, rotation_vector, is_initial):
    if is_initial:
        object_points = object_points[:, 0, :]
        image_points = image_points.T
        rotation_vector = rotation_vector.T
    _, calculated_rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, distortion_coeffs, cv2.SOLVEPNP_ITERATIVE)
    calculated_rotation_matrix, _ = cv2.Rodrigues(calculated_rotation_vector)

    if inliers is not None:
        image_points = image_points[inliers[:, 0]]
        object_points = object_points[inliers[:, 0]]
        rotation_vector = rotation_vector[inliers[:, 0]]
    return calculated_rotation_matrix, translation_vector, image_points, object_points, rotation_vector

def matching_points(image_points_1, image_points_2, image_points_3):
    common_points_1 = []
    common_points_2 = []
    for i in range(image_points_1.shape[0]):
        match_indices = np.where(image_points_2 == image_points_1[i, :])
        if match_indices[0].size != 0:
            common_points_1.append(i)
            common_points_2.append(match_indices[0][0])

    masked_array_1 = np.ma.array(image_points_2, mask=False)
    masked_array_1.mask[common_points_2] = True
    masked_array_1 = masked_array_1.compressed()
    masked_array_1 = masked_array_1.reshape(int(masked_array_1.shape[0] / 2), 2)

    masked_array_2 = np.ma.array(image_points_3, mask=False)
    masked_array_2.mask[common_points_2] = True
    masked_array_2 = masked_array_2.compressed()
    masked_array_2 = masked_array_2.reshape(int(masked_array_2.shape[0] / 2), 2)
    return np.array(common_points_1), np.array(common_points_2), masked_array_1, masked_array_2

def optimal_reprojection_error(params) -> np.array:
    transformation_matrix = params[0:12].reshape((3, 4))
    camera_matrix = params[12:21].reshape((3, 3))
    num_points = int(len(params[21:]) * 0.4)
    image_points = params[21:21 + num_points].reshape((2, int(num_points / 2))).T
    object_points = params[21 + num_points:].reshape((int(len(params[21 + num_points:]) / 3), 3))
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    projected_points, _ = cv2.projectPoints(object_points, rotation_vector, translation_vector, camera_matrix, None)
    projected_points = projected_points[:, 0, :]
    errors = [(image_points[idx] - projected_points[idx])**2 for idx in range(len(image_points))]
    return np.array(errors).ravel() / len(image_points)

def bundle_adjustment(object_points, image_points, initial_transformation_matrix, initial_camera_matrix, reprojection_threshold) -> tuple:
    optimization_variables = np.hstack((initial_transformation_matrix.ravel(), initial_camera_matrix.ravel()))
    optimization_variables = np.hstack((optimization_variables, image_points.ravel()))
    optimization_variables = np.hstack((optimization_variables, object_points.ravel()))

    optimized_values = least_squares(optimal_reprojection_error, optimization_variables, gtol=reprojection_threshold).x
    optimized_camera_matrix = optimized_values[12:21].reshape((3, 3))
    num_points = int(len(optimized_values[21:]) * 0.4)
    optimized_image_points = optimized_values[21:21 + num_points].reshape((2, int(num_points / 2))).T
    optimized_object_points = optimized_values[21 + num_points:].reshape((int(len(optimized_values[21 + num_points:]) / 3), 3))
    optimized_transformation_matrix = optimized_values[0:12].reshape((3, 4))
    return optimized_object_points, optimized_image_points, optimized_transformation_matrix

def triangulation(image_points_1, image_points_2, projection_matrix_1, projection_matrix_2):
    projection_matrix_1 = projection_matrix_1.T
    projection_matrix_2 = projection_matrix_2.T
    point_cloud = cv2.triangulatePoints(image_points_1, image_points_2, projection_matrix_1, projection_matrix_2)
    return projection_matrix_1, projection_matrix_2, (point_cloud / point_cloud[3])

def plot_3d_points(total_points, total_colors):

    total_points = np.array(total_points)
    total_colors = np.array(total_colors)

    x = total_points[:, 0]
    y = total_points[:, 1]
    z = total_points[:, 2]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=total_colors / 255.0)  

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()