import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

class Camera:
    def __init__(self, P):
        self.P = P
        self.K, self.R, self.t = self.decompose_projection_matrix(P)

    def decompose_projection_matrix(self,P):
        M = P[:, :3]
        
        K, R = np.linalg.qr(np.linalg.inv(M))
        
        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R
        
        if np.linalg.det(R) < 0:
            R = -R
            K = -K
        
        t = np.linalg.inv(K) @ P[:, 3]

        K /= K[2, 2]

        return K, R, t
    def getP(self):
        return self.P

    def get_intrinsics(self):
        return self.K

    def get_rotation(self):
        return self.R

    def get_translation(self):
        return self.t
    


def display_image(img,title = 'Displayed Image'):
    if img is None:
        print("Error: Could not read image.")
        return
    plt.figure(figsize=(20, 10))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()

def detect_and_match_keypoints(img1, img2, num_matches=50):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    matches = matches[:num_matches]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    img1_keypoints = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_keypoints = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    display_image(img1_keypoints, title = 'image1 keypoints')
    display_image(img2_keypoints, title = 'image2 keypoints')
    display_image(img_matches, title = 'Matches between Image 1 and Image 2')

    return points1, points2

def triangulate_points(cam1, cam2, points1, points2):
    points4D = cv2.triangulatePoints(cam1.getP(), cam2.getP(), points1.T, points2.T)
    points4D /= points4D[3]
    return points4D

def reprojection_error(params, points2D, camera_indices, point_indices, P):
    n_cameras = P.shape[0]
    n_points = points2D.shape[0]
    camera_params = params[:n_cameras * 12].reshape((n_cameras, 3, 4))
    points_3d = params[n_cameras * 12:].reshape((n_points, 3))

    projected_points = np.zeros((n_points, 2))
    for i in range(n_points):
        camera_index = camera_indices[i]
        point_3d = np.hstack((points_3d[i], 1))
        proj_2d = camera_params[camera_index] @ point_3d
        projected_points[i] = proj_2d[:2] / proj_2d[2]

    return (projected_points - points2D).ravel()

def bundle_adjustment(points_3d, points2D, camera_indices, point_indices, P):
    n_cameras = P.shape[0]
    n_points = points_3d.shape[0]
    camera_params = P.reshape((n_cameras, 12))
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    res = least_squares(reprojection_error, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(points2D, camera_indices, point_indices, P))
    return res.x[n_cameras * 12:].reshape((n_points, 3))


def icp(A, B, max_iterations=20, tolerance=1e-5):
    src = np.copy(A.T)
    dst = np.copy(B.T)

    prev_error = 0

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src, dst)
        T = best_fit_transform(src, dst[:, indices])

        src = (T @ np.vstack((src, np.ones((1, src.shape[1])))))[:3]

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    T = best_fit_transform(A.T, src)
    return T, distances

def nearest_neighbor(src, dst):
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst.T)
    distances, indices = neigh.kneighbors(src.T, return_distance=True)
    return distances.ravel(), indices.ravel()

def best_fit_transform(A, B):
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    Am = A - centroid_A[:, None]
    Bm = B - centroid_B[:, None]

    H = Am @ Bm.T

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def pca_base_plane(points_3d):
    mean = np.mean(points_3d, axis=1)
    centered_points = points_3d - mean[:, np.newaxis]
    U, S, Vt = np.linalg.svd(centered_points.T)
    normal = Vt[-1]
    return normal, mean