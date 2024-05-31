import cv2
import numpy as np
from matplotlib import pyplot as plt

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