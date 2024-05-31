import os
import numpy as np
import cv2

def read_camera_params(path) -> np.array:
    result = []
    for file_path in os.listdir(path):
        if file_path.endswith('.P'):
            with open(os.path.join(path,file_path), 'r') as file:
                res = []
                for line in file:
                    res.append([(token if token != '*' else -1)
                                for token in line.strip().split()])
        result.append(res)
    return np.asarray(result).astype(np.float64)

def read_images(path):
    result = []
    for file_path in os.listdir(path):
        if file_path.endswith('.pgm'):
            img = cv2.imread(os.path.join(path,file_path))
                
        result.append(img)
    return result

if __name__ == '__main__':
    print(read_camera_params('data/camera_params')[0])
    pass