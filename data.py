import os
import numpy as np

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

def read_points_3d(path) -> np.array:
    with open(os.path.join(path,'house.p3d'), 'r') as file:
                result = []
                for line in file:
                    result.append([(token if token != '*' else -1)
                                for token in line.strip().split()])
    return np.asarray(result).astype(np.float64)

if __name__ == '__main__':
    print(read_camera_params('data/camera_params')[0])
    pass