import cv2
import numpy as np
import os

class Image():
    def __init__(self, img_path):
        with open(os.path.join(img_path, 'K.txt')) as f:
            lines = f.read().split('\n')
            matrix = []
            for line in lines:
                values = line.strip().split(' ')
                float_values = list(map(float, values))
                matrix.append(float_values)
            self.K = np.array(matrix)
            self.image_list = []

        for image in sorted(os.listdir(img_path)):
            if image.lower().endswith(('.jpg', '.png')):
                self.image_list.append(os.path.join(img_path, image))

        self.path = os.getcwd()
        self.factor = 2 # scale factor = 2, obtained experimentaly
        self.downscale()

    def downscale(self):
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor

    def downscale_image(self, image):
        for _ in range(1,int(self.factor / 2) + 1):
            image = cv2.pyrDown(image)
        return image