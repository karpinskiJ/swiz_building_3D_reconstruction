import cv2
import numpy as np
import os

class Image_loader():
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
        self.K[:2, :3] /= self.factor

    def downscale_image(self, image):
        for _ in range(1,int(self.factor / 2) + 1):
            image = cv2.pyrDown(image)
        return image
    
    def to_ply(self,point_cloud, colors):
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])


        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(self.path + '\\res\\' + self.image_list[0].split('\\')[-2] + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')  

