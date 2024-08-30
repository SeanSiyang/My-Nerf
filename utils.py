import os
import json
import imageio
import cv2

import numpy as np

class DatasetProvider:
    """
    root: 数据集的路径
    transforms_file: json文件的名字
    half_resolution: 是否进行一半的分辨率采样
    """
    def __init__(self, root, transforms_file, half_resolution=False):
        self.meta = json.load(open(os.path.join(root, transforms_file), "r")) # 使用meta读取json数据
        self.root = root
        self.frames = self.meta['frames']
        self.images = []
        self.poses = [] # matrix
        self.camera_angle_x = self.meta['camera_angle_x']
        
        for frame in self.frames:
            image_file = os.path.join(self.root, frame["file_path"] + '.png')
            image = imageio.imread(image_file) # 800 x 800 x 4 rgb+透明度，需要把这个透明度处理掉，只需要RGB
            
            if half_resolution:
                image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) # 将 image 图像缩小为原始尺寸的 50%（即宽度和高度都缩小一半）
            self.images.append(image)
            self.poses.append(frame['transform_matrix'])
        
        # 将列表都stack处理了，因为希望以矩阵的方式来处理
        self.poses = np.stack(self.poses)
        # 对images进行stack处理的同时，再进行归一化，并指定类型
        self.images = (np.stack(self.images) / 255.).astype(np.float32)
        
        self.width = self.images.shape[2]
        self.height = self.images.shape[1]
        
        # 计算焦距
        self.focal = 0.5 * self.width / np.tan(0.5 * self.camera_angle_x)
        
        # 去掉透明度
        alpha = self.images[..., [3]]
        rgb = self.images[..., :3]
        self.images = rgb * alpha + (1 - alpha)