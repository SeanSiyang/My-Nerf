import numpy as np
import torch

from utils import DatasetProvider

class NeRFDataset:
    # 传入provider是因为要取里面的东西
    # provider: DatasetProvider 形参这样写是为了能有提示
    def __init__(self, provider: DatasetProvider, batch_size=1024, device='cuda:0'):
        self.images = provider.images
        self.poses = provider.poses
        self.focal = provider.focal
        self.width = provider.width
        self.height = provider.height
        self.batch_size = batch_size
        self.num_image = len(self.images)
        
        # 在每个png中，有效像素的位置都在中间
        self.precrop_iters = 500    # 前500轮进行procrop，让模型快速收敛 --> warmup
        self.precrop_frac = 0.5     # 取中间部分来训练
        self.niter = 0              # 用来控制procrop的循环次数
        
        self.device = device
        self.initialize()           # 初始化操作
        
    def initialize(self):
        """
        生成每一张画布上面的像素坐标，用来构建射线
        为了precrop，需要拿到一些中间像素的坐标
        """
        # 原画布的像素坐标
        warange = torch.arange(self.width, dtype=torch.float32, device=self.device)
        harange = torch.arange(self.height, dtype=torch.float32, device=self.device)
        # 使用 meshgrid 生成 x 和 y
        y, x = torch.meshgrid(harange, warange)
        
        # 将像素坐标转换为相机坐标 -- 相似三角形 通过将坐标中心移动到图像中心并按焦距进行归一化
        self.transformed_x = (x - self.width * 0.5) / self.focal
        self.transformed_y = (y - self.height * 0.5) / self.focal
        
        # 给整张图建立索引
        # 创建一个从 0 到 self.width * self.height - 1 的张量，并将其重塑为图像的高宽形状，表示每个像素的索引
        self.precrop_index = torch.arange(self.width * self.height).view(self.height, self.width)
        
        # 偏移量
        dH = int(self.height // 2 * self.precrop_frac)
        dW = int(self.width // 2 * self.precrop_frac)
        
        # 从 precrop_index 中提取出中央区域的索引，并将其重塑为一维张量，以用于后续的裁剪操作。
        self.precrop_index = self.precrop_index[
            self.height // 2 - dH : self.height // 2 + dH,
            self.width // 2 - dW  : self.width // 2 + dW
        ].reshape(-1)
        
        # 将poses转为Tensor
        # poses = torch.FloatTensor(self.poses, device=self.device)
        poses = torch.FloatTensor(self.poses).to(self.device)
        
        # 取出构造光线所需的像素和原点，采样，加权求和，得到像素
        all_ray_dirs, all_ray_origins = [], []
        
        for i in range(len(self.images)):
            ray_dirs, ray_origins = self.make_rays(self.transformed_x, self.transformed_y, poses[i]) # 将相机坐标传入，得到世界坐标系下的坐标和原点
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)
            
        # 使用stack转为tensor
        # dim=0: 按批次维度堆叠，也就是按照片的个数来堆叠在一起
        self.all_ray_dirs = torch.stack(all_ray_dirs, dim=0) 
        self.all_ray_origins = torch.stack(all_ray_origins, dim=0)
        # self.images = torch.FloatTensor(self.images, device=self.device).view(self.num_image, -1, 3) # 把宽高放到一个维度，就可以直接拿来索引了。其中 最后一维的 3 代表 RGB 值
        self.images = torch.FloatTensor(self.images).to(device).view(self.num_image, -1, 3)
        # precrop_index 的索引是宽高乘积，而上面的操作把宽高放到一个维度，就可以让两者对应起来
        
        
        
        pass
    
    # 返回每个照片里所有的像素在世界坐标系下的坐标点，每张照片相机坐标系的原点在世界坐标系下的表示
    def make_rays(self, x, y, pose):
        # 之前的坐标系是COLMAP的，需要进行转换
        directions = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)
        camera_matrix = pose[:3, :3]
        
        # 相机坐标系转世界坐标系
        ray_dirs = directions.reshape(-1, 3) @ camera_matrix.T
        
        # 平移向量就是原点
        ray_origin = pose[:3, 3].view(1, 3).repeat(len(ray_dirs), 1)
        
        return ray_dirs, ray_origin
        


if __name__ == '__main__':
    root = 'data/nerf_synthetic/lego'
    transforms_file = 'transforms_train.json'
    half_resolution = False

    provider = DatasetProvider(root, transforms_file, half_resolution)
    batch_size = 1024
    device     = 'cuda:0'       
    trainset = NeRFDataset(provider, batch_size, device)
    pass