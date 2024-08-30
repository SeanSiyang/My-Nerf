- NeRF 官网: https://www.matthewtancik.com/nerf
- NeRF 官方 GitHub: https://github.com/bmild/nerf

# vscode Debug设置

修改launch.json文件，在里面添加下面的内容：
```json
{
    "cwd": "${fileDirname}",
}
```
将${fileDirname}改为要调试的目录

# Error

TypeError("'DatasetProvider' object is not subscriptable")

使用的变量不可以使用下标

# dataset

在深度学习的图像处理中，“precrop”指的是在图像进行主要处理或训练之前，对图像进行裁剪。这通常用于以下目的：

去除不相关区域：裁剪掉图像中不需要的部分，以减少数据噪声和计算量。
统一图像大小：将所有图像裁剪为统一的尺寸，以适应模型的输入要求。
增强训练数据：通过裁剪不同的区域生成更多的训练样本，从而增强模型的泛化能力。