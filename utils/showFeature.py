# 测试
# 开发时间：2023/3/9 19:17
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import models

import torchvision.transforms as transforms
import torchvision.datasets as dataset

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from models.basicblock import Conv_Attention
# 准备测试所用的模型
# modelVGG = models.vgg16(pretrained=True)  # 采用VGG16的预训练模型
# print(modelVGG)

# dual attention unit
class testModel(nn.Module):
    def __init__(self, dim, bias=True):
        super(testModel, self).__init__()
        self.LR_act = torch.nn.SiLU(inplace=True)  # SiLU
        self.LRtoL3_1_conv3 = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.LRtoL3_2_conv1_1 = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.LRtoL3_2_conv1_2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.LRtoL3_3_dw3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=3, stride=1, padding=1,
                                      groups=int(dim * 2 ** 2), dilation=1, bias=bias)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # to L2
        x = self.LRtoL3_1_conv3(x)
        lr_x_tol3 = self.LRtoL3_2_conv1_1(x)
        lr_x_tol3 = self.LR_act(lr_x_tol3)
        lr_x_tol3 = self.LRtoL3_2_conv1_2(lr_x_tol3)
        lr_x_tol3 = self.LRtoL3_3_dw3(lr_x_tol3)

        return lr_x_tol3

model1 = testModel(48)

# 准备测试图像
img = cv.imread("./18.bmp")  # 读取本地一张图片
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()  # 展示原始测试图像

# 准备测试图像转化函数，因为只有将测试图像转化为tensor形式，才可以进行测试
transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 测试图像由3D形式扩展为 [samples, rows, cols, channels]的4D数组
img = np.array(img)
img = transform(img)
img = img.unsqueeze(0)
print(img.size())

# 接下来就需要访问模型所有的卷积层
no_of_layers = 0
conv_layers = []

model_children = list(model1.children())

for child in model_children:
    if type(child) == nn.Conv2d:
        no_of_layers += 1
        conv_layers.append(child)
    elif type(child) == nn.Sequential:
        for layer in child.children():
            if type(layer) == nn.Conv2d:
                no_of_layers += 1
                conv_layers.append(layer)
print(no_of_layers)

# 将测试图像作为第一个卷积层的输入，使用for循环，依次将最后一个结果传递给最后一层卷积。
results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[-1]))
outputs = results

# 依次展示特征可视化结果
for num_layer in range(len(outputs)):
    plt.figure(figsize=(50, 10))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print("Layer ", num_layer + 1)
    for i, filter in enumerate(layer_viz):
        if i == 16:
            break
        plt.subplot(2, 8, i + 1)
        plt.imshow(filter, cmap='gray')  # 如果需要彩色的，可以修改cmap的参数
        # plt.axis("off")
        plt.ylabel('layer'+str(num_layer + 1))
    plt.show()
    plt.close()

r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    'jet' 表示使用一种常见的颜色映射方式，会将数值较小的地方映射为青色至蓝色，数值较大的地方映射为黄色至红色。
    如果你不需要彩色的效果，可以将 cmap 参数的值设置为 'gray'，它将会呈现出灰度图像。
    """