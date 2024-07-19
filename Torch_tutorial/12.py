# 姓 名 ：FangHao
# 开发时间：2024/7/19 10:14
#transform


import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


# 自定义数据集类
class WineDataset(Dataset):
    def __init__(self, transform=None): #overwrite transform
        # 加载数据
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # 分割数据
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]

        # 可选的变换
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


# 定义转换类 ToTensor
class ToTensor:
    def __call__(self, sample): # call
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


# 定义转换类 MulTransform
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


# 无转换测试
print("Without Transform")
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

# 使用 ToTensor 转换测试
print('\nWith Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

# 组合使用 ToTensor 和 MulTransform 转换测试
print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)]) #组合用torchvision.tansforms.composed
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)
