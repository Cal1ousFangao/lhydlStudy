# 姓 名 ：FangHao
# 开发时间：2024/7/19 9:12

#dataset & dataloader
#dataset 就是读取数据，获取sample，知道sample个数，
#dataloader包含dataset，还有一些data的设置,batchsize，shuffle
#导入包
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
#定义一个dataset
class WineDataset(Dataset):
    def __init__(self): #定义自己  主要是读取数据,涉及numpy转torch
        #delimiter 分隔符        skiprows 跳过第几行，一般第一行为注释
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows = 1) #因为data是从numpy里来的，所以是np.loadtxt

        self.n_samples = xy.shape[0] #定义在类中，直接可以从numpy转换为tensor
        self.x_data = torch.from_numpy(xy[:, 1:])
        self.y_data = torch.from_numpy(xy[:, [0]])

    def __getitem__(self, index): #dataset中如果要看目前到哪个sample，就得定义读取
        return self.x_data[index], self.y_data[index]

    def __len__(self):  #这是方便看有多少个sample
        return self.n_samples


dataset = WineDataset() #生成一个dataset

first_data = dataset[0] #读取第一个sample
features, labels = first_data
print(features, labels) #看看sample读取的正确与否

#定义训练loader
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle = True, #随机打乱是否开启
                          num_workers=2)  #类似于效率，工人越多效率越高


dataiter = iter(train_loader)  #生成一个迭代器
data = next(dataiter) #指向下一个，注意，这里data是train_loader中的第一个不是第二个
features, labels = data
print(features, labels)
'''
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4) 

for epoch in range(num_epochs):
    for i, (input, labels) in enumerate(train_loader):

        #train

        if (i + 1) % 5 == 0:
            print(
                f'Epoch: {epoch + 1}/{num_epochs}, Step {i + 1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
'''