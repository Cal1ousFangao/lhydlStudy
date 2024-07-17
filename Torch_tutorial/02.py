# 姓 名 ：FangHao
# 开发时间：2024/7/17 15:22
'''tensor basic'''
import torch
import numpy as np
'''tensor 即多维数组
1d -> scalar
2d -> vector
3d -> matrix
higher -> tensor
'''

#torch.empty(size) 生成一个空的tensor
x = torch.empty(1) #scalar 1d
print(x)

x = torch.empty(3) #vector 2d
print(x)

x = torch.empty(2,3) # matrix 2r 3c
print(x)

x = torch.empty(2,2,3) # tensor 3d
print(x)

#torch.zeros(size)
#torch.ones()
#print(x.size()) check size
#print(x.dtype)  check data type

#将数据变为tensor
x = torch.tensor([5.5,3])
print(x.size())

#requires_grad 就是告诉torch后续你将计算这个tensor的梯度
x = torch.tensor([5.5,3], requires_grad = True)

#元素相加
y = torch.rand(2,2)
print(y)
x = torch.rand(2,2)
print(x)
print(torch.add(x, y)) #不会改变变量
print(y.add(x)) # 会改变变量

#减
print(torch.sub(x, y))
#乘
print(torch.mul(x, y))
#除
print(torch.div(x, y))


#切片
x = torch.rand(5,3)
print(x)
print(x[:,0])
print(x[1,:])
print(x[1,1])

#得到某位置上的值
print(x[1,1].item())

# 通过torch.view()来改变size
x = torch.randn(4,4)
print(x)
y = x.view(16)
print(y)
z = x.view(-1, 8) #给 -1 可以通过其他参数自适应算出
print(z)
print(x.size(), y.size(), z.size())


#torch 与 numpy
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
print(type(b))

#注意：如果tensor在cpu上，那么所有对象共用内存，改变一个就会改变全部


#numpy -> torch 用 .from_numpy(x)
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

#默认所有tensor都在cpu上创建，但是可以将他们移动到gpu上
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device = device) #这样是直接将tensor创建在gpu上了
    x = x.to(device)  #将 x 移动到gpu上
    z = x + y
    #z = z.numpy() 会报错，因为numpy无法操作位于 gpu 上的tensor
    z = z.to("cpu")
    z = z.numpy()
    print(z)
