# 姓 名 ：FangHao
# 开发时间：2024/7/18 9:14
#1)设计模型（输入，输出，通过不同layers向前）
#2)构造 loss 和 optimizer
#3)训练：
#       向前：计算预测和损失
#       向后：计算梯度
#       更新

import torch
import torch.nn as nn

#线性回归
#f = w * x
#f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')



learning_rate = 0.01
n_iters = 100
#引用了 torch.nn 直接用包里的MSELoss
loss = nn.MSELoss()
#优化器设置 SGD = Stochastic Gradient Descent 随机梯度下降 [w]可以放所有的参数，lr为学习率
optimizer = torch.optim.SGD([w], lr = learning_rate)

for epoch in range(n_iters):
    y_predicted = forward(X)

    l = loss(Y, y_predicted)

    l.backward()
    #更新 optimizer
    optimizer.step()
    # optimizer 清零
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print('epoch ', epoch + 1, ': w = ', w, ' loss = ', l)

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')