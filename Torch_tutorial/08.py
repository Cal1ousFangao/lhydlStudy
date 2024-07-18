# 姓 名 ：FangHao
# 开发时间：2024/7/18 9:31

#Linear regression

import torch
import torch.nn as nn
#输入数据
X = torch.tensor([[1], [2], [3], [4]],dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]],dtype=torch.float32)
#样本和特征数目
n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')
#预测X = 5的值
X_test = torch.tensor([5], dtype=torch.float32)
#线性回归问题，输入输出数目均为 1
input_size = n_features
output_size = n_features
#取出nn中的一个线性回归模型 y = wx + b
model = nn.Linear(input_size,output_size)

'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
'''

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

learning_rate = 0.01
n_iters = 100
#定义loss
loss = nn.MSELoss()
#定义optimizer
optimizer = torch.optim.SGD(model.parameters(), learning_rate)
# 训练
for epoch in range(n_iters):
    #之前的前向
    y_predicted = model(X)
    #损失
    l = loss(Y, y_predicted)
    #找回函数
    l.backward()
    #optimizer 迭代
    optimizer.step()
    #梯度清零
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()  # unpack parameters
        print('epoch ', epoch + 1, ': w = ', w[0][0].item(), ' loss = ', l)

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')