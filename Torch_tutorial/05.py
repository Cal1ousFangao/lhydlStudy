# 姓 名 ：FangHao
# 开发时间：2024/7/18 8:47
#利用numpy手动线性回归 f = w * x

#导入numpy
import numpy as np
#训练集导入
X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8],dtype=np.float32)
#此时训练目标的随机数
w = 0.0

#模型输出
def forward(x):
    return w * x

#loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# J = MSE = 1 / N * (w * x - y) ** 2
#dJ/dw = 1 / N * 2(w * x - y) * x
def gradient(x, y, y_pred):
    return np.mean(2 * x * (y_pred - y))

print(f'Prediction before training: f(5) = {forward(5):.3f}')

#训练
learning_rate = 0.01
n_iters = 20

#每轮
for epoch in range(n_iters):
    #先输出一个pred
    y_pred = forward(X)
    #计算loss
    l = loss(Y, y_pred)
    #计算梯度
    dw = gradient(X, Y, y_pred)
    #往梯度最小的地方移动
    w -= learning_rate * dw

    if epoch % 2 ==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')