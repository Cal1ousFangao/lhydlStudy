# 姓 名 ：FangHao
# 开发时间：2024/7/18 8:58
#利用torch 自动回归 y = w * x
import torch
#数据导入
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
'''关键，被求梯度的对象，梯度允许'''
w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)
#模型输出
def forward(x):
    return w * x

#loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# 注意 torch中就不用像numpy那样自己求导写出公式了
print(f'Prediction before training: f(5) = {forward(5).item():.3f}')


#训练
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)
    '''关键，l是通过一个表达式算来的，backward就是去找回这个表达式'''
    l.backward()
    #with torch.no_grad 就是这部分代码不需要被跟踪，如果没写，那么下个循环 l.backward 就会包含w -= lea....这条代码
    with torch.no_grad():
        w -= learning_rate * w.grad #这里的 w.grad 就是找回的表达式对 w 求导
    '''关键，梯度算完清零'''
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w.item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')