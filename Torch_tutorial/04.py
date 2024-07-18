# 姓 名 ：FangHao
# 开发时间：2024/7/17 17:08
import torch
x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

y_predicted = w * x
loss = (y_predicted - y) ** 2
print(loss)

loss.backward()
print(w.grad)


with torch.no_grad():
    w -= 0.01 * w.grad

w.grad.zero_()