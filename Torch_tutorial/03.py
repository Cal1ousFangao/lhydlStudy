# 姓 名 ：FangHao
# 开发时间：2024/7/17 15:59
import torch
x = torch.randn(3,requires_grad=True)
y =  x  + 2

print(x)
print(y)
print(y.grad_fn) # grad_fn 是一个指向“Function”对象的引用，记录了创建y的操作
y.backward(torch.tensor([1.0, 1.0, 1.0])) # 给个权重
print(x.grad)

z = y * y * 3
print(z)
z = z.mean()
print(z)

z.backward()
print(x.grad)  #dz/dx

x = torch.randn(3, requires_grad = True)
y = x * 2
for _ in range(10):
    y = y * 2
print(y)
print(y.shape)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
y.backward(v) # dy/dx = 2048    2048*v = x.grad
print(x.grad)

a = torch.randn(2, 2)
print(a.requires_grad) #默认为False
b = ((a * 3) / (a - 1))
print(b.grad_fn) #False状态下不会记录
a.requires_grad_(True)  # 改变
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# .detach():复制一个相同内容但是没有梯度计算功能的Tensor
a = torch.randn(2,2,requires_grad=True)
print(a.requires_grad) # True
b = a.detach()
print(b.requires_grad) # False

a = torch.randn(2,2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():  # 额外计算一个不需要梯度计算的
    print((a ** 2).requires_grad)



#下面这个例子说明了在Torch中梯度是累积的，需要手动清零以便下次计算新的梯度,非常重要
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights * 3).sum()  # 12 8.4 4.8
    model_output.backward() # 计算梯度

    print(weights.grad) #3333,3333,3333

    with torch.no_grad():
        weights -= 0.1 * weights.grad # 0.7 0.4 0.1

    weights.grad.zero_() # Torch中梯度会累计，需要手动清零便于下一次迭代

print(weights) # 0.1 0.1 0.1 0.1
print(model_output) # 4.8