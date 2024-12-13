import torch

# 1. introduction
x = torch.arange(4.0)
print(f"x = {x}")
x.requires_grad_(True)
print(f"x.requires_grad = {x.requires_grad}")
print(f"x.grad = {x.grad}")

y = 2 * torch.dot(x, x)
print(f"y = {y}")

y.backward()
print(f"x.grad = {x.grad}")
print(f"x.grad == 4 * x = {x.grad == 4 * x}")

x.grad.zero_()
y = x.sum()
y.backward()
print(f"y = {y}")
print(f"x.grad = {x.grad}")

# 2. 非标量变量的反向传播
print("-" * 100)
x = torch.arange(4.0, requires_grad=True)
print(f"x = {x}")

y = x * x
print(f"y = {y}")

# 为非标量y提供gradient参数进行反向传播
y.backward(gradient=torch.ones(len(y)))
print(f"x.grad = {x.grad}")

# 3. 分离计算
print("-" * 100)
# 演示如何将计算图分离
x = torch.arange(4.0, requires_grad=True)
y = x ** 2
u = y.detach()  # 从计算图中分离y
print(f"u = {u}")
z = u * x  # u被视为常数
print(f"z = {z}")
print(f"z.sum() = {z.sum()}")
z.sum().backward()  # 反向传播
print(f"x.grad = {x.grad}")  # 梯度等于u
print(f"x.grad == u = {x.grad == u}")

# 对比不分离时的梯度计算
x.grad.zero_()
y.sum().backward()
print(f"x.grad = {x.grad}")
print(f"x.grad == 2 * x = {x.grad == 2 * x}")

# 4. Python控制流的梯度计算
print("-" * 100)
# 在控制流中计算梯度
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# 测试控制流中的梯度计算
a = torch.randn(size=(), requires_grad=True)
print(f"a = {a}")
d = f(a)
print(f"d = {d}")
d.backward()
print(f"a.grad = {a.grad}")
print(f"a.grad == d / a = {a.grad == d / a}")

# test
print("-" * 100)
# 测试梯度计算的基本操作
x = torch.arange(4.0, requires_grad=True)
print(f"x = {x}")
y = x * x
print(f"y = {y}")
y.sum().backward()
print(f"x.grad = {x.grad}")
# 注意：不能连续两次反向传播
# y.sum().backward()
# print(f"x.grad = {x.grad}")
