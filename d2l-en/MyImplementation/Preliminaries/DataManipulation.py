import torch


# 1. 创建张量
x = torch.arange(12, dtype=torch.bfloat16)
print(x)

print(f"x.shape = {x.shape}, x.numel() = {x.numel()}, len(x) = {len(x)}")

x = x.reshape(3, 4)
print(x)
print(f"after reshape: x.shape = {x.shape}, x.numel() = {x.numel()}, len(x) = {len(x)}")

zeros = torch.zeros((2, 3, 4))
print(zeros)

ones = torch.ones((2, 3, 4))
print(ones)

randns = torch.randn(3, 4)
print(randns)

lst_tensor = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(lst_tensor)


# 2. 索引和切片
print("-" * 100)
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(f"X = {X}")
print(f"X[-1] = {X[-1]}, X[-1, :] = {X[-1, :]}")
print(f"X[1:3] = {X[1:3]}")
print(f"X[1, 2] = {X[1, 2]}")
print(f"X[1:3, :2] = {X[1:3, :2]}")


# 3. 运算符
print("-" * 100)
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(f"torch.exp(X) = {torch.exp(X)}")

a = torch.tensor([1, 2, 4, 8])
b = torch.tensor([2, 2, 2, 2])
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** b = {a ** b}")

a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
b = torch.arange(12, 36, dtype=torch.float32).reshape(3, -1)
c = torch.arange(12, 36, dtype=torch.float32).reshape(-1, 4)
print(f"torch.cat((a, b), dim=1) = {torch.cat((a, b), dim=1)}")
print(f"torch.cat((a, c), dim=0) = {torch.cat((a, c), dim=0)}")
print(f"a == torch.ones(a.shape) = {a == torch.ones(a.shape)}")
print(f"a.sum() = {a.sum()}")


# 4. 广播机制
print("-" * 100)
a = torch.arange(3).reshape((3, 1))
b = torch.arange(3).reshape((1, 3))

print(f"a = {a},\nb = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")


# 5. 节省内存
print("-" * 100)
Y = torch.randn(3, 4)
before_id = id(Y)
Y = Y + X
print(f"id(Y) = {id(Y)}, \nid(before_id) = {before_id}")

Z = torch.zeros_like(Y)
print(f"id(Z) = {id(Z)}")
Z[:] = Z + Y
print(f"id(Z) = {id(Z)}")

Z += Y
print(f"id(Z) = {id(Z)}")

# 6. 转换为其他Python对象
print("-" * 100)
A = X.numpy()
B = torch.from_numpy(A)
print(f"A = {A}, \nB = {B}")

tmp = torch.tensor([3.5])
print(f"tmp = {tmp}, \ntmp.item() = {tmp.item()}")
