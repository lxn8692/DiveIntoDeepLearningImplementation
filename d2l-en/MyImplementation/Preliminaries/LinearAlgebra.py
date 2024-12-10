import torch

# 1. 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(f"x = {x}, y = {y}")
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")
print(f"x / y = {x / y}")
print(f"x ** y = {x ** y}")

# 2. 向量
print("-" * 100)
x = torch.arange(4)
print(f"x = {x}")
print(f"x[3] = {x[3]}")
print(f"len(x) = {len(x)}")
print(f"x.shape = {x.shape}")

# 3. 矩阵
A = torch.arange(20).reshape(5, 4)
print(f"A = {A}")
print(f"A.T = {A.T}")
print(f"A.shape = {A.shape}")
print(f"A.T.shape = {A.T.shape}")

# 4. 张量
print("-" * 100)
X = torch.arange(24).reshape(2, 3, 4)
print(f"X = {X}")
print(f"X.shape = {X.shape}")
print(f"X.numel() = {X.numel()}")

# 5. 张量运算的基本性质
print("-" * 100)
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()
print(f"A = {A}")
print(f"A + B = {A + B}")
print(f"A * B = {A * B}")

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(f"a + X = {a + X}")
print(f"(a * X).shape = {(a * X).shape}")

# 6. 降维
print("-" * 100)
x = torch.arange(4, dtype=torch.float32)
print(f"x = {x}")
print(f"x.sum() = {x.sum()}")

A = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print(f"A = {A}")
print(f"A.shape = {A.shape}")
print(f"A.sum() = {A.sum()}")
print(f"A.sum(axis=0) = {A.sum(axis=0)}")
print(f"A.sum(axis=0).shape = {A.sum(axis=0).shape}")
print(f"A.sum(axis=1) = {A.sum(axis=1)}")
print(f"A.sum(axis=1).shape = {A.sum(axis=1).shape}")
print(f"A.sum(axis=[0, 1]) = {A.sum(axis=[0, 1])}")
print(f"A.sum(axis=[0, 1]).shape = {A.sum(axis=[0, 1]).shape}")
print(f"A.sum(axis=[1, 2]) = {A.sum(axis=[1, 2])}")
print(f"A.sum(axis=[1, 2]).shape = {A.sum(axis=[1, 2]).shape}")
