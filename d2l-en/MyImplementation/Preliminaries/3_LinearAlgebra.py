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

print(f"A.mean() = {A.mean()}")
print(f"A.sum() / A.numel() = {A.sum() / A.numel()}")
print(f"A.mean(axis=0) = {A.mean(axis=0)}")
print(f"A.mean(axis=0) / A.shape[0] = {A.mean(axis=0) / A.shape[0]}")

# 7. 求和（不用降维）
print("-" * 100)
A = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
sum_A = A.sum(axis=1, keepdim=True)
print(f"A = {A}")
print(f"sum_A = {sum_A}")
print(f"sum_A.shape = {sum_A.shape}")
print(f"A.sum(axis=[0, 1], keepdim=True) = {A.sum(axis=[0, 1], keepdim=True)}")
print(f"A / A.sum(axis=0, keepdim=True) = {A / A.sum(axis=0, keepdim=True)}")
print(f"A.cumsum(axis=0) = {A.cumsum(axis=0)}")

# 8. 点积
print("-" * 100)
x = torch.randn(3)
y = torch.ones(3, dtype=torch.float32)
print(f"x = {x}, y = {y}")
print(f"torch.dot(x, y) = {torch.dot(x, y)}")
print(f"x.dot(y) = {x.dot(y)}")
print(f"torch.sum(x * y) = {torch.sum(x * y)}")

# 9. 矩阵-向量积
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)
print(f"A = {A}")
print(f"x = {x}")
print(f"A.shape = {A.shape}")
print(f"x.shape = {x.shape}")
print(f"A.shape[1] = {A.shape[1]}")
print(f"A.shape[1] == x.shape[0] = {A.shape[1] == x.shape[0]}")
print(f"A.matmul(x) = {A.matmul(x)}")
print(f"A @ x = {A @ x}")
print(f"torch.mv(A, x) = {torch.mv(A, x)}")

# 10. 矩阵-矩阵积
print("-" * 100)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.ones(4, 3, dtype=torch.float32)
print(f"A = {A}")
print(f"B = {B}")
print(f"A.shape = {A.shape}")
print(f"B.shape = {B.shape}")
print(f"A.shape[1] = {A.shape[1]}")
print(f"A.shape[1] == B.shape[0] = {A.shape[1] == B.shape[0]}")
print(f"A.matmul(B) = {A.matmul(B)}")
print(f"A @ B = {A @ B}")
print(f"torch.mm(A, B) = {torch.mm(A, B)}")

# 11. 范数
print("-" * 100)
u = torch.tensor([3.0, -4.0])
print(f"u = {u}")
print(f"torch.norm(u) = {torch.norm(u)}")
print(f"torch.abs(u).sum() = {torch.abs(u).sum()}")
A = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print(f"A = {A}")
print(f"torch.norm(A) = {torch.norm(A)}")
