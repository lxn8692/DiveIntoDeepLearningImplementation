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
print(f"x.grad.zero_() = {x.grad.zero_()}")

