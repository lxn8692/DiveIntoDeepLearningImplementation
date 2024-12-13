import torch
import random
from torch.distributions import Multinomial
from d2l import torch as d2l

# 1. 投硬币
num_tosses = 1000
heads = sum([random.random() > 0.5 for i in range(num_tosses)])
tails = num_tosses - heads
print(f"heads, tails = [{heads}, {tails}]")

fair_probs = torch.tensor([0.5, 0.5])
print(f"Multinomial(100, fair_probs).sample() = {Multinomial(1000, fair_probs).sample()}")

counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label="P(coin=heads)")
d2l.plt.plot(estimates[:, 1], label="P(coin=tails)")
d2l.plt.axhline(y=0.5, color='black', linestyle="dashed")
d2l.plt.gca().set_xlabel("Samples")
d2l.plt.gca().set_ylabel("Estimated probability")
d2l.plt.legend()
d2l.plt.show()
