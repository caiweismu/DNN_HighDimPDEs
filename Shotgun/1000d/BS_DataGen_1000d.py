import numpy as np
import torch

torch.set_default_dtype(torch.float32)

# FIXME Edit by user
TORCH_GPU_INDEX = 0
torch.set_default_device(f'cuda:{TORCH_GPU_INDEX}')

from einops import repeat

D = 1000
T = 1.0

idx = np.linspace(1, D, D)
sigma = 0.1 + 0.4 * idx / D
mu = -0.05
r = 0.05

def g(x):
    xmax = torch.max(x, dim=-1, keepdim=True).values
    return torch.maximum(xmax - torch.ones_like(xmax), torch.zeros_like(xmax)) * float(np.exp(-r * T))


def MC_solution(x, bs1 = 10000, bsm = 10000):
    tot_sum = 0
    x_mul = repeat(x, 'd -> m d', m=bs1)
    sigma = torch.tensor(sigma)
    for _ in range(bsm):
        expoval = sigma * torch.randn_like(sigma) * np.sqrt(T) + (mu - 0.5 * sigma * sigma) * T
        expval = g(x_mul * torch.exp(expoval))
        tot_sum += torch.mean(expval, dim=0)
    res = tot_sum / bsm
    return res.item()


X0 = torch.rand((1000, D)) * 0.2 + 0.9
X0 = torch.tensor(X0, dtype=torch.float32, device=f'cuda:{TORCH_GPU_INDEX}')

sol = []
for n in range(1000):
    x = X0[n, :]
    sol.append(MC_solution(x))


np.save(f'./BS_X0_f32.npy', X0.numpy(force=True))
np.save(f'./BS_Y0_f32.npy', np.array(sol))
