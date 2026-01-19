import numpy as np
import torch

torch.set_default_dtype(torch.float32)

# FIXME Edit by user
# TORCH_GPU_INDEX = 0
# torch.set_default_device(f'cuda:{TORCH_GPU_INDEX}')

from einops import repeat

c = 1
D = 100

ci = np.random.uniform(0.5, 1.5, size=(D, 2))
np.save('./HJB_ci.npy', ci)

def g(x):
    c = torch.tensor(ci, dtype=torch.float32)
    xp1 = torch.roll(x, -1, dims=-1)
    return torch.log(0.5 + 0.5 * torch.sum(c[:, 0] * (xp1 - x) * (xp1 - x) + c[:, 1] * xp1 * xp1, dim=-1, keepdim=True))


def MC_solution(x, bs1 = 10000, bsm = 1000):
    tot_sum = 0
    x_mul = repeat(x, 'd -> m d', m=bs1)
    for _ in range(bsm):
        expval = torch.exp(-g(x_mul + np.sqrt(2) * torch.randn_like(x_mul)))
        tot_sum += -torch.log(torch.mean(expval))
    res = tot_sum / bsm
    return res.item()

X0 = torch.randn((1000, D))
sol = []
for n in range(1000):
    x = X0[n, :]
    sol.append(MC_solution(x))

np.save(f'HJB_X0_f32.npy', X0.numpy(force=True))
np.save(f'HJB_Y0_f32.npy', np.array(sol))
