import torch
torch.manual_seed(46701)
import torch.nn as nn
from torch import Tensor

import os
import sys
import shutil

torch.set_default_dtype(torch.float32)
try:
    TORCH_GPU_INDEX = int(sys.argv[1])
except:
    TORCH_GPU_INDEX = 0 # FIXME
torch.set_default_device(f'cuda:{TORCH_GPU_INDEX}')
print(f'Using GPU cuda:{TORCH_GPU_INDEX}')

import numpy as np
from einops import repeat


D = 100
T = 1.0

TEST_NAME = f'BS_D{D}_Shotgun'

save_idx = 0
while os.path.exists(rf'./result_{TEST_NAME}_{save_idx}'):
    save_idx += 1
save_dir = rf'./result_{TEST_NAME}_{save_idx}/'
os.makedirs(save_dir)
shutil.copy2(__file__, save_dir + 'script.py')


M = 100 # num of paths
N = 20 # num of time snapshots
M1 = 32 # num of local random batch
dt1 = 1e-5 # local dt

lr = 1e-4

embed_terminal = True
terminal_weight = 1.0
acti = torch.nn.functional.mish

dt = T / N
sqdt = np.sqrt(dt)
sqdt1 = np.sqrt(dt1)
idx = torch.linspace(1, D, D)
sigma = 0.1 + 0.4 * idx / D
mu = -0.05
r = 0.05

class DNN(nn.Module):
    def __init__(self, n_dim: int = D + 1, n_out: int = 1, width_int: int = 1024):
        super().__init__()
        self.fc_1 = nn.Linear(n_dim, width_int)
        self.fc_2 = nn.Linear(width_int, width_int)
        self.fc_3 = nn.Linear(width_int, width_int)
        self.fc_4 = nn.Linear(width_int, width_int)
        self.out = nn.Linear(width_int, n_out)

    def forward(self, state: Tensor) -> Tensor:
        state = acti(self.fc_1(state))
        state = acti(self.fc_2(state))
        state = acti(self.fc_3(state))
        state = acti(self.fc_4(state))
        fn_u = self.out(state)
        return fn_u

dnn = DNN(n_dim=D + 1, n_out=1)


def g(x):
    xmax = torch.max(x, dim=-1, keepdim=True).values
    return torch.maximum(xmax - torch.ones_like(xmax), torch.zeros_like(xmax)) * float(np.exp(-r * T))

phi = 0


def embed_weight(t):
    return torch.maximum(torch.zeros_like(t), (N * t / T - N + 1) ** 3)


def net_u(t, x) -> Tensor:
    state = torch.cat([t, x], dim=-1)
    val = dnn(state)
    if embed_terminal:
        ewt = embed_weight(t)
        return (1 - ewt) * val + ewt * g(x)
    else:
        return val


def fetch_Wt():
    Dt = torch.zeros((M, N + 2, 1))
    DW = torch.zeros((M, N + 2, D))
    dt_first = torch.rand((M, 1)) * dt
    Dt[:, 1, :] = dt_first
    Dt[:, 2:, :] = dt
    Dt[:, -1, :] = dt - dt_first
    DW[:, 1, :] = torch.sqrt(dt_first) * torch.randn((M, D))
    DW[:, 2:-1, :] = np.sqrt(dt) * torch.randn((M, N - 1, D))
    DW[:, -1, :] = torch.sqrt(dt - dt_first) * torch.randn((M, D))

    t = torch.cumsum(Dt, dim=1)
    W = torch.cumsum(DW, dim=1)
    return t, W


def loss_function(t, W):
    loss = torch.zeros(1)
    t0 = t[:, 0, :]
    W0 = W[:, 0, :]
    X0 = torch.rand((M, D)) * 0.2 + 0.9
    X0.requires_grad = True
    Y0 = net_u(t0, X0)

    for n in range(N + 1):
        t1 = t[:, n + 1, :]
        W1 = W[:, n + 1, :]

        mu0 = mu * X0
        X1 = X0 + mu0 * (t1 - t0) + (W1 - W0) * (X0 * sigma)

        dWt = torch.randn((M, M1, D)) * np.sqrt(dt1)
        X1mean = repeat(X0 + mu0 * dt1, 'n d -> n m d', m=M1)
        t1mean = repeat(t0 + dt1, 'n d -> n m d', m=M1)
        diffusion = dWt * repeat((X0 * sigma), 'n d -> n m d', m=M1)
        Y1_loc0 = net_u(t1mean, X1mean + diffusion)
        Y1_loc1 = net_u(t1mean, X1mean - diffusion)
        loss_val = (0.5 * torch.mean(Y1_loc0, dim=1) + 0.5 * torch.mean(Y1_loc1, dim=1) - Y0) / dt1 - phi

        loss = loss + torch.sum(loss_val * loss_val)

        t0 = t1
        W0 = W1
        X0 = X1
        Y0 = net_u(t0, X0)

    if not embed_terminal:
        loss = loss + torch.sum((Y0 - g(X0)) ** 2) * terminal_weight

    return loss / M


def check():
    X0_ref = np.load('./BS_X0_f32.npy')
    Y0_ref = np.load('./BS_Y0_f32.npy')

    Mc = 1000
    X0 = torch.tensor(X0_ref, dtype=torch.float32)
    t0 = torch.zeros((Mc, 1))
    Y0 = net_u(t0, X0)[:, 0].numpy(force=True)

    err_inf = np.max(np.abs(Y0 - Y0_ref)) / np.max(np.abs(Y0_ref))
    err_l2 = np.linalg.norm(Y0 - Y0_ref) / np.linalg.norm(Y0_ref)
    return err_inf, err_l2


def train(N_iter):
    optimizer = torch.optim.Adam(dnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.1)
    try:
        for it in range(N_iter):
            t, W = fetch_Wt()
            loss = loss_function(t, W)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    except KeyboardInterrupt:
        pass

train(5000)

torch.save(dnn.state_dict(), save_dir + rf'network')
