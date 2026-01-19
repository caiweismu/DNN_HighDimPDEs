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

import numpy as np

D = 1000
T = 1.0

TEST_NAME = f'HJB_Rosenbrock_D{D}_DeepBSDE'

save_idx = 0
while os.path.exists(rf'./result_{TEST_NAME}_{save_idx}'):
    save_idx += 1
save_dir = rf'./result_{TEST_NAME}_{save_idx}/'
os.makedirs(save_dir)
shutil.copy2(__file__, save_dir + 'script.py')


M = 100 # num of paths
N = 100 # num of time snapshots

lr = 1e-4

embed_terminal = True
acti = torch.nn.functional.mish

dt = T / N

class DNN(nn.Module):
    def __init__(self, n_dim: int = 1000 + 1, n_out: int = 1, width_int: int = 1024):
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


class DeepBSDENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Y0net = DNN(D, 1)
        self.Ztnet = DNN(D + 1, D)

    def Y0(self, X0: Tensor) -> Tensor:
        return self.Y0net(X0)
    
    def Zt(self, t: Tensor, Xt: Tensor) -> Tensor:
        tx = torch.cat([t, Xt], dim=-1)
        return self.Ztnet(tx)


dnns = DeepBSDENet()

c = np.load('./HJB_ci.npy') # (D, 2)
c = torch.tensor(c, dtype=torch.float32) # (D, 2)


def g(x):
    xp1 = torch.roll(x, -1, dims=-1)
    return torch.log(0.5 + 0.5 * torch.sum(c[:, 0] * (xp1 - x) * (xp1 - x) + c[:, 1] * xp1 * xp1, dim=-1, keepdim=True))

mu = 0
sigma = np.sqrt(2.0)


def phi(z):
    return torch.sum(z * z, dim=-1, keepdim=True)


def fetch_Wt():
    Dt = torch.zeros((M, N + 1, 1))
    DW = torch.zeros((M, N + 1, D))
    Dt[:, 1:, :] = dt
    DW[:, 1:, :] = np.sqrt(dt) * torch.randn((M, N, D))

    t = torch.cumsum(Dt, dim=1)
    W = torch.cumsum(DW, dim=1)
    return t, W


def loss_function(t, W):
    loss = torch.zeros(1)
    t0 = t[:, 0, :]
    W0 = W[:, 0, :]
    X0 = torch.randn((M, D))
    X0.requires_grad = True
    Y0: Tensor = dnns.Y0(X0)
    Z0: Tensor = dnns.Zt(t0, X0)

    for n in range(N):
        t1 = t[:, n + 1, :]
        W1 = W[:, n + 1, :]

        mu0 = mu
        X1 = X0 + mu0 * (t1 - t0) + sigma * (W1 - W0)
        Y1 = Y0 + phi(Z0) * (t1 - t0) + sigma * torch.sum(Z0 * (W1 - W0), dim=-1, keepdim=True)
        Z1 = dnns.Zt(t1, X1)

        t0 = t1
        W0 = W1
        X0 = X1
        Y0 = Y1
        Z0 = Z1

    loss = torch.sum((Y0 - g(X0)) ** 2)

    return loss / M


def check():
    X0_ref = np.load('HJB_X0_f32.npy')
    Y0_ref = np.load('HJB_Y0_f32.npy')
    X0 = torch.tensor(X0_ref, dtype=torch.float32)
    Y0 = dnns.Y0(X0)[:, 0].numpy(force=True)

    err_inf = np.max(np.abs(Y0 - Y0_ref)) / np.max(np.abs(Y0_ref))
    err_l2 = np.linalg.norm(Y0 - Y0_ref) / np.linalg.norm(Y0_ref)
    return err_inf, err_l2


def train(N_iter):
    optimizer = torch.optim.Adam(dnns.parameters(), lr=lr)
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

torch.save(dnns.state_dict(), save_dir + rf'network')
