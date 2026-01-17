import math
import warnings

import torch
import torch.distributed as dist
import numpy as np
from ex_meta import PDEwithVtrue, diag_curve, e1_curve, BVP, isin_ddp


def absorbing_path(x_start, te, num_dt, next_xt: callable):
    ''' 
    生成带吸收壁的随机过程路径，吸收壁位于单位球面上
    '''

    # 计算最大步数
    num_tn = num_dt + 1
    dt = te / (num_tn - 1)

    num_path = x_start.shape[0]
    dim_x = x_start.shape[-1]
    device = x_start.device

    # 初始化路径数组 (M, max_steps, d)
    xt = torch.zeros((num_path, num_tn, dim_x), device=device)
    active = torch.ones(num_path, dtype=torch.bool, device=device)  # 标记活跃路径
    abs_times = torch.full((num_path, ), float('nan'), device=device)  # 吸收时间
    abs_points = torch.zeros((num_path, dim_x), device=device)  # 吸收位置

    # 设置起点
    xt[:, 0, :] = x_start

    # 初始范数检查
    norms = torch.norm(x_start, dim=1)
    abs_at_start = torch.isclose(norms,
                                 torch.tensor(1.0, device=device),
                                 atol=1e-6)
    active[abs_at_start] = False
    abs_times[abs_at_start] = 0.0
    abs_points[abs_at_start] = x_start[abs_at_start]

    # 生成增量 (M, max_steps-1, d)
    dwt = torch.normal(0,
                       math.sqrt(dt),
                       size=(num_path, num_tn - 1, dim_x),
                       device=device)

    # 模拟过程
    for step in range(1, num_tn):
        if not torch.any(active):
            break

        # 只更新活跃路径
        act_mask = active
        xt_act = xt[act_mask, step - 1, :]
        dwt_act = dwt[act_mask, step - 1, :]
        xt[act_mask, step, :] = next_xt(step * dt, xt_act, dt, dwt_act)

        # 计算范数
        norms = torch.norm(xt[act_mask, step, :], dim=1)

        # 检查是否接触吸收壁
        hit_boundary = norms >= 1.0

        # 处理被吸收的路径
        if torch.any(hit_boundary):
            # 找到被吸收的路径索引
            abs_mask = act_mask.clone()
            abs_mask[act_mask] = hit_boundary

            # 记录吸收时间和位置
            abs_times[abs_mask] = step * dt
            abs_points[abs_mask] = xt[abs_mask, step, :]

            # 归一化到球面上（精确吸收点）
            for i in torch.where(abs_mask)[0]:
                abs_points[i] /= torch.norm(abs_points[i])
                xt[i, step, :] = abs_points[i]  # 确保在球面上

            # 标记为不活跃
            active[abs_mask] = False

    # 构建路径列表并截断
    xt_list = []
    for i in range(num_path):
        # 确定有效路径长度
        if not torch.isnan(abs_times[i]):
            last_step = int(abs_times[i] / dt) + 1
        else:
            last_step = num_tn

        # 截断路径
        xt_list.append(xt[i, :last_step, :].clone())

    return xt_list, abs_times, abs_points


class PBE1(PDEwithVtrue):

    name = 'PBE1'
    x0_for_train = {'S1': e1_curve, 'S2': diag_curve}
    x0pil_range = 1.

    musys_online = False
    sgmsys_online = False
    f_online = True

    w = 2.
    rad = 1.

    def __init__(self, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.te = 3.

    def make_nextxt_func(self, v_func=None):
        if v_func is None:

            def nextxt_func(t, x, dt, dwt):
                mu_t = self.mu_pil(t, x)
                sgm_dwt = self.sgm_pil(t, x, dwt)
                xt_next = x + mu_t * dt + sgm_dwt
                return xt_next
        else:

            def nextxt_func(t, x, dt, dwt):
                vtx = v_func(t, x)
                mu_t = self.mu_sys(t, x, vtx)
                sgm_dwt = self.sgm_sys(t, x, vtx, dwt)
                xt_next = x + mu_t * dt + sgm_dwt
                return xt_next

        return nextxt_func

    def gen_x0x1(self, x0: torch.Tensor, num_dt: int, v_func):
        # The generated path will be of shape (2, num_path, dim_x)
        # where the first element is x0 and the second is x0 + mu dt + sigma dwt.

        nextxt_func = self.make_nextxt_func(v_func)
        xt_list, _, _ = absorbing_path(x0, self.te, num_dt, nextxt_func)

        x_indom = torch.concat(xt_list, dim=0)
        dt = (self.te - self.t0) / num_dt
        dwt = torch.normal(0,
                           dt**(0.5),
                           size=(x_indom.shape[0], self.dim_w),
                           device=x_indom.device)
        x_next = nextxt_func(None, x_indom, dt, dwt)
        x_path = torch.stack([x_indom, x_next], dim=0)

        if v_func is None:
            v_path = None
        else:
            v_path = v_func(None, x_path)

        return x_path, v_path

    def gen_syspath(self, x0: torch.Tensor, num_dt: int, v_func):
        return self.gen_x0x1(x0, num_dt, v_func)

    def gen_pilpath(self, x0: torch.Tensor, num_dt: int):
        return self.gen_x0x1(x0, self.te, num_dt, None)[0]

    def path_projto_boundary(self, x_path):
        x_next = x_path[-1]
        xnext_norm = x_next.norm(dim=-1, keepdim=True)
        x_outball = (xnext_norm > self.rad).squeeze(-1)
        if x_outball.any():
            x_proj_ball = x_next[x_outball] / xnext_norm[x_outball] * self.rad
        else:
            x_proj_ball = None
            warnings.warn("No points projected to boundary", UserWarning)
        return x_proj_ball

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, _x, dw):
        return 2**(0.5) * dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, _x, _v, dw):
        return 2**(0.5) * dw

    def tr_sgm2vxx(self, _t, _x, _v, vxx):
        return 2 * torch.einsum('...ii->...', vxx)

    def f(self, _t, x, v):
        v_true = torch.cos(self.w * x).sum(dim=-1, keepdim=True)
        fval = torch.sinh(v) - torch.sinh(v_true)
        cos_wx = torch.cos(self.w * x).sum(dim=-1, keepdim=True)
        fval = fval + self.w**2 * cos_wx
        return fval

    def v(self, _t, x):
        return torch.cos(self.w * x).sum(dim=-1, keepdim=True)

    def additional_loss(self, _t, x_path, v_approx):
        """
        Additional loss for the boundary condition.
        """
        x_ball = self.path_projto_boundary(x_path)
        if x_ball is not None:
            vappr_val = v_approx(None, x_ball)
            vtrue_val = self.v(None, x_ball)
            loss_val = (vtrue_val - vappr_val).pow(2).mean()
        else:
            loss_val = torch.tensor(0.0, device=x_path.device)
        return loss_val

    def res_on_path(
        self,
        v_approx,
        ctrfunc_syspath=None,
        num_dt=50,
        num_path=8,
    ):
        return None


class AllenCahnBVP(BVP):
    name = 'Allen-Cahn-equation in unit ball'
    # This example is from subsection 5.4 of
    # @article{Hu2025BiasVariance,
    # author = {Hu, Zheyuan and Yang, Zhouhao and Wang, Yezhen and Karniadakis, George E. and Kawaguchi, Kenji},
    # title = {Bias-Variance Trade-Off in Physics-Informed Neural Networks with Randomized Smoothing for High-Dimensional PDEs},
    # journal = {SIAM Journal on Scientific Computing},
    # volume = {47},
    # number = {4},
    # pages = {C846-C872},
    # year = {2025},
    # doi = {10.1137/23M1621356},
    # }
    x0pil_range = 1.

    musys_online = False
    sgmsys_online = False
    f_online = True

    num_testpoint = 20000

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x
        assert dim_x >= 2, "dim_x should be at least 2"

        c_arr = np.random.default_rng(seed=1).normal(size=(dim_x - 1, ))
        c_tensor = torch.from_numpy(c_arr).to(self.t0.device).float()
        if isin_ddp():
            dist.broadcast(c_tensor, src=0)
        self.c = c_tensor
        # print(f'x_test_0:{c_tensor[0]}, rank: {c_tensor.device}')

        # print(f"c_0 = {self.c[0]}")

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, _x, dw):
        return 2**(0.5) * dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, _x, _v, dw):
        return 2**(0.5) * dw

    def tr_sgm2vxx(self, _t, _x, _v, vxx):
        return 2 * torch.einsum('...ii->...', vxx)

    def laplace_v(self, _t, x):
        cosx = torch.cos(x)
        sinx = torch.sin(x)
        x0 = x[..., :-1]
        x1 = x[..., 1:]
        cosx0 = cosx[..., :-1]
        cosx1 = cosx[..., 1:]
        sinx0 = sinx[..., :-1]
        sinx1 = sinx[..., 1:]

        theta = x0 + cosx1 + x1 * cosx0
        md = x0 * (1 - x1 * sinx0) + x1 * (cosx0 - sinx1)
        ml = torch.cos(theta) * (-x1 * cosx0 - cosx1) - torch.sin(theta) * (
            (1 - x1 * sinx0)**2 + (cosx0 - sinx1)**2)
        lap = -2 * self.dim_x * (self.c * torch.sin(theta)).sum(
            -1, keepdim=True) - 4 * (self.c * torch.cos(theta) * md).sum(
                -1, keepdim=True) + (1 - x.pow(2).sum(-1, keepdim=True)) * (
                    self.c * ml).sum(-1, keepdim=True)
        return lap

    def f(self, _t, x, v):
        v_exact = self.v(_t, x)
        return v - v**3 - self.laplace_v(_t, x) - v_exact + v_exact**3

    def v(self, _t, x):
        cos_x = torch.cos(x)
        x0 = x[..., :-1]
        x1 = x[..., 1:]
        y = self.c * torch.sin(x0 + cos_x[..., 1:] + x1 * cos_x[..., :-1])
        y = y.sum(-1, keepdim=True)
        return self.v_shellfunc(x, y)


class SineGordonBVP(AllenCahnBVP):
    name = 'Sine-Gordon-equation-in-unit-ball'

    # This example is from subsection 5.4 of
    # @article{Hu2025BiasVariance,
    # author = {Hu, Zheyuan and Yang, Zhouhao and Wang, Yezhen and Karniadakis, George E. and Kawaguchi, Kenji},
    # title = {Bias-Variance Trade-Off in Physics-Informed Neural Networks with Randomized Smoothing for High-Dimensional PDEs},
    # journal = {SIAM Journal on Scientific Computing},
    # volume = {47},
    # number = {4},
    # pages = {C846-C872},
    # year = {2025},
    # doi = {10.1137/23M1621356},
    # }

    def f(self, _t, x, v):
        v_exact = self.v(_t, x)
        return torch.sin(v) - self.laplace_v(_t, x) - torch.sin(v_exact)


class AllenCahnBVPLessTestPt(AllenCahnBVP):
    num_testpoint = 100


class SineGordonBVPLessTestPt(SineGordonBVP):
    num_testpoint = 100
