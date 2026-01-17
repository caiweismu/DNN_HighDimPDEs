# Define metaclass for PDEs and related functions
import abc
import gc
from abc import abstractmethod
from typing import Sequence, Union

import psutil
import torch
import torch.distributed as dist
from torch.func import grad_and_value, jacrev

from sav_res_via_exmeta import plot_landscape, res_on_curve_ver2, tensor2ndarray, v_histogram


def isin_ddp():
    return dist.is_available() and dist.is_initialized()


def get_interval(s_range: Union[float, Sequence[float]]):
    if isinstance(s_range, Sequence) is True:
        left_end, right_end = s_range
    else:
        left_end = -s_range
        right_end = s_range
    return left_end, right_end


def e1_curve(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    e_1stcoord = torch.zeros([dim_x])
    e_1stcoord[0] = 1.
    xe1 = torch.outer(s_coord, e_1stcoord)
    return s_coord, xe1


def diag_curve(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    e_diag = torch.ones((dim_x, ))
    s_coord = torch.linspace(left_end, right_end, num_points)
    xdiag = torch.outer(s_coord, e_diag)
    return s_coord, xdiag


def manifold_curve(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    e_vec = torch.arange(1, dim_x + 1)

    x_diag = torch.outer(s_coord, torch.sign(torch.sin(e_vec)))
    x = x_diag + torch.cos(e_vec + s_coord.unsqueeze(-1) * torch.pi)
    return s_coord, x


def origin_point(dim_x, num_points):
    s_coord = torch.zeros([num_points])
    x = torch.zeros([num_points, dim_x])
    return s_coord, x


def standard_normal(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
):
    # s_range has no effect on the returned points `x`
    x = torch.randn([num_points, dim_x])
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    return s_coord, x


def cube(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    x = torch.rand([num_points, dim_x]) * (right_end - left_end) + left_end
    return s_coord, x


def unit_ball(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
    uniform_radius: bool = True,
):
    # s_range has no effect on the returned points `x`
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)

    x = torch.randn([num_points, dim_x])
    epsilon = torch.finfo(x.dtype).eps * 100
    norms = torch.clamp(torch.norm(x, dim=1), min=epsilon)
    u = torch.rand(num_points)
    if uniform_radius:
        radii = u
    else:
        radii = u**(1.0 / dim_x)
    x = x / norms.unsqueeze(-1)
    x = x * radii.unsqueeze(-1)
    return s_coord, x


def free_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'xpu':
        torch.xpu.empty_cache()
    elif device.type == 'cpu':
        gc.collect()
    else:
        raise ValueError('The device type is not supported.')


def get_safe_chunksize(num_item_persize: int,
                       dtype: torch.dtype,
                       device: torch.device,
                       use_percent=0.5):
    # Determine a safe Monte Carlo chunksize that fits in device memory (avoid OOM).
    # num_item_persize: how many items (in dtype) per chunksize

    if device.type == 'cuda':
        avail_mem = torch.cuda.mem_get_info(device)[0]
    elif device.type == 'xpu':
        avail_mem = torch.xpu.mem_get_info(device)[0]
    elif device.type == 'cpu':
        avail_mem = psutil.virtual_memory().available
    else:
        raise ValueError('The device type is not supported.')

    safe_chunksize = avail_mem * use_percent // (num_item_persize *
                                                 dtype.itemsize)
    safe_chunksize = int(safe_chunksize)
    if safe_chunksize == 0:
        raise MemoryError(
            f'The memory of {device} is not enough to get safe_chunksize >= 1.'
        )
    return safe_chunksize


def dt_dx_dxx_single(func, t, x):
    # t shape: [1]
    # x shape: [dim_x]
    # func: (t, x) -> y, y shape: []
    # output: (\partial_t, \partial_x, \partial_xx) func(t, x), with shape [], [dim_x], [dim_x, dim_x]

    def wrap_func(_t, _x):
        y = func(_t, _x).squeeze()
        return y, y

    func_dy_y = grad_and_value(wrap_func, argnums=(0, 1), has_aux=True)

    def wrap_dy_y(_t, _x):
        (dydt, dydx), (y, _) = func_dy_y(_t, _x)
        return dydx, (y, dydt, dydx)

    func_ddy_y = jacrev(wrap_dy_y, argnums=1, has_aux=True)
    dydxx, (y, dydt, dydx) = func_ddy_y(t, x)
    return y, dydt, dydx, dydxx


def dt_dx_dxx(func, t, x):
    # t shape: [some batch dims, 1]
    # x shape: [some batch dims, dim_x]
    # func: (t, x) -> y, y shape: [some batch dims, 1]
    # output: y, dydt, dydx, dydxx with shape [some batch dims, 1], [some batch dims, 1, dim_x], [some batch dims, 1, dim_x, dim_x]

    req_grad_t = t.requires_grad
    req_grad_x = x.requires_grad
    t.requires_grad_(True)
    x.requires_grad_(True)

    dim_x = x.shape[-1]
    batch_shape = t.shape[:-1]
    t_flat = t.reshape(-1, 1)
    x_flat = x.reshape(-1, dim_x)

    y, dydt, dydx, dydxx = torch.vmap(
        dt_dx_dxx_single,
        in_dims=(None, 0, 0),
        out_dims=0,
    )(func, t_flat, x_flat)
    y = y.reshape(batch_shape + (1, ))
    dydt = dydt.reshape(batch_shape + (1, ))
    dydx = dydx.reshape(batch_shape + (1, dim_x))
    dydxx = dydxx.reshape(batch_shape + (1, dim_x, dim_x))

    t.requires_grad_(req_grad_t)
    x.requires_grad_(req_grad_x)
    return y, dydt, dydx, dydxx


def mc_for_v(
    t: torch.Tensor,
    x: torch.Tensor,
    v_term: callable,
    samp_xte_intft: callable,
    use_dist: bool,
    nsamp_mc: int = 10**6,
) -> torch.Tensor:
    # Compute the value function v(t, x) using Monte Carlo simulation.
    # the chunksize is determined adaptively to fit in memory.
    # Note: If use_dist is True, this function will utilize torch.distributed for accelerated computation.
    # Therefore, do not call this function only on a single rank when use_dist is True.

    assert x.ndim >= 2
    assert (t.ndim == 1) or (t.ndim == x.ndim)

    if use_dist is True:
        world_size = dist.get_world_size()
        nsamp_mc = int(nsamp_mc // world_size) + 1
        rank = dist.get_rank()
    else:
        rank = 0

    # nsamp_mc = self.nsamp_mc
    multiplier = (5 * x.shape[-1]) * x[..., 0].numel()
    cum_size = 0
    cum_mean = 0.
    progress = 0.
    print(f"Monte-Carlo for Ref. solution on {x.device}...")
    while cum_size < nsamp_mc:
        try:
            chunksize = get_safe_chunksize(multiplier,
                                           x.dtype,
                                           x.device,
                                           use_percent=0.3)
            chunksize = max(1, min(chunksize, nsamp_mc - cum_size))
            if use_dist is True:
                # Ensure all processes use the minimum chunksize
                chunksize_tensor = torch.tensor(chunksize, device=x.device)
                dist.all_reduce(chunksize_tensor, op=dist.ReduceOp.MIN)
                chunksize = int(chunksize_tensor.item())
            xte_chunk, ift_chunk = samp_xte_intft(
                t,
                x,
                num_mc=chunksize,
            )
            mean_vte = v_term(xte_chunk).mean(0)
            mean_ift = ift_chunk.mean(0)
            new_mean = mean_vte + mean_ift
            cum_size += chunksize
            if use_dist is True:
                dist.all_reduce(new_mean, op=dist.ReduceOp.SUM)
                new_mean = new_mean / world_size
            new_rate = chunksize / cum_size
            cum_mean = (1 - new_rate) * cum_mean + new_rate * new_mean

            if (cum_size / nsamp_mc > progress + 0.01) or \
                    (cum_size == nsamp_mc):
                progress = cum_size / nsamp_mc
                if rank == 0:
                    print(
                        f"Progress: {progress:.2%}, chunksize per rank: {chunksize}"
                    )

        except RuntimeError as err:
            if 'out of memory' in str(err):
                free_cache(x.device)
                chunksize = int(chunksize // 2)
                print(
                    f"Restricted by memory of {x.device}, reduce chunksize to {chunksize}"
                )
                if chunksize == 0:
                    print('Memory Error: Cannot reduce chunksize=0')
                    raise err
            else:
                raise err

    return cum_mean


class PDE(metaclass=abc.ABCMeta):

    name = 'Default_PDE_Name'

    # set to True if self.mu_sys/sgm_sys/f do not depend on v (or u for class HJB)
    musys_online = True
    sgmsys_online = True
    f_online = True

    # the nonlinear grad term: coefficient * |v_x|^exponential
    ceoff_vx = 0.
    exponential_vx = 2

    def __init__(self, dim_x, t0=0., te=1., use_dist=False) -> None:
        self.dim_x = dim_x
        self.dim_w = dim_x

        if isinstance(t0, float) or isinstance(t0, int):
            t0 = torch.tensor(t0)
        if isinstance(te, float) or isinstance(te, int):
            te = torch.tensor(te)
        self.t0 = t0
        self.te = te

        self.use_dist = use_dist
        if use_dist is True:
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

    @abstractmethod
    def mu_pil(self, _t, _x):
        pass

    @abstractmethod
    def sgm_pil(self, _t, _x, _dw):
        pass

    @abstractmethod
    def mu_sys(self, _t, _x, _v):
        pass

    @abstractmethod
    def sgm_sys(self, _t, _x, _v, _dw):
        pass

    @abstractmethod
    def tr_sgm2vxx(self, _t, _x, _v, _vxx):
        pass

    @abstractmethod
    def f(self, _t, _x, _v):
        pass

    @abstractmethod
    def x0_points(self, num_points):
        pass

    def produce_logfunc(self, _v_approx) -> callable:
        # return a logging function which will be called during training
        # the logging function should return a dict of log items
        pass

    def produce_results(self,
                        _v_approx,
                        _sav_prefix,
                        ctrfunc_syspath=None) -> None:
        pass

    def gen_pilpath(
        self,
        x0: torch.Tensor,
        te: torch.Tensor,
        num_dt: int,
    ):
        dt = (te - self.t0) / num_dt
        dwt = torch.normal(mean=0.,
                           std=torch.sqrt(dt),
                           size=(num_dt, x0.shape[-2], self.dim_w),
                           device=x0.device)

        t = [torch.full((1, 1), self.t0)]
        xt = [x0]
        for n in range(num_dt):
            tn = t[n]
            xtn = xt[n]
            mu_tn = self.mu_pil(tn, xtn)
            sgm_dwtn = self.sgm_pil(tn, xtn, dwt[n])
            t.append(tn + dt)
            xt.append(xtn + mu_tn * dt + sgm_dwtn)
        xt = torch.stack(xt, dim=0)
        # xt: [time, path, dim of x]
        return xt

    def gen_syspath(
        self,
        x0: torch.Tensor,
        te: torch.Tensor,
        num_dt: int,
        v_func,
    ):
        dt = (te - self.t0) / num_dt
        dwt = torch.normal(mean=0.,
                           std=torch.sqrt(dt),
                           size=(num_dt, x0.shape[-2], self.dim_w),
                           device=x0.device)

        t = [torch.full((1, 1), self.t0)]
        xt = [x0]
        vt = [v_func(t[0], x0)]
        for n in range(num_dt):
            tn = t[n]
            xtn = xt[n]
            vtn = vt[n]
            mu_tn = self.mu_sys(tn, xtn, vtn)
            sgm_dwtn = self.sgm_sys(tn, xtn, vtn, dwt[n])
            t.append(tn + dt)
            xt.append(xtn + mu_tn * dt + sgm_dwtn)
            vt.append(v_func(t[-1], xt[-1]))
        xt = torch.stack(xt, dim=0)
        vt = torch.stack(vt, dim=0)
        # xt: [time, path, dim of x]
        return xt, vt


def split_number(num, num_parts):
    part_size = num // num_parts
    parts = [part_size] * num_parts
    for i in range(num % num_parts):
        parts[i] += 1
    return parts


class PDEwithVtrue(PDE):
    name = 'Default_TVP_Name'

    x0pil_range = 1.
    x0_for_train = {'S2': diag_curve}

    record_linf_error = True
    num_testpoint = 1000

    @abstractmethod
    def v(self, _t, _x):
        pass

    @abstractmethod
    def v_term(self, _x):
        pass

    def x0_points(self, num_points):
        curve_funcs = self.x0_for_train.values()
        num_points = split_number(num_points, len(curve_funcs))
        x0_list = [
            c_func(self.dim_x, nump, self.x0pil_range)[1]
            for nump, c_func in zip(num_points, curve_funcs)
        ]
        x0 = torch.concatenate(x0_list, dim=0)
        return x0

    def produce_logfunc(self, v_approx):
        t0 = self.t0.unsqueeze(-1)
        x_test = self.x0_points(self.num_testpoint)
        if isin_ddp():
            # Ensure all processes have the same test points
            dist.broadcast(x_test, src=0)

        vtrue_ontest = self.v(t0, x_test)
        mean_vtrue = vtrue_ontest.mean()
        std_vtrue = vtrue_ontest.std()
        abs_vtrue = torch.abs(vtrue_ontest)
        vtrue_ontest_l1 = abs_vtrue.mean()
        vtrue_ontest_l2 = abs_vtrue.pow(2).mean().sqrt()
        vtrue_ontest_linf = abs_vtrue.max()

        def log_func(_it: int, _t: torch.Tensor, _xt: torch.Tensor):
            err = v_approx(t0, x_test) - vtrue_ontest
            abs_err = torch.abs(err)
            l1err = abs_err.mean() / vtrue_ontest_l1
            l2err = abs_err.pow(2).mean().sqrt() / vtrue_ontest_l2
            log = {
                'vtrue_mean': mean_vtrue.item(),
                'vtrue_std': std_vtrue.item(),
                'rel_l1err': l1err.item(),
                'rel_l2err': l2err.item(),
            }
            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()
            return log

        return log_func

    def res_on_path(
        self,
        v_approx,
        ctrfunc_syspath=None,
        num_dt=50,
        num_path=8,
    ):
        x0 = self.x0_points(num_path)
        with torch.no_grad():
            if ctrfunc_syspath is None:
                xt = self.gen_pilpath(x0, self.te, num_dt)
            else:
                xt, _ = self.gen_syspath(x0, self.te, num_dt, ctrfunc_syspath)
            t = torch.linspace(self.t0, self.te, xt.shape[0])
            t = t.unsqueeze(-1).unsqueeze(-1)
            vt_appr = v_approx(t, xt)
            vt_true = self.v(t, xt)
        err = (vt_appr - vt_true).abs()
        re1_tensor = err.mean(1, keepdims=True) / vt_true.abs().mean(
            1, keepdims=True)

        if self.record_linf_error:
            vture_max = vt_true.abs().max(1, keepdims=True)[0]
            reinf_tensor = err.max(1, keepdims=True)[0] / vture_max

        path_res = {
            't_path': tensor2ndarray(t),
            'vtrue_path': tensor2ndarray(vt_true),
            'vappr_path': tensor2ndarray(vt_appr),
            'rel_l1err_path': tensor2ndarray(re1_tensor),
        }
        if self.record_linf_error:
            path_res['rel_linferr_path'] = tensor2ndarray(reinf_tensor)
        return path_res

    def _residual(self, t, x, vnn):
        t = t.expand_as(x[..., :1])
        v, dvdt, dvdx, dvdxx = dt_dx_dxx(vnn, t, x)
        drift = torch.einsum(
            '...i, ...ji->...j',
            self.mu_sys(t, x, v),
            dvdx,
        )
        sgm2dydxx = self.tr_sgm2vxx(t, x, v, dvdxx)
        dv = dvdt + drift + 0.5 * sgm2dydxx

        if self.ceoff_vx != 0.:
            dv = dv + self.ceoff_vx * (dvdx.pow(self.exponential_vx)).sum(-1)

        res = dv + self.f(t, x, v)
        return res

    def _plot_landscape(self,
                        v_approx,
                        sav_prefix,
                        ctrfunc_syspath=None,
                        num_scatter_path=24,
                        numdt_scatter_path=16):
        assert self.dim_x == 1

        x0_path = self.x0_points(num_scatter_path)
        if ctrfunc_syspath is None:
            xt_path = self.gen_pilpath(x0_path, self.te, numdt_scatter_path)
        else:
            xt_path, _ = self.gen_syspath(
                x0_path,
                self.te,
                numdt_scatter_path,
                ctrfunc_syspath,
            )
        t = torch.linspace(self.t0, self.te, xt_path.shape[0])
        t = t.unsqueeze(-1).unsqueeze(-1)
        plot_landscape(
            self.t0,
            self.te,
            self.v,
            t,
            xt_path,
            sav_prefix + 'vtx_',
            cbar_label="$v(t, x)$",
        )

        res_func = lambda _t, _x: torch.clamp(
            self._residual(_t, _x, v_approx).abs(), max=1.)
        plot_landscape(
            self.t0,
            self.te,
            res_func,
            t,
            xt_path,
            sav_prefix + 'residual_',
            cbar_label=r"$\hat{r}(t, x)$",
        )

    def _xcurve_for_res(self, num_points=50, x0_range=None):
        if x0_range is None:
            x0_range = self.x0pil_range
        args = (self.dim_x, num_points, x0_range)
        curve_name = list(self.x0_for_train.keys())
        s, xs = zip(*[x0f(*args) for x0f in self.x0_for_train.values()])
        return curve_name, s, xs

    def produce_results(self, v_approx, sav_prefix, ctrfunc_syspath=None):
        args = (v_approx, self.v, self._xcurve_for_res)
        res_on_curve_ver2(*args,
                          f'{sav_prefix}resonline_',
                          type_vappr='scatter',
                          type_vtrue='plot')
        res_on_curve_ver2(*args,
                          f'{sav_prefix}resonline_type2_',
                          type_vappr='plot',
                          type_vtrue='scatter')
        v_histogram(self.v,
                    self._xcurve_for_res,
                    f'{sav_prefix}vtrueHistogram_',
                    num_pts=self.num_testpoint)

        if self.dim_x == 1:
            self._plot_landscape(v_approx,
                                 sav_prefix,
                                 ctrfunc_syspath=ctrfunc_syspath)


class LinearPDE(PDEwithVtrue):
    """
    Linear parabolic PDE with v(t, x) approximated by the Feynman-Kac formula
    """

    musys_online = False
    sgmsys_online = False
    f_online = False

    nsamp_mc = 10**6

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)

    def samp_xte_intft(self, t, x, num_dt=100, num_mc=10**4):
        dt = ((self.te - t) / num_dt)
        sqrt_dt = dt.pow(0.5)
        tn = t
        xtn = x
        int_ft = self.f(tn, xtn, None) * dt
        shape_normal = [num_mc] + list(x.shape[:-1]) + [self.dim_w]
        for _ in range(num_dt):
            normal_samp = torch.normal(mean=0.,
                                       std=1.,
                                       size=shape_normal,
                                       device=x.device)
            dwt = sqrt_dt * normal_samp
            mu_tn = self.mu_sys(tn, xtn, None)
            sgm_dwtn = self.sgm_sys(tn, xtn, None, dwt)
            tn = tn + dt
            xtn = xtn + mu_tn * dt + sgm_dwtn
            int_ft = int_ft + self.f(tn, xtn, None) * dt
        # xtn: [batch of mc, time, batch of x, dim of x]
        return xtn, int_ft

    def v(self,
          t: torch.Tensor,
          x: torch.Tensor,
          num_dt: int = 100) -> torch.Tensor:
        # Note: If self.use_dist is True, this function will utilize torch.distributed for accelerated computation.
        # Therefore, do not call this function only on a single rank when self.use_dist is True.

        assert self.musys_online is False
        assert self.sgmsys_online is False
        assert self.f_online is False
        if torch.all(t == self.te):
            return self.v_term(x)
        else:
            return mc_for_v(t, x, self.v_term, self.samp_xte_intft,
                            self.use_dist, self.nsamp_mc)


class LinearPDEwithConstantCoeff(LinearPDE):
    # LinearPDE variant with constant scalar drift (mu_sys = mu_const),
    # constant scalar diffusion (sgm_sys = sgm_const),
    # and zero source term: f ≡ 0.
    # The method self.v in this class is more efficient than that in LinearPDE.

    musys_online = False
    sgmsys_online = False
    f_online = False

    nsamp_mc = 10**6

    mu_const = 1.
    sgm_const = 1.

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)

    def mu_pil(self, t, x):
        return self.mu_sys(t, x, None)

    def sgm_pil(self, t, x, dw):
        return self.sgm_sys(t, x, None, dw)

    def mu_sys(self, _t, x, _v):
        return torch.full_like(x, self.mu_const)

    def sgm_sys(self, _t, _x, _v, dw):
        return self.sgm_const * dw

    def tr_sgm2vxx(self, _t, _x, _v, vxx):
        return self.sgm_const**2 * torch.einsum('...ii->...', vxx)

    def f(self, _t, x, _v):
        return torch.zeros_like(x[..., [0]])

    def samp_xte_intft(self, t, x, num_dt=100, num_mc=10**4):
        # num_dt is not used in this function

        dt = (self.te - t)
        norm_samp = torch.normal(
            mean=0.,
            std=1.,
            size=(num_mc, ) + x.shape[:-1] + (self.dim_w, ),
            device=x.device,
        )
        xte = x + self.mu_const * dt + self.sgm_const * dt.pow(0.5) * norm_samp
        int_ft = torch.zeros_like(xte[..., [0]])
        # xtn: [batch of mc, time, batch of x, dim of x]
        return xte, int_ft


class HJB(PDEwithVtrue):
    name = 'Default_HJB_equation'
    compute_cost_gap = None
    compute_cost_maxit = torch.inf

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim_u = None

    def comput_cost(self, x0, u_func, num_dt=100):
        t, xt, ut = self.gen_syspath(x0, num_dt, u_func)
        f_val = self.f(t[:-1], xt[:-1], ut[:-1])
        run_cost = f_val.mean([0, 1]) * (self.te - self.t0)
        term_cost = self.v_term(xt[-1]).mean(0)
        cost = run_cost + term_cost
        if self.use_dist is True:
            dist.all_reduce(cost)
            cost = cost / self.world_size
        return cost

    def compute_cost_onx0points(self,
                                x0,
                                u_approx,
                                num_dt=100,
                                num_cost_path=64):
        # x0: [points, x]
        cost_list = []
        for x0i in x0:
            x0i_mc = torch.ones([num_cost_path, 1]) * x0i
            cost_x0i = self.comput_cost(x0i_mc, u_approx, num_dt=num_dt)
            cost_list.append(cost_x0i)
        cost_onpoints = torch.stack(cost_list, dim=0)
        return cost_onpoints

    def x0_for_cost(self, num_points):
        return self.x0_points(num_points)

    def produce_logfunc(self, v_approx, u_approx, num_cost_path=64):
        t0 = self.t0.unsqueeze(-1)
        x0_cost = self.x0_for_cost(num_cost_path)
        x0_value = self.x0_points(self.num_testpoint)

        if isin_ddp():
            # Ensure all processes have the same test points
            dist.broadcast(x0_cost, src=0)
            dist.broadcast(x0_value, src=0)

        vtrue_onx0 = self.v(t0, x0_value)
        vtrue_onx0cost = self.v(t0, x0_cost).mean()

        vtrue_ontest_l1 = torch.abs(vtrue_onx0).mean()
        vtrue_ontest_linf = torch.abs(vtrue_onx0).max()

        def log_func(it: int, _t: torch.Tensor, _xt: torch.Tensor):
            with torch.no_grad():
                err = v_approx(t0, x0_value) - vtrue_onx0
                abs_err = torch.abs(err)
                l1err = abs_err.mean() / vtrue_ontest_l1
            log = {'rel_l1err': l1err.item()}

            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()

            if self.compute_cost_gap is not None:
                if (it % self.compute_cost_gap == 0) and \
                        (it < self.compute_cost_maxit):
                    with torch.no_grad():
                        cost = self.comput_cost(x0_cost, u_approx).item()
                        cost_v0 = cost / vtrue_onx0cost.item()
                    log['cost'] = cost
                    log['vtrue_on_x0cost'] = vtrue_onx0cost.item()
                    log['cost/v0'] = cost_v0
                else:
                    log['cost'] = None
                    log['vtrue_on_x0cost'] = None
                    log['cost/v0'] = None
            return log

        return log_func


class BVP(PDE):
    """
    Boundary Value Problem (BVP) where v(t, x) is zero for x on the boundary of the unit ball.
    The spatial domain is the unit ball in R^d.
    The variable t is included for compatibility with other solvers, but is not used in for BVPs.
    """
    x0_for_train = {'UB': unit_ball}
    # v_shellfunc = lambda x, y: (1 - x.pow(2).sum(-1, keepdim=True)) * y

    num_testpoint = 1000

    def v_shellfunc(self, x, y):
        return (1 - x.pow(2).sum(-1, keepdim=True)) * y

    def x0_points(self, num_points):
        curve_funcs = self.x0_for_train.values()
        num_points = split_number(num_points, len(curve_funcs))
        x0_list = [
            c_func(self.dim_x, nump, self.x0pil_range)[1]
            for nump, c_func in zip(num_points, curve_funcs)
        ]
        x0 = torch.concatenate(x0_list, dim=0)
        return x0

    def produce_logfunc(self, v_approx):
        t0 = self.t0.unsqueeze(-1)
        x_test = self.x0_points(self.num_testpoint)
        if isin_ddp():
            # Ensure all processes have the same test points
            dist.broadcast(x_test, src=0)

        vtrue_ontest = self.v(t0, x_test)
        # print(f'x_test_0:{x_test[0]}, rank: {x_test.device}')

        abs_vtrue = torch.abs(vtrue_ontest)
        vtrue_ontest_l1 = abs_vtrue.mean()
        vtrue_ontest_l2 = abs_vtrue.pow(2).mean().sqrt()
        vtrue_ontest_linf = abs_vtrue.max()

        def log_func(_it: int, _t: torch.Tensor, _xt: torch.Tensor):
            err = v_approx(t0, x_test) - vtrue_ontest
            abs_err = torch.abs(err)
            l1err = abs_err.mean() / vtrue_ontest_l1
            l2err = abs_err.pow(2).mean().sqrt() / vtrue_ontest_l2
            linf_err = abs_err.max() / vtrue_ontest_linf
            log = {
                'rel_l1err': l1err.item(),
                'rel_l2err': l2err.item(),
                'rel_linferr': linf_err.item(),
                'l1_vtrue': vtrue_ontest_l1.item(),
                'l2_vtrue': vtrue_ontest_l2.item(),
                'linf_vtrue': vtrue_ontest_linf.item(),
            }
            return log

        return log_func

    def produce_results(self, v_approx, sav_prefix, ctrfunc_syspath=None):
        args = (v_approx, self.v, self._xcurve_for_res)
        res_on_curve_ver2(*args,
                          f'{sav_prefix}resonline_',
                          type_vappr='scatter',
                          type_vtrue='plot')
        res_on_curve_ver2(*args,
                          f'{sav_prefix}resonline_type2_',
                          type_vappr='plot',
                          type_vtrue='scatter')


class EVP(PDEwithVtrue):

    name = 'EigenValueMeta'
    x0_for_train = {'S1': e1_curve}
    x0pil_range = 3.

    musys_online = False
    sgmsys_online = False
    f_online = True

    rate_newlamb = 0.1
    fourier_frequency = None

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.te = 10
        self.dim_w = dim_x

        self.relu6 = torch.nn.ReLU6()
        self.softplus = torch.nn.Softplus()

        self.lamb_init = 0.
        self.lamb_val = torch.tensor(self.lamb_init)
        self.lamb_from_v = self.lamb_val

    def register_network(self, vnn: torch.nn.Module):
        self.vnn = vnn

    def get_lamb(self):
        new_lamb = self.vnn.module.lamb
        r = self.rate_newlamb
        self.lamb_val = (1 - r) * self.lamb_val.detach().item() + r * new_lamb
        return self.lamb_val

    def lamb_v(self, t, vpath):
        v_mean = vpath.mean(1, keepdim=True)
        lamb_batch = torch.log(
            (v_mean[1:] / v_mean[:1]).abs()) / (t[1:] - t[:1])
        lamb_val = lamb_batch.mean()

        r = self.rate_newlamb
        self.lamb_from_v = (
            1 - r) * self.lamb_from_v.detach().item() + r * lamb_val

        lamb_var = (lamb_batch - lamb_val).pow(2).mean()
        return lamb_val, lamb_var

    def f(self, t, _x, v):
        # lamb_val = self.get_lamb()
        with torch.no_grad():
            lamb_val = self.lamb_v(t, v)[0]
        f_val = -lamb_val * v
        return f_val

    def additional_loss(self, t_path, x_path, v_approx):
        # x0 = self.x0_points(100)
        x0 = torch.zeros((1, self.dim_x), device=self.t0.device)
        vappr_val = v_approx(None, x0)
        vtrue_val = self.v(None, x0)
        loss_val = (vtrue_val - vappr_val).mean().pow(2)

        vappr_path = v_approx(t_path, x_path)
        xnorm_mask = (x_path.norm(dim=-1, keepdim=True) > 3.)
        loss_val = loss_val + (xnorm_mask * vappr_path.pow(2)).mean()

        return loss_val

    def x0_points(self, num_points):
        curve_funcs = self.x0_for_train.values()
        num_points = split_number(num_points, len(curve_funcs))
        x0_list = [
            c_func(self.dim_x, nump, self.x0pil_range)[1]
            for (nump, c_func) in zip(num_points, curve_funcs)
        ]
        x0 = torch.concatenate(x0_list, dim=0)
        return x0

    def produce_logfunc(self, v_approx):
        t0 = self.t0.unsqueeze(-1)
        x_test = self.x0_points(self.num_testpoint)

        if isin_ddp():
            # Ensure all processes have the same test points
            dist.broadcast(x_test, src=0)

        vtrue_ontest = self.v(t0, x_test)
        vtrue_ontest_l1 = torch.abs(vtrue_ontest).mean()
        vtrue_ontest_linf = torch.abs(vtrue_ontest).max()

        def log_func(_it: int, t: torch.Tensor, xt: torch.Tensor):
            v0_approx = v_approx(t0, x_test)
            err = v0_approx - vtrue_ontest
            abs_err = torch.abs(err)
            l1err = abs_err.mean() / vtrue_ontest_l1
            with torch.no_grad():
                vappr_path = v_approx(t, xt)
                lamb_from_v = self.lamb_v(t, vappr_path)[0]

            log = {
                'ev': self.lamb_val.detach().item(),
                'ev_from_v': lamb_from_v.item(),
                'ev_error': (self.true_eigenval - lamb_from_v).abs().item(),
                'rel_l1err': l1err.item(),
            }
            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()
            return log

        return log_func
