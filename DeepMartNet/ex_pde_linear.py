import torch
import ex_meta
import numpy as np
from ex_meta import (LinearPDE, PDEwithVtrue, diag_curve, e1_curve, mc_for_v,
                     tensor2ndarray)


class Shock(LinearPDE):
    name = 'Shock'
    x0_for_train = {'S2': diag_curve}
    record_linf_error = True

    musys_online = False
    sgmsys_online = False
    f_online = False
    delta = 0.1

    coeff_mu = 1.
    coeff_osc = 1.0
    num_testpoint = 300
    nsamp_mc = 10**6

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x
        self.te = 2

    def mu_pil(self, t, x):
        return self.mu_sys(t, x, None)

    def sgm_pil(self, _t, _x, dw):
        return self.delta**(0.5) * dw

    def mu_sys(self, _t, x, _v):
        mu = self.coeff_mu * torch.tanh(10 * x)
        return mu

    def sgm_sys(self, _t, _x, _v, dw):
        return self.delta**(0.5) * dw

    def tr_sgm2vxx(self, t, x, _v, vxx):
        return self.delta * torch.einsum('...ii->...', vxx)

    def v_term(self, x):
        tanhx = torch.tanh(x).mean(-1, keepdim=True)
        osc = torch.cos(10 * x).mean(-1, keepdim=True)
        return tanhx + self.coeff_osc * osc

    def f(self, _t, x, _v):
        return torch.zeros_like(x[..., [0]])


class Shock1a(Shock):
    pass


class Shock1aNoOsc(Shock1a):
    coeff_osc = 0.0


class Shock1aBmPil(Shock1a):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, _x, dw):
        return dw


class Shock1b(Shock):
    coeff_mu = 5.0


class Shock1bNoOsc(Shock1b):
    coeff_osc = 0.0


class Shock1bBmPil(Shock1b):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, _x, dw):
        return dw


class Counter(PDEwithVtrue):
    name = 'Counter_Example'
    x0_for_train = {'S1': e1_curve, 'S2': diag_curve}

    musys_online = False
    sgmsys_online = False
    f_online = False

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, _x, dw):
        return dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, _x, _v, dw):
        return dw

    def tr_sgm2vxx(self, t, x, _v, vxx):
        return torch.einsum('...ii->...', vxx)

    def v_term(self, x):
        return 1. + torch.sin(self.te + x).mean(-1, keepdim=True)

    def f(self, t, x, _v):
        dt_u = torch.cos(t + x).mean(-1, keepdim=True)
        dxx_u = -torch.sin(t + x).mean(-1, keepdim=True)
        return -dt_u - 0.5 * dxx_u

    def v(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)


class LinearSin(PDEwithVtrue):
    name = 'Linear_Sinx'
    x0_for_train = {'diag': diag_curve}

    musys_online = False
    sgmsys_online = False
    f_online = False
    c_in_sgm = 5.

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

    def mu_pil(self, _t, x):
        mu = torch.sin(2 * x)
        return mu

    def sgm_pil(self, t, x, dw):
        sgm = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        return sgm * dw

    def mu_sys(self, _t, x, _v):
        mu = torch.sin(2 * x)
        return mu

    def sgm_sys(self, t, x, _v, dw):
        sgm = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        return sgm * dw

    def tr_sgm2vxx(self, t, x, _v, vxx):
        sgm = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        return torch.einsum('...kij, ...i, ...j->...k', vxx, sgm, sgm)

    def v_term(self, x):
        return 1. + torch.sin(self.te + x).mean(-1, keepdim=True)

    def f(self, t, x, _v):
        sintx = torch.sin(t + x)
        costx = torch.cos(t + x)
        dt_v = costx.mean(-1, keepdim=True)

        mu = torch.sin(2 * x)
        mu_vx = (mu * costx).mean(-1, keepdim=True)

        sgm_val = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        tr_sgm_vxx = -(sgm_val.pow(2) * sintx).mean(-1, keepdim=True)
        return -dt_v - mu_vx - 0.5 * tr_sgm_vxx

    def v(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)


class BlackScholes(PDEwithVtrue):
    name = 'BlackScholes'
    x0_for_train = {'CUBE': ex_meta.cube}
    x0pil_range = (0.9, 1.1)

    musys_online = False
    sgmsys_online = False
    f_online = False

    nsamp_mc = 10**6
    mu_coeff = -1 / 20
    sgm_max = 0.5
    r = 1 / 20
    k = 1.

    def __init__(self, dim_x, te=1., **kwargs) -> None:
        super().__init__(dim_x, te=te, **kwargs)
        self.dim_w = dim_x
        self.sgm_vec = torch.linspace(0.1, self.sgm_max, dim_x)

    def mu_pil(self, t, x):
        return self.mu_sys(t, x, None)

    def sgm_pil(self, t, x, dw):
        return self.sgm_sys(t, x, None, dw)

    def mu_sys(self, _t, x, _v):
        return self.mu_coeff * x

    def sgm_sys(self, _t, x, _v, dw):
        return self.sgm_vec * x * dw

    def tr_sgm2vxx(self, t, x, _v, vxx):
        sgm = self.sgm_vec * x
        return torch.einsum('...kij, ...i, ...j->...k', vxx, sgm, sgm)

    def v_term(self, x):
        payoff = torch.clamp(
            x.max(-1, keepdim=True)[0] - torch.tensor(self.k),
            min=0.,
        )
        return torch.exp(-self.r * self.te) * payoff

    def f(self, _t, x, _v):
        return torch.zeros_like(x[..., [0]])

    def samp_xte_intft(self, t, x, num_dt=100, num_mc=10**4):
        # num_dt is not used in this function

        dt = (self.te - t)
        norm_samp = torch.normal(mean=0.,
                                 std=1.,
                                 size=(num_mc, ) + x.shape[:-1] +
                                 (self.dim_w, ),
                                 device=x.device)
        in_exp = self.sgm_vec * norm_samp * dt.pow(0.5) + (
            self.mu_coeff - 0.5 * (self.sgm_vec**2)) * dt
        xte = x * torch.exp(in_exp)
        int_ft = torch.zeros_like(xte[..., [0]])
        # xtn: [batch of mc, time, batch of x, dim of x]
        return xte, int_ft

    def v(self, t, x):
        assert self.musys_online is False
        assert self.sgmsys_online is False
        assert self.f_online is False
        if torch.all(t == self.te):
            return self.v_term(x)
        else:
            return mc_for_v(t, x, self.v_term, self.samp_xte_intft,
                            self.use_dist, self.nsamp_mc)


class BlackScholesUnScaled(BlackScholes):
    name = 'BlackScholesUnScaled'
    x0pil_range = (90, 110)
    k = 100.

    def produce_results(self, v_approx, sav_prefix, ctrfunc_syspath=None):
        super().produce_results(v_approx, sav_prefix, ctrfunc_syspath)
        t_uns = torch.tensor(0.).unsqueeze(-1)
        curve_name, _, xs = self._xcurve_for_res(num_points=1000)
        vtrue_xs = [tensor2ndarray(self.v(t_uns, x)) for x in xs]
        xs = [tensor2ndarray(x) for x in xs]
        for (cname, vtrue, x) in zip(curve_name, vtrue_xs, xs):
            header = ['v'] + [f'x{i+1}' for i in range(self.dim_x)]
            v_x = np.concatenate([vtrue, x], axis=-1)
            np.savetxt(sav_prefix + 'vtrue_' + cname + '.csv',
                       v_x,
                       delimiter=',',
                       header=','.join(header))


class BlackScholesUnScaled02(BlackScholesUnScaled):
    name = 'BlackScholesUnScaled02'
    x0pil_range = (0, 200)


# class BlackScholesUnScaled_VarSgm(BlackScholesUnScaled):
#     name = 'BlackScholesUnScaled_VarSgm'

#     def __init__(self, dim_x, te=1., **kwargs) -> None:
#         super().__init__(dim_x, te=te, **kwargs)
#         self.sgm_vec = 0.1 + torch.arange(0, dim_x) / 200


# class BlackScholesUnScaled_VarSgm02(BlackScholesUnScaled_VarSgm):
#     name = 'BlackScholesUnScaled_VarSgm02'
#     x0pil_range = (0, 200)
