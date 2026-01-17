# Some examples of PDE and SOCP
import torch

from ex_meta import PDE, PDEwithVtrue, diag_curve


class AllenCahnSin(PDEwithVtrue):
    name = 'AllenCahn_Sinx'
    x0_for_train = {'diag': diag_curve}

    musys_online = False
    sgmsys_online = False
    f_online = True
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

    def f(self, t, x, v):
        sintx = torch.sin(t + x)
        costx = torch.cos(t + x)
        dt_v = costx.mean(-1, keepdim=True)

        mu = torch.sin(2 * x)
        mu_vx = (mu * costx).mean(-1, keepdim=True)

        sgm_val = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        tr_sgm_vxx = -(sgm_val.pow(2) * sintx).mean(-1, keepdim=True)

        v_v3 = v - v.pow(3)
        v_exact = 1 + sintx.mean(-1, keepdim=True)
        v_v3_exact = v_exact - v_exact.pow(3)
        return -dt_v - mu_vx - 0.5 * tr_sgm_vxx - v_v3_exact + v_v3

    def v(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)


class AllenCahn(PDE):
    '''
    This example is taken from Eq. [15] in
        J. Han, A. Jentzen, W. E, Solving high-dimensional partial differential equations using deep learning, Proc. Natl. Acad. Sci. U.S.A. 115 (34) 8505-8510, (2018).
    '''
    name = 'Allen-Cahn-equation'
    musys_online = False
    sgmsys_online = False
    f_online = True

    def __init__(self, dim_x, t0=0., te=0.3, use_dist=False) -> None:
        super().__init__(dim_x, t0=t0, te=te, use_dist=use_dist)
        self.dim_w = dim_x

        if dim_x != 100:
            raise ValueError("Only dim_x = 100 is available for this example.")
        if t0 != 0.:
            raise ValueError("Only t_0 = 0 is available for this example.")

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return torch.full_like(x, 2**(0.5)) * dw

    def mu_sys(self, t, x, _v):
        return self.mu_pil(t, x)

    def sgm_sys(self, t, x, _v, dw):
        return self.sgm_pil(t, x, dw)

    def tr_sgm2vxx(self, _t, _x, _v, vxx):
        return 2 * torch.einsum('...ii->...', vxx)

    def v_term(self, x):
        y = 1 / (2 + 0.4 * x.pow(2).sum(-1, keepdims=True))
        return y

    def f(self, _t, _x, v):
        return v - v.pow(3)

    def x0_points(self, num_points):
        x0 = torch.zeros((num_points, self.dim_x))
        return x0

    def produce_logfunc(self, v_approx):
        v_val = torch.tensor((0.0528, ))
        x_test = torch.zeros((self.dim_x, ))
        v_val_l1 = torch.abs(v_val).mean()

        def log_func(_it: int, _t: torch.Tensor, _xt: torch.Tensor):
            vpred = v_approx(self.t0.unsqueeze(-1), x_test)
            err = vpred - v_val
            l1err = torch.abs(err).mean() / v_val_l1
            log = {
                'rel_l1err': l1err.item(),
                'vpred_x0': vpred.item(),
                'vtrue_x0': v_val.item(),
            }
            return log

        return log_func


class BSE(PDE):
    '''
    This example is taken from Eq. [11] in
        J. Han, A. Jentzen, W. E, Solving high-dimensional partial differential equations using deep learning, Proc. Natl. Acad. Sci. U.S.A. 115 (34) 8505-8510, (2018).
    '''
    name = 'Black-Scholes-equation'
    musys_online = False
    sgmsys_online = False
    f_online = True

    def __init__(self, dim_x, t0=0., **kwargs) -> None:
        super().__init__(dim_x, t0=t0, **kwargs)
        self.dim_w = dim_x

        if t0 != 0.:
            raise ValueError("Only t_0 = 0 is available for this example.")

        self.bar_mu = 0.02
        self.bar_sgm = 0.2
        self.delta = 2 / 3
        self.r = 0.02

        if dim_x == 100:
            self.vh = 50.
            self.vl = 70.
            self.v0 = torch.tensor((60.781, ))
        elif dim_x == 1:
            self.vh = 50.
            self.vl = 120.
            self.v0 = torch.tensor((97.705, ))
        else:
            assert dim_x in (1, 100)

        self.relu6 = torch.nn.ReLU6()

    def mu_pil(self, _t, x):
        return self.bar_mu * x

    def sgm_pil(self, _t, x, dw):
        return self.bar_sgm * x * dw

    def mu_sys(self, t, x, _v):
        return self.mu_pil(t, x)

    def sgm_sys(self, t, x, _v, dw):
        return self.sgm_pil(t, x, dw)

    def tr_sgm2vxx(self, t, x, _v, vxx):
        sgm = self.bar_sgm * x
        return torch.einsum('...kij, ...i, ...j->...k', vxx, sgm, sgm)

    def v_term(self, x):
        y = x.min(-1, keepdims=True)[0]
        return y

    def q_func(self, v):
        relu6_val = self.relu6((v - self.vh) / (self.vl - self.vh) * 6.)
        q = 0.2 + (0.02 - 0.2) * relu6_val / 6.
        return q

    def f(self, _t, _x, v):
        f_val = -(1 - self.delta) * self.q_func(v) * v - self.r * v
        return f_val

    def x0_points(self, num_points):
        x0 = torch.full((num_points, self.dim_x), 100.)
        return x0

    def produce_logfunc(self, v_approx):
        v_val = self.v0
        x_test = torch.full((self.dim_x, ), 100.)
        v_val_l1 = torch.abs(v_val).mean()

        def log_func(_it: int, _t: torch.Tensor, _xt: torch.Tensor):
            vpred = v_approx(self.t0.unsqueeze(-1), x_test)
            err = vpred - v_val
            l1err = torch.abs(err).mean() / v_val_l1
            log = {
                'rel_l1err': l1err.item(),
                'vpred_x0': vpred.item(),
                'vtrue_x0': v_val.item(),
            }
            return log

        return log_func


class AllenCahnSDGD(PDEwithVtrue):
    '''
    This example is modified from 
    title = {Tackling the curse of dimensionality with physics-informed neural networks},
    journal = {Neural Networks},
    author = {Zheyuan Hu and Khemraj Shukla and George Em Karniadakis and Kenji Kawaguchi},
    volume = {176},
    pages = {106369},
    year = {2024},
    doi = {https://doi.org/10.1016/j.neunet.2024.106369}.
    '''
    name = 'Allen-Cahn-SDGD'
    musys_online = False
    sgmsys_online = False
    f_online = True
    x0pil_range = 1.5
    x0_for_train = {'S2': diag_curve}

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

        xi = torch.linspace(-torch.pi / 2, torch.pi / 2, dim_x)
        self.ci = (1.5 + torch.sin(xi)) / dim_x

    @staticmethod
    def x_shift(t, x):
        return t + x - torch.tensor(0.5)

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return torch.full_like(x, 2**0.5) * dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, x, _v, dw):
        return torch.full_like(x, 2**0.5) * dw

    def tr_sgm2vxx(self, t, x, _v, vxx):
        return 2 * torch.einsum('...ii->...', vxx)

    def v(self, t, x):
        x = self.x_shift(t, x)
        cosx = torch.cos(x)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        s = torch.sin(x + cosx_r1 + x_r1 * cosx)
        return (self.ci * s).sum(-1, keepdims=True)

    def v_term(self, x):
        return self.v(self.te, x)

    def d_v(self, t, x):
        x = self.x_shift(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        sinx_r1 = torch.roll(sinx, -1, dims=-1)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        x_inner = x + cosx_r1 + x_r1 * cosx

        sin_xinn = torch.sin(x_inner)
        cos_xinn = torch.cos(x_inner)

        dx_xinn1 = 1. - x_r1 * sinx
        dx_xinn2 = cosx - sinx_r1
        dx_xinn = dx_xinn1 + dx_xinn2
        sumdx_v = (self.ci * cos_xinn * dx_xinn).sum(-1, keepdims=True)

        dxx_xinn1 = -dx_xinn1.pow(2) - dx_xinn2.pow(2)
        dxx_xinn2 = -x_r1 * cosx - cosx_r1

        sxinn_cxinn = sin_xinn * dxx_xinn1 + cos_xinn * dxx_xinn2
        sumdxx_v = (self.ci * sxinn_cxinn).sum(-1, keepdims=True)

        d_v = sumdx_v + sumdxx_v
        v = (self.ci * sin_xinn).sum(-1, keepdims=True)
        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        return v - v.pow(3) - dv_val - v_true + v_true.pow(3)


class AllenCahnSDGDXrad0d5(AllenCahnSDGD):
    x0_scale = 0.5
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'
    x0pil_range = x0_scale


class AllenCahnSDGDXrad0(AllenCahnSDGDXrad0d5):
    x0_scale = 0.
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'

    @staticmethod
    def x_shift(t, x):
        return t + x - torch.tensor(1.0)


class AllenCahnSDGDXrad1d0(AllenCahnSDGDXrad0d5):
    x0_scale = 1.0
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'


class AllenCahnSDGDXrad1d5(AllenCahnSDGDXrad0d5):
    x0_scale = 1.5
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'


class AllenCahnSDGDXrad2d0(AllenCahnSDGDXrad0d5):
    x0_scale = 2.0
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'


class AllenCahnSDGDXrad2d5(AllenCahnSDGDXrad0d5):
    x0_scale = 2.5
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'
