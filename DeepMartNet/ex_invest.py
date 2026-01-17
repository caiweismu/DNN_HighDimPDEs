import torch

from ex_meta import (HJB, LinearPDE)


class CRRA(HJB):
    '''
    Investment problem with constant relative risk aversion (CRRA). 
    This class is imcomplete. 
    
    This model is from 
        @article{Zar2001solution,
        author = {Zariphopoulou, Thaleia},
        title = {A solution approach to valuation with unhedgeable risks},
        fjournal = {Finance and Stochastics},
        journal = {Finance Stoch.},
        issn = {0949-2984},
        volume = {5},
        number = {1},
        pages = {61--82},
        year = {2001},
        doi = {10.1007/PL00000040},
        }
    '''

    name = 'CRRA_Investment'
    nsamp_mc = 10**6

    musys_online = True
    sgmsys_online = True
    f_online = False

    r = 0.02  # interest rate
    corr = -0.7  # correlation
    gamma = -1  # should be < 1 and neq 0

    def __init__(self, dim_x, **kwargs) -> None:
        assert dim_x == 2, "This example only supports dim_x=2."

        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x
        self.dim_u = dim_x
        self.delta = (1 - self.gamma) / (1 - self.gamma +
                                         self.corr**2 * self.gamma)

    def b(self, _t, y):
        return 2.0 * (0.04 - y)

    def a(self, _t, y):
        return 0.3 * y**0.5

    def mu_stock(self, _t, y):
        return 0.05 - 0.2 * y

    def sgm_stock(self, _t, y):
        return y**0.5

    def mu_pil(self, t, x):
        k = torch.full_like(x[..., [0]], 0.5)
        return self.mu_sys(t, x, k)

    def sgm_pil(self, t, x, dw):
        k = torch.full_like(x[..., [0]], 0.5)
        return self.sgm_sys(t, x, k, dw)

    def mu_sys(self, t, x, k):
        x0 = x[..., [0]]
        x1 = x[..., [1]]

        mu_0 = self.r * x0 + (self.mu_stock(t, x1) - self.r) * k
        mu_1 = self.b(t, x1)
        return torch.cat([mu_0, mu_1], dim=-1)

    def sgm_sys(self, t, x, k, dw):
        x1 = x[..., [1]]
        dw0 = dw[..., [0]]
        dw1 = dw[..., [1]]

        sgm0 = self.sgm_stock(t, x1) * k * dw0
        sgm1 = self.a(t, x1) * (self.corr * dw0 +
                                (1 - self.corr**2)**(0.5) * dw1)
        return torch.cat([sgm0, sgm1], dim=-1)

    def tr_sgm2vxx(self, _t, _x, _v, _vxx):
        raise NotImplementedError

    def f(self, _t, x, _k):
        return torch.zeros_like(x[..., [0]])

    def h(self, y):
        return torch.exp(-0.1 * y)

    def v_term(self, x):
        x0 = x[..., [0]]
        x1 = x[..., [1]]
        return -x0**self.gamma / self.gamma * self.h(x1)**self.delta

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [time step, batch, x], or [batch, x]
        # the shape of t should admit the validity of t + x
        assert (t.ndim == 1) or (t.ndim == x.ndim)
        assert x.ndim == 2

        class LinPDE(LinearPDE):
            musys_online = False
            sgmsys_online = False
            f_online = True
            nsamp_mc = self.nsamp_mc

            def v_term(lpde_self, y):
                return self.h(y)

            def f(lpde_self, t, y, v):
                i0 = self.gamma * self.delta
                mu_st = self.mu_stock(t, y)
                sgm_st = self.sgm_stock(t, y)
                i1 = (mu_st - self.r)**2 / 2 / sgm_st**2 / (1 - self.gamma)
                return i0 * (self.r + i1) * v

            def mu_pil(lpde_self, t, y):
                return lpde_self.mu_sys(t, y, None)

            def sgm_pil(lpde_self, t, y, dw):
                return lpde_self.sgm_sys(t, y, None, dw)

            def mu_sys(lpde_self, t, y, _v):
                a_val = self.a(t, y)
                b_val = self.b(t, y)
                sgm_st = self.sgm_stock(t, y)
                mu_st = self.mu_stock(t, y)
                i1 = self.corr * self.gamma * (mu_st - self.r) * a_val / (
                    1 - self.gamma) / sgm_st
                return b_val + i1

            def sgm_sys(lpde_self, t, y, _v, dw):
                return self.a(t, y) * dw

            def tr_sgm2vxx(lpde_self, t, y, _v, vyy):
                return self.a(t, y)**2 * vyy

            def x0_points(lpde_self, num_points):
                return self.x0_points(num_points)

        lin_pde = LinPDE(1, t0=self.t0, te=self.te, use_dist=self.use_dist)
        v_inner = lin_pde.v(t, x[..., [1]])
        v_val = x[..., [0]]**self.gamma / self.gamma * v_inner**self.delta
        return v_val
