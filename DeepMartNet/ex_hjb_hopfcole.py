from decimal import Decimal, getcontext
import torch

from ex_meta import (HJB, LinearPDE, LinearPDEwithConstantCoeff, diag_curve,
                     manifold_curve)


def sqrt2_groups(n: int, k: int):
    """
    Computes the square root of 2 with sufficient arbitrary precision (using Python's `decimal` module),
    slices the first n*k fractional digits into n consecutive, non-overlapping groups of length k,
    and converts each group into a floating-point number of the form 0.<group>.
    The result is returned as a list of floats.

    This function is used to generate a sequence of random-like floats in the interval (0, 1),
    ensuring reproducibility across different runs and platforms.

    Parameters
    ----------
    n : int
        Number of groups to extract.
    k : int
        Number of digits per group.
    """

    assert n > 0 and k > 0, "n and k must be positive integers."
    precision = 2 + n * k + 10
    getcontext().prec = precision

    sqrt2 = Decimal(2).sqrt()
    s = str(sqrt2)
    _, decimal_part = s.split('.')
    decimal_digits = decimal_part[:n * k]

    groups = []
    for i in range(n):
        start = i * k
        segment = decimal_digits[start:start + k]
        float_val = float('0.' + segment)
        groups.append(float_val)
    return groups


class HJB0(HJB):
    # The HJB version (with inf H) of
    # \partial_t u(x, t) + 1/2 \Delta u(x, t) - |\nabla_x u(x,t)|^2 = 0

    name = 'HJB0'
    musys_online = True
    sgmsys_online = False
    f_online = True
    compute_cost_gap = None

    x0_for_train = {'S2': diag_curve, 'S3': manifold_curve}

    nsamp_mc = 10**6
    num_testpoint = 300

    coeff_b = 0.
    coeff_sgm = 1
    c_val = 2**0.5

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        assert self.sgmsys_online is False

        self.dim_w = dim_x
        self.dim_u = dim_x

    def b(self, _t, x):
        return torch.full_like(x, self.coeff_b)

    def mu_pil(self, t, x):
        return self.b(t, x)

    def sgm_pil(self, t, x, dw):
        return self.sgm_sys(t, x, None, dw)

    def sgm_sys(self, _t, _x, _k, dw):
        return self.coeff_sgm * dw

    def tr_sgm2vxx(self, _t, _x, _v, vxx):
        return self.coeff_sgm**2 * torch.einsum('...ii->...', vxx)

    def mu_sys(self, t, x, k):
        btx = self.b(t, x)
        sgm_k = self.sgm_sys(t, x, None, k)
        return btx + self.c_val * sgm_k

    def g(self, x):
        return 0.5 + 0.5 * (x**2).sum(-1, keepdim=True)

    def v_term(self, x):
        return torch.log(self.g(x))

    def f(self, _t, _x, k):
        return 0.5 * k.pow(2).sum(-1, keepdims=True)

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [time step, batch, x], or [batch, x]
        # the shape of t should admit the validity of t + x
        assert (t.ndim == 1) or (t.ndim == x.ndim)
        assert x.ndim >= 2

        class LinPDE(LinearPDEwithConstantCoeff):
            musys_online = False
            sgmsys_online = False
            f_online = False
            nsamp_mc = self.nsamp_mc

            mu_const = self.coeff_b
            sgm_const = self.coeff_sgm

            def v_term(lpde_self, x):
                return self.g(x)**(-self.c_val**2)

            def x0_points(lpde_self, num_points):
                return self.x0_points(num_points)

        lin_pde = LinPDE(self.dim_x,
                         t0=self.t0,
                         te=self.te,
                         use_dist=self.use_dist)
        v_inner = lin_pde.v(t, x)
        v_val = -self.c_val**(-2) * torch.log(v_inner)
        return v_val


class HJB0a(HJB0):
    name = 'HJB0a'


class HJB0_Rosenbrock(HJB0):
    # A variant of HJB0 with Rosenbrock terminal condition
    name = 'HJB0_Rosenbrock'

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        c_coeff = sqrt2_groups(dim_x * 2, 3)
        self.c1_coeff = torch.tensor(c_coeff[:dim_x]) + 0.5
        self.c2_coeff = torch.tensor(c_coeff[dim_x:]) + 0.5

    def g(self, x):
        x0 = x[..., :-1]
        x1 = x[..., 1:]
        delta_xx = (self.c1_coeff[1:] * (x1 - x0).pow(2))
        x3 = delta_xx + self.c2_coeff[:-1] * x1.pow(2)
        return 0.5 + 0.5 * x3.sum(-1, keepdim=True)


class HJB1a(HJB):
    name = 'HJB1a'
    musys_online = True
    sgmsys_online = False
    f_online = True
    compute_cost_gap = None

    x0_for_train = {'SN': diag_curve, 'S3': manifold_curve}

    nsamp_mc = 10**6
    num_testpoint = 300

    coeff_sgm = 0.5
    c_val = 2

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        assert self.sgmsys_online is False

        self.dim_w = dim_x
        self.dim_u = dim_x

    def b(self, t, x):
        i = torch.arange(self.dim_x, device=x.device)
        return torch.sin(t + i + torch.roll(x, -1, dims=-1))

    def sgm_sys(self, _t, _x, _k, dw):
        # dist = (x - torch.sin(torch.pi * t)).pow(2).mean(-1, keepdim=True)
        # return self.coeff_sgm * torch.tanh(dist) * dw
        return self.coeff_sgm * dw

    def tr_sgm2vxx(self, _t, _x, _v, vxx):
        return self.coeff_sgm**2 * torch.einsum('...ii->...', vxx)

    def mu_sys(self, t, x, k):
        btx = self.b(t, x)
        sgm_k = self.sgm_sys(t, x, None, k)
        return btx + self.c_val * sgm_k

    def mu_pil(self, t, x):
        return self.b(t, x)

    def sgm_pil(self, t, x, dw):
        return self.sgm_sys(t, x, None, dw)

    def g(self, x):
        g1 = 1 + (x**2).mean(-1, keepdim=True)
        g2 = 0.5 * torch.sin(10 * x).mean(-1, keepdim=True)
        return g1 + g2

    def v_term(self, x):
        return torch.log(self.g(x))

    def f(self, _t, _x, k):
        return 0.5 * k.pow(2).sum(-1, keepdims=True)

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [time step, batch, x], or [batch, x]
        # the shape of t should admit the validity of t + x
        assert (t.ndim == 1) or (t.ndim == x.ndim)
        assert x.ndim >= 2

        class LinPDE(LinearPDE):
            musys_online = False
            sgmsys_online = False
            f_online = False
            nsamp_mc = self.nsamp_mc

            def v_term(lpde_self, x):
                return self.g(x)**(-self.c_val**2)

            def f(lpde_self, t, x, _v):
                return torch.zeros_like(x[..., [0]])

            def mu_pil(lpde_self, _t, _x):
                return lpde_self.mu_sys(_t, _x, None)

            def sgm_pil(lpde_self, _t, _x, dw):
                return lpde_self.sgm_sys(_t, _x, None, dw)

            def mu_sys(lpde_self, t, x, _v):
                return self.b(t, x)

            def sgm_sys(lpde_self, t, x, _v, dw):
                return self.sgm_sys(t, x, None, dw)

            def tr_sgm2vxx(lpde_self, t, x, _v, vxx):
                return self.tr_sgm2vxx(t, x, None, vxx)

            def x0_points(lpde_self, num_points):
                return self.x0_points(num_points)

        lin_pde = LinPDE(self.dim_x,
                         t0=self.t0,
                         te=self.te,
                         use_dist=self.use_dist)
        v_inner = lin_pde.v(t, x)
        v_val = -self.c_val**(-2) * torch.log(v_inner)
        return v_val


class HjbHopfCole1aPert1(HJB1a):
    """Variant of HjbHopfCole1a with a perturbation term on self.f.
    Due to this perturbation, self.v(t, x) is nolonger the true solution of this PDE unless c_purb = 0. 
    """
    c_purb = 1.

    def f(self, t, x, k):
        f_val = super().f(t, x, k)
        purb = self.c_purb * torch.sin(k.sum(-1, keepdims=True))
        return f_val + purb


class HjbHopfCole1aPert1div2(HjbHopfCole1aPert1):
    c_purb = 0.5


class HjbHopfCole1aPert1div4(HjbHopfCole1aPert1):
    c_purb = 0.25


class HjbHopfCole1aPert1div8(HjbHopfCole1aPert1):
    c_purb = 0.125


class HjbHopfCole1aPert0(HjbHopfCole1aPert1):
    c_purb = 0.


class HJB1aBmPil(HJB1a):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, t, x, dw):
        return dw


class HJB1b(HJB1a):
    coeff_sgm = 0.025


class HJB2(HJB1a):
    x0_for_train = {'S2': diag_curve, 'S3': manifold_curve}

    def b(self, _t, x):
        return torch.sin(torch.roll(x, -1, dims=-1))

    def sgm_sys(self, t, x, _k, dw):
        distance = (t - 0.5)**2 + x.pow(2).mean(-1, keepdim=True)
        return self.coeff_sgm * torch.tanh(distance) * dw

    def tr_sgm2vxx(self, t, x, _k, vxx):
        distance = (t - 0.5)**2 + x.pow(2).mean(-1, keepdim=True)
        sgm = self.coeff_sgm * torch.tanh(distance)
        return sgm**2 * torch.einsum('...ii->...', vxx)


class HJB2BmPil(HJB2):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, t, x, dw):
        return dw


class HJB2v(HJB2):
    x0_for_train = {'S2': diag_curve}
    num_testpoint = 61
    x0pil_range = 2.
    c_val = 4
