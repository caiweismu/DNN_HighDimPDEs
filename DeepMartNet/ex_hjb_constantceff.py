import torch

from ex_meta import (HJB, diag_curve, free_cache, get_safe_chunksize,
                     origin_point)


class HJBCons0(HJB):
    name = 'HJB_Example0'
    musys_online = True
    sgmsys_online = False
    f_online = True
    nsamp_mc = 10**6

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x
        self.dim_u = dim_x

        self.b_val = torch.tensor(1.)
        self.delta0 = torch.tensor(1 / 4)

    def mu_sys(self, _t, x, k):
        drift = self.b_val
        return drift + 2 * k

    def sgm_sys(self, t, _x, _k, dw):
        c_sgm = self.delta0 * torch.tensor(2.).pow(0.5)
        return c_sgm * dw

    def tr_sgm2vxx(self, _t, _x, _v, vxx):
        c_sgm = self.delta0 * torch.tensor(2.).pow(0.5)
        return c_sgm**2 * torch.einsum('...ii->...', vxx)

    def mu_pil(self, _t, x):
        return torch.full_like(x, self.b_val)

    def sgm_pil(self, t, x, dw):
        return self.sgm_sys(t, x, None, dw)

    def g(self, x):
        x = x - self.b_val
        return torch.log(0.5 * (1 + (x**2).sum(-1, keepdim=True)))

    def v_term(self, x):
        return self.g(x)

    def f(self, _t, _x, k):
        return self.delta0.pow(-2) * k.pow(2).sum(-1, keepdims=True)

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [time step, batch, x], or [batch, x]
        # the shape of t should admit the validity of t + x
        assert (t.ndim == 1) or (t.ndim == x.ndim)
        assert x.ndim >= 2

        c_sgm = self.delta0 * torch.tensor(2.).pow(0.5)
        std_dw = c_sgm * (self.te - t).pow(0.5)
        x_bdt = (x + self.b_val * (self.te - t)).unsqueeze(-2)

        nsamp_mc = self.nsamp_mc
        multiplier = self.dim_w * (1 + std_dw.numel() + x_bdt.numel())
        chunksize = min(
            get_safe_chunksize(multiplier, x.dtype, x.device),
            nsamp_mc,
        )
        cum_size = 0
        cum_mean = 0.
        print(f"Monte-Carlo for Ref. solution on {x.device}...")
        while cum_size < nsamp_mc:
            try:
                chunksize = min(chunksize, nsamp_mc - cum_size)
                rand_mc = torch.normal(
                    mean=0.,
                    std=1.,
                    size=(chunksize, self.dim_w),
                    device=x.device,
                )
                # dw_tte: [axis of t[..., 0], MC samples, dim_w]
                dw_tte = torch.einsum('...j, ij -> ...ij', std_dw, rand_mc)
                del rand_mc
                new_mean = torch.exp(-self.g(x_bdt + dw_tte)).mean(-2)
                del dw_tte
                cum_size += chunksize
                new_rate = chunksize / cum_size
                cum_mean = (1 - new_rate) * cum_mean + new_rate * new_mean
                del new_mean
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

        return -torch.log(cum_mean)


class HJBCons0a(HJBCons0):
    name = 'HJB_Example0a'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.5)


class HJBCons0b(HJBCons0):
    name = 'HJB_Example0b'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.1)


class HJBCons0c(HJBCons0):
    name = 'HJB_Example0c'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.05)


class HJBCons1(HJBCons0):
    '''
    This example is modified from Section 3.1 of
        Bachouch, A., Huré, C., Langrené, N. et al. Deep Neural Networks Algorithms for Stochastic Control Problems on Finite Horizon: Numerical Applications. Methodol Comput Appl Probab 24, 143-178 (2022).
    and Section 4.3 of
        Weinan E, Han J, Jentzen A (2017) Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations.
        In: Communications in mathematics and statistics 5, vol 5, pp 349-380.
    '''
    name = 'HJB_Example1'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.b_val = torch.tensor(0.)
        self.delta0 = torch.tensor(1.)


class HJB1ConsShiftTarget(HJBCons0):
    """
    This example is inspired by section 4.2.3 of 
    @article {MR4793480,
    AUTHOR = {Li, Xingjian and Verma, Deepanshu and Ruthotto, Lars},
     TITLE = {A neural network approach for stochastic optimal control},
   JOURNAL = {SIAM J. Sci. Comput.},
  FJOURNAL = {SIAM Journal on Scientific Computing},
    VOLUME = {46},
      YEAR = {2024},
    NUMBER = {5},
     PAGES = {C535--C556},
      ISSN = {1064-8275,1095-7197},
   MRCLASS = {65M70 (35F21 49K45 49L20 68T07 93-08 93E20)},
  MRNUMBER = {4793480},
       DOI = {10.1137/23M155832X},
       URL = {https://doi.org/10.1137/23M155832X},
}
    """
    name = 'HJBCons_ShiftTarget'
    x0_for_train = {'diag': diag_curve}
    coeff_g = 10.
    compute_cost_gap = 1
    compute_cost_maxit = 300

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._target = torch.full((self.dim_x, ), 3.)
        self.b_val = torch.tensor(0.)
        self.delta0 = torch.tensor(0.1)

    def x0_for_cost(self, num_points):
        return torch.zeros((num_points, self.dim_x))

    def g(self, x):
        x = x - self.b_val - self._target
        g_inner = torch.log(0.5 * (1 + (x**2).sum(-1, keepdim=True)))
        return self.coeff_g * g_inner


class HJBCons2(HJBCons0):
    name = 'HJB_Example2'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.b_val = torch.tensor(1.)
        self.delta0 = torch.tensor(0.1)
        self.delcpi = 0.3 / torch.pi

    def g(self, x):
        x = x - self.b_val
        return (torch.sin(x - torch.pi / 2) +
                torch.sin(1 / (self.delcpi + x.pow(2)))).mean(-1, keepdim=True)


class HJBCons2a(HJBCons2):
    name = 'HJB_Example2a'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.5)


class HJBCons2b(HJBCons2):
    name = 'HJB_Example2b'
    # This example is HJB-3a in the paper arXiv:2408.14395v4
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.2)


class HJBCons2c(HJBCons2):
    name = 'HJB_Example2c'
    # This example is HJB-3c in the paper arXiv:2408.14395v4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.1)


class HJBCons1Onep(HJBCons1):
    name = 'HJB_Example1Onep'
    x0_for_train = {'x0': origin_point}


class HJBCons0b1p(HJBCons0b):
    name = 'HJB_Example0b1p'
    x0_for_train = {'x0': origin_point}


class HJBCons0c1p(HJBCons0c):
    name = 'HJB_Example0c1p'
    x0_for_train = {'x0': origin_point}


class HJBCons2b1p(HJBCons2b):
    name = 'HJB_Example2b1p'
    x0_for_train = {'x0': origin_point}


class HJBCons2c1p(HJBCons2c):
    name = 'HJB_Example2c1p'
    x0_for_train = {'x0': origin_point}


class HJBCons2bPba(HJBCons2b):
    c_purb = 2.0
    name = 'HJB_Example2bPba'
    x0_for_train = {'diag': diag_curve}

    def f(self, _t, _x, k):
        f_val = self.delta0.pow(-2) * k.pow(2).sum(-1, keepdims=True)
        purb = self.c_purb * torch.sin(k.sum(-1, keepdims=True))
        return f_val + purb


class HJBCons2bPbb(HJBCons2bPba):
    c_purb = 1.0
    name = 'HJB_Example2bPbb'


class HJBCons2bPbc(HJBCons2bPba):
    c_purb = 1 / 2
    name = 'HJB_Example2bPbc'


class HJBCons2bPbd(HJBCons2bPba):
    c_purb = 1 / 4
    name = 'HJB_Example2bPbd'


class HJBCons2bPbe(HJBCons2bPba):
    c_purb = 1 / 8
    name = 'HJB_Example2bPbe'


class HJBCons2bPbf(HJBCons2bPba):
    c_purb = 1 / 18
    name = 'HJB_Example2bPbf'


class HJBCons2bPbg(HJBCons2bPba):
    c_purb = 1 / 32
    name = 'HJB_Example2bPbg'
