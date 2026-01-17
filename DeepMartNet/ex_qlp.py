# Some examples of PDE and SOCP
import torch

import ex_meta
from ex_meta import PDEwithVtrue
from ex_hjb_hopfcole import HJB0a, HJB0_Rosenbrock


class QLP1(PDEwithVtrue):
    '''
    This example and its derived class are modified from 
    title = {Tackling the curse of dimensionality with physics-informed neural networks},
    journal = {Neural Networks},
    author = {Zheyuan Hu and Khemraj Shukla and George Em Karniadakis and Kenji Kawaguchi},
    volume = {176},
    pages = {106369},
    year = {2024},
    doi = {https://doi.org/10.1016/j.neunet.2024.106369}.
    '''
    name = 'QLP-1'
    musys_online = False
    sgmsys_online = True
    f_online = True

    x0_for_train = {'S2': ex_meta.diag_curve}
    x0pil_range = 1.5

    coeff_diffusion = 1.0

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

        assert dim_x >= 3, "dim_x should be at least 3 for this example"

        # The definition of self.ci differs from the example used in SDGD,
        # as our test typically involves solving v(0, x)
        # for x located on the diagonal of a cube, rather than within a ball.
        self.ci = self.coeff_of_v()

    def coeff_of_v(self):
        xi = torch.linspace(-torch.pi / 2, torch.pi / 2, self.dim_x)
        ci = (1.5 + torch.sin(xi)) / self.dim_x
        return ci

    @staticmethod
    def x_shift(t, x):
        return t + x - torch.tensor(0.5)

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return (2**0.5 * self.coeff_diffusion) * dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, x, v, dw):
        return self.coeff_diffusion * v * dw

    def sgm(self, _t, x, v):
        return v.unsqueeze(-2) * (self.coeff_diffusion * torch.eye(self.dim_x))

    def tr_sgm2vxx(self, t, x, v, vxx):
        return self.coeff_diffusion**2 * v**2 * torch.einsum('...ii->...', vxx)

    def v(self, t, x):
        x = self.x_shift(t, x)
        cosx = torch.cos(x)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        s = torch.sin(x + cosx_r1 + x_r1 * cosx)
        return (self.ci * s).sum(-1, keepdims=True)

    def v_term(self, x):
        return self.v(self.te, x)

    def derivative_of_v(self, t, x):
        zeta = self.x_shift(t, x)
        sinz = torch.sin(zeta)
        cosz = torch.cos(zeta)

        z_r1 = torch.roll(zeta, -1, dims=-1)
        sinz_r1 = torch.roll(sinz, -1, dims=-1)
        cosz_r1 = torch.roll(cosz, -1, dims=-1)

        ai = zeta + cosz_r1 + z_r1 * cosz
        bi = 1. - z_r1 * sinz
        di = -sinz_r1 + cosz

        sin_ai = torch.sin(ai)
        cos_ai = torch.cos(ai)

        ci_cosai = self.ci * cos_ai
        dvdx = ci_cosai * bi + (ci_cosai * di).roll(1, dims=-1)

        v = (self.ci * sin_ai).sum(-1, keepdims=True)

        dv_dxidxi_1 = -self.ci * (sin_ai * bi.pow(2) + cos_ai * z_r1 * cosz)
        cosai_coszir1 = cos_ai * cosz.roll(-1, dims=-1)
        sinai_di2 = sin_ai * di.pow(2)
        dv_dxidxi_2 = -self.ci * (cosai_coszir1 + sinai_di2)
        dv_dxidxi = dv_dxidxi_1 + dv_dxidxi_2.roll(1, dims=-1)

        dv_dxidxir1 = -self.ci * (sin_ai * bi * di + cos_ai * sinz)
        return v, dvdx, dv_dxidxi, dv_dxidxir1

    def d_v(self, t, x):
        v, dvdx, dv_dxidxi, _ = self.derivative_of_v(t, x)
        sum_dvdx = dvdx.sum(-1, keepdims=True)
        diffusion = v.pow(2) * dv_dxidxi.sum(-1, keepdims=True)
        d_v = sum_dvdx + 0.5 * self.coeff_diffusion**2 * diffusion

        # diffusion_ad, hess, d0_v = self.dv_ad(t, x)

        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        return v - v.pow(3) - dv_val - v_true + v_true.pow(3)

    def dv_ad(self, t, x):
        # just for debugging

        from torch.func import jacrev, vmap
        assert t.ndim == x.ndim

        batch_dims = x.shape[:-1]
        t = (t * torch.ones_like(x[..., :1])).flatten(0, 1)
        x = x.flatten(0, 1)

        dt, dx = vmap(jacrev(self.v, argnums=(0, 1)), in_dims=0)(t, x)
        hess = vmap(
            jacrev(jacrev(self.v, argnums=1), argnums=1),
            in_dims=0,
        )(t, x)

        v_val = self.v(t, x)
        mu_val = self.mu_sys(t, x, v_val).unsqueeze(-2)
        sgm_val = self.sgm(t, x, v_val)
        diffusion = torch.einsum(
            '...ik, ...jk, ...pij -> ...p',
            sgm_val,
            sgm_val,
            hess,
        )
        dt_mudx = dt.squeeze(-1) + (mu_val * dx).sum(-1)
        d0_v = dt_mudx + 0.5 * diffusion

        d0_v = d0_v.unflatten(0, batch_dims)
        dt_mudx = dt_mudx.unflatten(0, batch_dims)
        # dt = dt.unflatten(0, batch_dims).squeeze(-1)
        # dx = dx.unflatten(0, batch_dims)
        diffusion = diffusion.unflatten(0, batch_dims)
        hess = hess.unflatten(0, batch_dims)
        return diffusion, hess, d0_v


class QLP1DiffusionA(QLP1):
    coeff_diffusion = 0.1


class QLP1DiffusionB(QLP1DiffusionA):
    coeff_diffusion = 0.2


class QLP1DiffusionC(QLP1DiffusionA):
    coeff_diffusion = 0.5


class QLP1DiffusionD(QLP1DiffusionA):
    coeff_diffusion = 1.0


class QLP1DiffusionE(QLP1DiffusionA):
    coeff_diffusion = 2.0


class QLP1DiffusionF(QLP1DiffusionA):
    coeff_diffusion = 3.0


class QLP1DiffusionG(QLP1DiffusionA):
    coeff_diffusion = 4.0


class QLP1DiffusionH(QLP1DiffusionA):
    coeff_diffusion = 5.0


class QLP1BmPil(QLP1):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, _x, dw):
        return dw


class QLP2a(QLP1):
    x0_for_train = {'S2': ex_meta.diag_curve}

    name = 'QLP-2a'
    musys_online = True
    sgmsys_online = True
    f_online = True

    def mu_sys(self, _t, x, v):
        return -1. + 0.5 * v * torch.ones_like(x)

    def sgm_sys(self, _t, x, v, dw):
        return v * torch.ones_like(x) * dw

    def tr_sgm2vxx(self, t, x, v, vxx):
        return v**2 * torch.einsum('...ii->...', vxx)

    def d_v(self, t, x):
        v, dvdx, dv_dxidxi, _ = self.derivative_of_v(t, x)
        sum_dvdx = dvdx.sum(-1, keepdims=True)
        drift = 0.5 * v * sum_dvdx
        diffusion = v.pow(2) * dv_dxidxi.sum(-1, keepdims=True)

        d_v = drift + 0.5 * diffusion

        # diffusion_ad, hess, d0_v = self.dv_ad(t, x)
        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        f_val = v.pow(2) - v_true.pow(2) - dv_val
        return f_val


class QLP2b(QLP2a):
    name = 'QLP-2b'
    musys_online = True
    sgmsys_online = True
    f_online = True

    def mu_sys(self, _t, x, v):
        return -1 / self.dim_x + v * torch.full_like(x, 0.5 / self.dim_x)

    def d_v(self, t, x):
        v, dvdx, dv_dxidxi, dv_dxidxir1 = self.derivative_of_v(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        sgm_1 = (v * sinx).pow(2).mean(-1, keepdims=True)
        sgm_2 = v * sinx.mean(-1, keepdims=True)
        sgm2_ii = cosx.pow(2) + sgm_1 + sgm_2 * (2 * cosx)
        sgm2_i_ir1 = cosx * cosx_r1 + sgm_1 + sgm_2 * (cosx + cosx_r1)

        sgm2_dv_dxidxi = (sgm2_ii * dv_dxidxi).mean(-1, keepdim=True)
        sgm2_dv_dxi_dxir1 = (sgm2_i_ir1 * dv_dxidxir1).mean(-1, keepdim=True)
        diffusion = sgm2_dv_dxidxi + 2 * sgm2_dv_dxi_dxir1

        sum_dvdx = dvdx.sum(-1, keepdims=True)
        drift = (-1 + 0.5 * v) * sum_dvdx / self.dim_x
        d_v = sum_dvdx + drift + 0.5 * diffusion
        # diffusion_ad, hess, d0_v = self.dv_ad(t, x)
        return d_v, v

    def sgm(self, _t, x, v):
        val = torch.cos(x).unsqueeze(-1) + (v * torch.sin(x)).unsqueeze(-2)
        return val / self.dim_x

    def sgm_sys(self, _t, x, v, dw):
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        val = cosx * dw.mean(-1, keepdims=True)
        val = val + v * (sinx * dw).mean(-1, keepdims=True)
        return val

    def tr_sgm2vxx(self, _t, x, v, vxx):
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        sgm = (cosx.unsqueeze(-1) + v * sinx) / self.dim_x
        return torch.einsum('...kij, ...i, ...j->...k', vxx, sgm, sgm)

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        f_val = v.pow(2) - v_true.pow(2) - dv_val
        return f_val


class QLP2c(QLP2b):
    """
    Variant of QLP2b where the drift term is 'not' normalized by the dimensionality (scaled by 1 / dim_x).
    """
    name = 'QLP-2c'

    def mu_sys(self, _t, x, v):
        return -1. + 0.5 * v * torch.ones_like(x)

    def d_v(self, t, x):
        v, dvdx, dv_dxidxi, dv_dxidxir1 = self.derivative_of_v(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        sgm_1 = (v * sinx).pow(2).mean(-1, keepdims=True)
        sgm_2 = v * sinx.mean(-1, keepdims=True)
        sgm2_ii = cosx.pow(2) + sgm_1 + sgm_2 * (2 * cosx)
        sgm2_i_ir1 = cosx * cosx_r1 + sgm_1 + sgm_2 * (cosx + cosx_r1)

        sgm2_dv_dxidxi = (sgm2_ii * dv_dxidxi).mean(-1, keepdim=True)
        sgm2_dv_dxi_dxir1 = (sgm2_i_ir1 * dv_dxidxir1).mean(-1, keepdim=True)
        diffusion = sgm2_dv_dxidxi + 2 * sgm2_dv_dxi_dxir1

        sum_dvdx = dvdx.sum(-1, keepdims=True)
        drift = 0.5 * v * sum_dvdx
        d_v = drift + 0.5 * diffusion

        # diffusion_ad, hess, d0_v = self.dv_ad(t, x)
        return d_v, v


class QLP_HJB0a(PDEwithVtrue):
    # HJB equation with explicit inf H provided to algorithms

    name = 'QLP_Version_of_HJB0a'
    musys_online = False
    sgmsys_online = False
    f_online = False

    x0_for_train = {'SN': ex_meta.standard_normal}
    x0pil_range = 1.
    num_testpoint = 1000

    HjbClass = HJB0a
    exponential_vx = 2
    ceoff_vx = -0.5 * (HjbClass.c_val * HjbClass.coeff_sgm)**exponential_vx

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.hjb_equ = self.HjbClass(dim_x, **kwargs)
        # self.hjb_equ.nsamp_mc = 100  # only for debugging

    def mu_pil(self, t, x):
        return self.hjb_equ.b(t, x)

    def sgm_pil(self, t, x, dw):
        return self.hjb_equ.sgm_pil(t, x, dw)

    def mu_sys(self, t, x, _v):
        return self.hjb_equ.b(t, x)

    def sgm_sys(self, t, x, v, dw):
        return self.hjb_equ.sgm_sys(t, x, v, dw)

    def tr_sgm2vxx(self, t, x, v, vxx):
        return self.hjb_equ.tr_sgm2vxx(t, x, v, vxx)

    def v(self, t, x):
        return self.hjb_equ.v(t, x)

    def v_term(self, x):
        return self.hjb_equ.v_term(x)

    def f(self, _t, _x, _v):
        return torch.zeros_like(_x[..., [0]])


class QLP_HJB0_Rosenbrock(QLP_HJB0a):
    name = 'QLP_Version_of_HJB0_Rosenbrock'
    HjbClass = HJB0_Rosenbrock
    num_testpoint = 1000
