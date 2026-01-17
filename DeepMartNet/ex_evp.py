import math

import torch
import torch.distributed as dist

from ex_meta import EVP, diag_curve, e1_curve, isin_ddp


class EVP1(EVP):
    """
    This example is modified from an Eigenvalue problem considered by 
    https://arxiv.org/abs/2307.11942
    """

    name = 'EigenValueProblem1'
    x0_for_train = {'S1': e1_curve}
    # x0_for_train = {'S2': diag_curve}
    x0pil_range = 3.

    musys_online = False
    sgmsys_online = False
    f_online = True

    true_eigenval = -1.
    rate_newlamb = 0.1
    fourier_frequency = None

    def __init__(self, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.te = 10

        self.it_ev = 0
        self.weight_lamb = 0.

        self.lamb_init = self.true_eigenval + 1
        self.lamb_val = torch.tensor(self.lamb_init)
        self.lamb_from_v = self.lamb_val

    def mu_pil(self, _t, x):
        # return torch.zeros_like(x)
        return x / self.dim_x

    def sgm_pil(self, _t, _x, dw):
        return dw / self.dim_x**(0.5)

    def mu_sys(self, _t, x, _v):
        return x / self.dim_x

    def sgm_sys(self, _t, _x, _v, dw):
        return dw / self.dim_x**(0.5)

    def v(self, _t, x):
        return torch.exp(-x.pow(2).sum(dim=-1, keepdim=True))

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

    # def set_vnn_forward(self, vnn):

    #     def forward(vnn_inst, t, x):
    #         raw_output = vnn_inst.call(t, x)
    #         # x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).pow(2)
    #         output = raw_output
    #         # output = raw_output / (
    #         #     1. + torch.norm(x, p=2, dim=-1, keepdim=True).pow(2))
    #         return output

    #     vnn.forward = types.MethodType(forward, vnn)
    #     return vnn


class EVPFokkerPlanck(EVP):
    #

    name = 'EVPFokkerPlanck'
    # x0_for_train = {'S1': e1_curve}
    x0_for_train = {'S2': diag_curve}
    x0pil_range = math.pi

    musys_online = False
    sgmsys_online = False
    f_online = True

    true_eigenval = 0.
    rate_newlamb = 1.
    fourier_frequency = (1, 5)

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.te = 1.
        # self.ci = torch.linspace(0.1, 1., dim_x)
        self.ci = torch.ones(dim_x) / dim_x

        self.lamb_init = self.true_eigenval + 1
        self.lamb_val = torch.tensor(self.lamb_init)
        self.lamb_from_v = self.lamb_val

    def mu_pil(self, _t, x):
        # return torch.zeros_like(x)
        return self.mu_sys(_t, x, None)

    def sgm_pil(self, _t, _x, dw):
        # return dw / self.dim_x**(0.5)
        return self.sgm_sys(_t, None, None, dw)

    def mu_sys(self, _t, x, _v):
        cos_cicosxi = torch.cos((self.ci * torch.cos(x)).sum(-1, keepdim=True))
        mu_val = -self.ci * torch.sin(x) * cos_cicosxi
        return mu_val

    def dxx_pot(self, x):
        ci_cosxi = self.ci * torch.cos(x)
        sum_ci_cosxi = ci_cosxi.sum(-1, keepdim=True)
        sum_ci2_sin2xi = (self.ci * torch.sin(x)).pow(2).sum(-1, keepdim=True)
        cos_cicosxi = torch.cos(sum_ci_cosxi)
        sin_cicosxi = torch.sin(sum_ci_cosxi)
        dxx_pot_val = -cos_cicosxi * sum_ci_cosxi - sin_cicosxi * sum_ci2_sin2xi
        return dxx_pot_val

    def sgm_sys(self, _t, x, _v, dw):
        return 2**0.5 * dw

    def tr_sgm2vxx(self, _t, _x, _v, vxx):
        return 2 * torch.einsum('...ii->...', vxx)

    def f(self, _t, x, v):
        lamb_val = self.get_lamb()
        lap_pot_val = self.dxx_pot(x)
        f_val = (-lamb_val + lap_pot_val) * v
        return f_val

    def v(self, _t, x):
        sin_cicosxi = torch.sin((self.ci * torch.cos(x)).sum(-1, keepdim=True))
        return torch.exp(-sin_cicosxi)

    # def set_vnn_forward(self, vnn):
    #     # def forward(vnn_inst, t, x):
    #     #     x = torch.cos(x)
    #     #     output = vnn_inst.call(t, x)
    #     #     output = self.softplus(output)
    #     #     return output

    #     # vnn.forward = types.MethodType(forward, vnn)
    #     return vnn

    def additional_loss(self, t_path, x_path, v_approx):
        x0 = self.x0_points(100)
        # x0 = torch.zeros((1, self.dim_x), device=self.t0.device)
        vappr_val = v_approx(None, x0)
        vtrue_val = self.v(None, x0)
        loss_val = (vtrue_val - vappr_val).mean().pow(2)

        # vappr_path = v_approx(t_path, x_path)
        # # self.lamb_from_v, lamb_var = self.comput_lamb(t_path, vappr_path)
        # # loss_val = loss_val + 10*lamb_var
        # xnorm_mask = (x_path.norm(dim=-1, keepdim=True) > 3.)
        # loss_val = loss_val + (xnorm_mask * vappr_path.pow(2)).mean()

        return loss_val

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
                'ev_error': (self.true_eigenval -
                             self.lamb_val).abs().item(),  # *************
                'rel_l1err': l1err.item(),
            }
            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()
            return log

        return log_func
