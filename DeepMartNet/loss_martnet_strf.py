import torch
import torch.distributed as dist

from loss_meta import LossCollection
from loss_martnet import rdm_vx_prod


class SocRdmStrForm(LossCollection):
    name = 'RDM-in-Strong-Formulation for HJB PDE'
    # This class contains the implementation of the strong-form random difference method for HJB equations; see arXiv:2506.20308v1. 
    
    def __init__(self,
                 *args,
                 num_rdmsamp: int = 32,
                 antithetic_variable: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.has_vterm = hasattr(self.problem, 'v_term')

        assert num_rdmsamp % 2 == 0, "num_rdmsamp must be even."
        self.num_rdmsamp = num_rdmsamp
        self.antithetic_variable = antithetic_variable

    def init_train(self):
        self.unn = self.net_dict['unn']
        self.vnn = self.net_dict['vnn']
        for net in self.net_dict.values():
            net.train()

    def _xt_next(self, t: torch.Tensor, x: torch.Tensor, ctr: torch.Tensor):
        x_next = x
        x_next = x_next + self.problem.mu_sys(t, x, ctr) * self.dt
        dwt_shape = torch.Size((int(self.num_rdmsamp // 2), 2) + x.shape[:-1] +
                               (self.problem.dim_w, ))
        dwt = torch.normal(mean=0.,
                           std=torch.sqrt(self.dt),
                           size=dwt_shape,
                           device=x.device)
        if self.antithetic_variable is True:
            dwt = torch.concat([dwt, -dwt], dim=0)
        x_next = x_next + self.problem.sgm_sys(t, x, ctr, dwt)
        return x_next

    def _v_next(self, t_next, x_next):
        # t_next: [time, path, 1]
        # x_next: [samp of rdm, dim of prod, time, path, dim of x]
        # return: [dim of prod, time, path, 1]
        if self.has_vterm is True:
            v_next0 = self.vnn(t_next[:-1], x_next[:, :, :-1])
            v_te = self.problem.v_term(x_next[:, :, -1:])
            v_next = torch.cat((v_next0, v_te), dim=2)
        else:
            v_next = self.vnn(t_next, x_next)

        v_next = v_next.mean(0)
        return v_next

    def delta_m(self,
                xt_pil,
                xtsys_offline=None,
                f_offline=None,
                compute_ugrad=True):
        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        ut = self.unn(t, xt)
        vt = self.vnn(t, xt)

        if self.problem.f_online is True:
            f_path = self.problem.f(t, xt, ut)
        else:
            f_path = f_offline[:-1]

        x_next = self._xt_next(t, xt, ut)
        t_next = self.t_path[1:]
        vnext_ud = self._v_next(t_next, x_next.detach())
        deltam_vgrad = (vnext_ud - vt) / self.dt + f_path.detach()

        if compute_ugrad is True:
            for p in self.vnn.parameters():
                p.requires_grad = False
            vnext_vd = self._v_next(t_next, x_next)
            for p in self.vnn.parameters():
                p.requires_grad = True
            deltam_ugrad = (vnext_vd - vt.detach()) / self.dt + f_path
        else:
            deltam_ugrad = None
        return deltam_vgrad, deltam_ugrad

    def loss_mart(self, deltam: torch.Tensor) -> torch.Tensor:
        # deltam, rho: [time, path, dim of self]
        # the number of path must be even
        deltam_detach = deltam.detach()
        if self.use_dist is True:
            dist.all_reduce(deltam_detach)
            deltam_detach = deltam_detach / self.world_size

        # Correct the bias caused by DDP and mini-batch sampling
        loss_val = (deltam_detach[0] * deltam[1] +
                    deltam_detach[1] * deltam[0]).mean() / 4
        return loss_val

    def loss_desc(self, xt_pil, xtsys_offline=None, ft_offline=None):
        self._deltam = None

        # deltam_vgrad: [rdm sample, time, path, dim of v]
        deltam_vgrad, deltam_ugrad = self.delta_m(
            xt_pil,
            xtsys_offline=None,
            f_offline=ft_offline,
            compute_ugrad=True,
        )

        if self.problem.ceoff_vx != 0.:
            # ad_vx = self.ad_for_grad(xt_pil)
            args = (self.vnn, self.t_path[:-1], xt_pil[:-1], self.dt)
            kwargs = {
                'num_samp': int(self.num_rdmsamp // 2),
                'exponential_vx': self.problem.exponential_vx
            }
            rdm_vx1 = rdm_vx_prod(*args, **kwargs)
            rdm_vx2 = rdm_vx_prod(*args, **kwargs)
            rdm_vx = torch.stack([rdm_vx1, rdm_vx2], dim=0)
            deltam_vgrad = deltam_vgrad + self.problem.ceoff_vx * rdm_vx

        mart_loss = self.loss_mart(deltam_vgrad)
        if deltam_ugrad is None:
            ctr_loss = torch.tensor(0., device=mart_loss.device)
        else:
            ctr_loss = deltam_ugrad.mean()
        loss_tot = mart_loss + ctr_loss

        self.mart_loss = mart_loss.detach()
        self.ctr_loss = ctr_loss.detach()
        return loss_tot

    def loss_asc(self, xt_pil, xtsys_offline=None, ft_offline=None):
        return 'no asc loss'

    def log_func(self):
        return {
            'pde_loss': self.mart_loss.abs().item(),
            'ctr_loss': self.ctr_loss.item(),
        }


class QuasiRdmStrForm(SocRdmStrForm):
    name = 'RDM-in-Strong-Formulation for Quasi-linear PDE'

    # This class contains the implementation of the strong-form random difference method for quasi-linear parabolic PDEs; see arXiv:2506.20308v1. 
    # It is a variant of the shotgun method \cite{zhang2025shotgun}, differing mainly in its offline path sampling and in not using antithetic variates for variance reduction. 
    #     @article{zhang2025shotgun,
    #   author    = {Xu, Wenjun and Zhang, Wenzhong},
    #   title     = {A Deep Shotgun Method for Solving High-Dimensional Parabolic Partial Differential Equations},
    #   fjournal   = {Journal of Scientific Computing},
    #   journal   = {J. Sci. Comput.},
    #   year      = {2025},
    #   volume    = {104},
    #   number    = {2},
    #   pages     = {69},
    #   issn      = {1573-7691},
    #   doi       = {10.1007/s10915-025-02983-1},
    #   url       = {https://doi.org/10.1007/s10915-025-02983-1},
    # }
    
    def init_train(self):
        self.vnn = self.net_dict['vnn']
        for net in self.net_dict.values():
            net.train()

    def delta_m(self,
                xt_pil,
                xtsys_offline=None,
                f_offline=None,
                compute_ugrad=None):
        # compute_ugrad, xtsys_offline, f_offline are not used

        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        vt = self.vnn(t, xt)
        f_path = self.problem.f(t, xt, vt)

        x_next = self._xt_next(t, xt, vt)
        t_next = self.t_path[1:]
        vnext_val = self._v_next(t_next, x_next)
        deltam_val = (vnext_val - vt) / self.dt + f_path
        return deltam_val, None
