# Core components of MartNet
import torch
import torch.distributed as dist

from loss_meta import LossCollection


def rdm_vx_prod(v_func: callable,
                t: torch.Tensor,
                x: torch.Tensor,
                dt: torch.Tensor,
                num_samp=1,
                exponential_vx=2,
                shuffle=False):
    samp_shape = torch.Size((num_samp, exponential_vx) + x.shape)
    samp = torch.normal(mean=0.,
                        std=torch.sqrt(dt),
                        size=(samp_shape.numel(), ),
                        device=x.device)
    # reduce variance by antithetic sampling, inspired by deep shotgun
    # https://link.springer.com/article/10.1007/s10915-025-02983-1
    if shuffle:
        samp = torch.concat([samp, -samp], dim=0)
        samp = samp[torch.randperm(2 * samp_shape.numel())]
        samp = samp.reshape((2 * num_samp, ) + samp_shape[1:])
    else:
        samp = samp.reshape(samp_shape)
        samp = torch.concat([samp, -samp], dim=0)
    vnn_on_samp = v_func(t, x + samp)
    prod = (samp * vnn_on_samp).mean(0).prod(0).sum(-1, keepdim=True)
    grad_prod = prod / dt**exponential_vx
    return grad_prod


# def rdm_for_grad(self, xt_pil: torch.Tensor, num_samp=1):
#     xt_pil_cut = xt_pil[:-1]
#     samp_shape = (num_samp, self.problem.exponential_vx) + xt_pil_cut.shape
#     samp = torch.normal(
#         mean=0.,
#         std=torch.sqrt(self.dt),
#         size=samp_shape,
#         device=xt_pil.device,
#     )
#     # reduce variance by antithetic sampling
#     samp = torch.concat([samp, -samp], dim=0)
#     xt_samp = xt_pil_cut + samp
#     vnn_on_samp = self.vnn(self.t_path[:-1], xt_samp)
#     prod = (samp * vnn_on_samp).mean(0).prod(0).sum(-1, keepdim=True)
#     grad_prod = prod / self.dt**self.problem.exponential_vx
#     return grad_prod


class DfSocMartNet(LossCollection):
    name = 'Df-SocMartNet'
    # This class contains the implementation of arXiv:2408.14395v4 for HJB equations.
    # It is also equivalent to random difference method for HJB equations; see arXiv:2506.20308v1.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_vterm = hasattr(self.problem, 'v_term')
        self._rho_val = None
        self._deltam = None

    def init_train(self):
        self.unn = self.net_dict['unn']
        self.vnn = self.net_dict['vnn']
        self.rhonn = self.net_dict['rhonn']
        for net in self.net_dict.values():
            net.train()

    def init_desc(self):
        self.vnn.train()
        self.rhonn.eval()
        self.unn.train()

    def init_asc(self):
        self.vnn.eval()
        self.rhonn.train()
        self.unn.eval()

    # def rdm_for_grad(self, xt_pil: torch.Tensor, num_samp=1):
    #     xt_pil_cut = xt_pil[:-1]
    #     samp_shape = (num_samp, self.problem.exponential_vx) + xt_pil_cut.shape
    #     samp = torch.normal(mean=0.,
    #                         std=torch.sqrt(self.dt),
    #                         size=samp_shape,
    #                         device=xt_pil.device)
    #     # reduce variance by antithetic sampling, inspired by deep shotgun
    #     # https://link.springer.com/article/10.1007/s10915-025-02983-1
    #     samp = torch.concat([samp, -samp], dim=0)
    #     xt_samp = xt_pil_cut + samp
    #     vnn_on_samp = self.vnn(self.t_path[:-1], xt_samp)
    #     prod = (samp * vnn_on_samp).mean(0).prod(0).sum(-1, keepdim=True)
    #     grad_prod = prod / self.dt**self.problem.exponential_vx
    #     return grad_prod

    def ad_for_grad(self, xt_pil: torch.Tensor):
        # just for debugging

        from torch.func import jacrev, vmap
        xtpil_cut = xt_pil[:-1, ...]
        dx_v = vmap(jacrev(self.vnn, argnums=1), in_dims=0)(
            self.t_path[:-1].repeat((1, xtpil_cut.shape[1], 1)).flatten(0, 1),
            xtpil_cut.flatten(0, 1),
        )
        grad_abs = dx_v.reshape(xtpil_cut.shape[:-1] + (-1, )).abs()
        grad_prod = (grad_abs**self.problem.exponential_vx).sum(-1,
                                                                keepdim=True)
        return grad_prod

    def _xt_next(self, t: torch.Tensor, x: torch.Tensor, ctr: torch.Tensor):
        x_next = x
        if self.problem.musys_online is True:
            x_next = x_next + self.problem.mu_sys(t, x, ctr) * self.dt
        if self.problem.sgmsys_online is True:
            dwt_shape = x.shape[:-1] + (self.problem.dim_w, )
            dwt = torch.normal(mean=0.,
                               std=torch.sqrt(self.dt),
                               size=dwt_shape,
                               device=self.rank)
            x_next = x_next + self.problem.sgm_sys(t, x, ctr, dwt)
        return x_next

    def _v_next(self, t_next, x_next):
        if self.has_vterm is True:
            v_next0 = self.vnn(t_next[:-1], x_next[:-1])
            v_te = self.problem.v_term(x_next[-1:])
            v_next = torch.cat((v_next0, v_te), dim=0)
        else:
            v_next = self.vnn(t_next, x_next)
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

        if xtsys_offline is None:
            xtsys_offline = xt_pil[:-1]
        if self.problem.f_online is True:
            f_path = self.problem.f(t, xt, ut)
        else:
            f_path = f_offline[:-1]

        x_next = self._xt_next(t, xtsys_offline, ut)
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

    def loss_mart(self, deltam: torch.Tensor,
                  rho: torch.Tensor) -> torch.Tensor:
        # deltam, rho: [time, path, dim of self]
        # the number of path must be even
        rdm = (rho * deltam).unflatten(1, (-1, 2)).mean([0, 1])
        rdm_det = rdm.detach()
        if self.use_dist is True:
            dist.all_reduce(rdm_det)
            rdm_det = rdm_det / self.world_size

        # Correct the bias caused by DDP and mini-batch sampling
        loss_val = (rdm_det[0] * rdm[1] + rdm_det[1] * rdm[0]).mean()
        return loss_val / 4

    def loss_desc(self, xt_pil, xtsys_offline=None, ft_offline=None):
        self._deltam = None
        if self._rho_val is None:
            with torch.no_grad():
                self._rho_val = self.rhonn(self.t_path[:-1], xt_pil[:-1])

        # deltam_vgrad: [time, path, dim of v]
        deltam_vgrad, deltam_ugrad = self.delta_m(
            xt_pil,
            xtsys_offline=xtsys_offline,
            f_offline=ft_offline,
            compute_ugrad=True,
        )
        if self.problem.ceoff_vx != 0.:
            rdm_vx = rdm_vx_prod(self.vnn,
                                 self.t_path[:-1],
                                 xt_pil[:-1],
                                 self.dt,
                                 num_samp=1,
                                 exponential_vx=self.problem.exponential_vx)
            # rdm_vx = self.rdm_for_grad(xt_pil)
            # ad_vx = self.ad_for_grad(xt_pil)
            deltam_vgrad = deltam_vgrad + self.problem.ceoff_vx * rdm_vx

        mart_loss = self.loss_mart(deltam_vgrad, self._rho_val)
        if deltam_ugrad is None:
            ctr_loss = torch.tensor(0., device=mart_loss.device)
        else:
            ctr_loss = deltam_ugrad.mean()
        loss_tot = mart_loss + ctr_loss

        self.mart_loss = mart_loss.detach()
        self.ctr_loss = ctr_loss.detach()
        return loss_tot

    def loss_asc(self, xt_pil, xtsys_offline=None, ft_offline=None):
        self._rho_val = None
        if self._deltam is None:
            with torch.no_grad():
                self._deltam, _ = self.delta_m(xt_pil,
                                               xtsys_offline=xtsys_offline,
                                               f_offline=ft_offline,
                                               compute_ugrad=False)
        rho_bat = self.rhonn(self.t_path[:-1], xt_pil[:-1])
        loss_test = -self.loss_mart(self._deltam, rho_bat)
        return loss_test

    def log_func(self):
        return {
            'pde_loss': self.mart_loss.abs().item(),
            'ctr_loss': self.ctr_loss.item(),
            # 'pde_loss': 1.0,
            # 'ctr_loss': 1.0,
        }


class SocMartNet(DfSocMartNet):
    name = 'SocMartNet'
    # This class contains the implementation of SOC-MartNet for HJB equations.
    # The loss function is modified to get rid of penalty term for control.
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.problem.sgmsys_online is False, \
            'Currently, SocMartNet is only for HJB equation with sgmsys_online == False'

    def delta_m(self,
                xt_pil,
                xtsys_offline=None,
                f_offline=None,
                compute_ugrad=True):
        # xtsys_offline is not used in this class

        ut = self.unn(self.t_path, xt_pil)
        if self.problem.f_online is True:
            f_path = self.problem.f(self.t_path, xt_pil, ut)
        else:
            f_path = f_offline
        fpath_mean = (f_path[1:] + f_path[:-1]) / 2

        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        if self.has_vterm is True:
            vt = self.vnn(t, xt)
            v_te = self.problem.v_term(xt_pil[-1:])
            v_path = torch.cat((vt, v_te), dim=0)
        else:
            v_path = self.vnn(self.t_path, xt_pil)
            vt = v_path[:-1]
        deltav = (v_path[1:] - vt) / self.dt

        # The following approximates (mu_sys - mu_pil)^{\top} v_x using finite differences instead of automatic differentiation.
        # This approach is adopted because torch.autograd.grad() is incompatible with torch.nn.parallel.DistributedDataParallel (PyTorch 2.6).
        # See, https://pytorch.org/docs/2.6/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
        sz_diff = self.dt**2  # The size of the finite difference
        musys_val = self.problem.mu_sys(t, xt, ut[:-1])
        mupil_val = self.problem.mu_pil(t, xt)
        delta_mu = musys_val - mupil_val
        x_forw = xt + delta_mu * sz_diff
        vforw_vgrad = self.vnn(t, x_forw.detach())
        deltamu_vdx = (vforw_vgrad - vt) / sz_diff

        deltam_vgrad = deltav + deltamu_vdx + fpath_mean.detach()
        if compute_ugrad is True:
            for p in self.vnn.parameters():
                p.requires_grad = False
            vforw_ugrad = self.vnn(t, x_forw)
            for p in self.vnn.parameters():
                p.requires_grad = True

            deltam_ugrad = (vforw_ugrad - vt.detach()) / sz_diff + fpath_mean
            # print(list(self.unn.parameters()))
        else:
            deltam_ugrad = None
        return deltam_vgrad, deltam_ugrad


class QuasiMartNet(SocMartNet):
    name = 'QuasiMartNet'

    # This class contains the implementation of SOC-MartNet for quasi-linear parabolic PDEs.

    def init_train(self):
        self.unn = None
        self.vnn = self.net_dict['vnn']
        self.rhonn = self.net_dict['rhonn']
        for net in self.net_dict.values():
            net.train()

    def init_desc(self):
        self.vnn.train()
        self.rhonn.eval()

    def init_asc(self):
        self.vnn.eval()
        self.rhonn.train()

    def delta_m(self,
                xt_pil,
                f_offline=None,
                xtsys_offline=None,
                compute_ugrad=False):
        # xtsys_offline and compute_ugrad are not used in this class

        ut = self.vnn(self.t_path, xt_pil)
        if self.problem.f_online is True:
            f_path = self.problem.f(self.t_path, xt_pil, ut)
        else:
            f_path = f_offline
        fpath_mean = (f_path[1:] + f_path[:-1]) / 2

        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        if self.has_vterm is True:
            vt = self.vnn(t, xt)
            v_te = self.problem.v_term(xt_pil[-1:])
            v_path = torch.cat((vt, v_te), dim=0)
        else:
            v_path = self.vnn(self.t_path, xt_pil)
            vt = v_path[:-1]
        deltav = (v_path[1:] - vt) / self.dt

        # Approximate (mu_sys - mu_pil)^{\top} v_x by finite difference
        sz_diff = self.dt**2
        musys_val = self.problem.mu_sys(t, xt, ut[:-1])
        mupil_val = self.problem.mu_pil(t, xt)
        delta_mu = musys_val - mupil_val
        x_forw = xt + delta_mu * sz_diff
        vforw_vgrad = self.vnn(t, x_forw)
        deltamu_vdx = (vforw_vgrad - vt) / sz_diff

        deltam_vgrad = deltav + deltamu_vdx + fpath_mean
        return deltam_vgrad, None


class DfQuasiMartNet(DfSocMartNet):
    name = 'Df-QuasiMartNet'
    # This class contains the implementation of arXiv:2408.14395v4 for quasi-linear parabolic PDEs.
    # It is also equivalent to random difference method for quasi-linear parabolic PDEs; see arXiv:2506.20308v1.

    def init_train(self):
        self.unn = None
        self.vnn = self.net_dict['vnn']
        self.rhonn = self.net_dict['rhonn']
        for net in self.net_dict.values():
            net.train()

    def init_desc(self):
        self.vnn.train()
        self.rhonn.eval()

    def init_asc(self):
        self.vnn.eval()
        self.rhonn.train()

    def delta_m(self,
                xt_pil,
                xtsys_offline=None,
                f_offline=None,
                compute_ugrad=None):
        # compute_ugrad is not used in this class

        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        vt = self.vnn(t, xt)
        if self.problem.f_online is True:
            f_path = self.problem.f(t, xt, vt)
        else:
            f_path = f_offline[:-1]

        if xtsys_offline is None:
            xtsys_offline = xt_pil[:-1]
        x_next = self._xt_next(t, xtsys_offline, vt)
        t_next = self.t_path[1:]
        vnext_val = self._v_next(t_next, x_next)
        deltam_val = (vnext_val - vt) / self.dt + f_path
        return deltam_val, None


class DfEvMartNet(DfSocMartNet):
    name = 'Df-EvMartNet'
    # This class remains to be improved, and may contain bugs.

    def init_train(self):
        super().init_train()
        self.problem.register_network(self.vnn)

    def init_desc(self):
        self.vnn.train()
        self.rhonn.eval()

    def init_asc(self):
        self.vnn.eval()
        self.rhonn.train()

    def delta_m(self,
                xt_pil,
                xtsys_offline=None,
                f_offline=None,
                compute_ugrad=None):
        # f_offline and compute_ugrad are not used in this class

        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        vt = self.vnn(t, xt)

        if xtsys_offline is None:
            xtsys_offline = xt_pil[:-1]
        x_next = self._xt_next(t, xtsys_offline, vt)
        t_next = self.t_path[1:]
        # with torch.no_grad():
        vnext = self._v_next(t_next, x_next)
        lv = (vnext - vt) / self.dt
        if hasattr(self.problem, 'update_eigenvalue'):
            self.problem.update_eigenvalue(xt, lv, vt)
        delta_mart = lv + self.problem.f(t, xt, vt)
        if hasattr(self.problem, 'shell_deltamart'):
            delta_mart = self.problem.shell_deltamart(t, xt, vt, delta_mart)

        # delta_mart = vt.detach() * delta_mart / vt.detach().mean()
        # print(f"lambda = {lamb.item()}")
        return delta_mart, None

    def loss_mart(self, deltam: torch.Tensor,
                  rho: torch.Tensor) -> torch.Tensor:
        # deltam, rho: [time, path, dim of self]
        # the number of path must be even
        rdm = (rho * deltam).unflatten(1, (-1, 2)).mean([0, 1])
        rdm_det = rdm.detach()
        if self.use_dist is True:
            dist.all_reduce(rdm_det)
            rdm_det = rdm_det / self.world_size

        # Correct the bias caused by DDP and mini-batch sampling
        loss_val = (rdm_det[0] * rdm[1] + rdm_det[1] * rdm[0]).mean()
        return loss_val / 4
