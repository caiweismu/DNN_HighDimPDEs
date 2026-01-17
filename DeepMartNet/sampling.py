import torch

from ex_meta import PDE, BVP
from loss_meta import make_rate_limiter


def update_paths(path, new_path):
    if new_path is None:
        updated_path = path
    elif path is None:
        updated_path = new_path
    elif new_path.shape[1] >= path.shape[1]:
        updated_path = new_path
    else:
        old_path = path[:, new_path.shape[1]:]
        updated_path = torch.cat((old_path, new_path), dim=1)
    return updated_path


class PathSampler(object):

    def __init__(self,
                 pde: PDE,
                 size_per_epoch: int,
                 num_dt: int = 100,
                 dt: float = None,
                 ctr_func=None,
                 rank=None,
                 rate_newpath=0.2,
                 ip_time_gap=0.):
        self.pde = pde
        self.rank = rank

        self.size_per_epoch = size_per_epoch
        self.epoch = 0.
        self.epoch_finished = 0.
        self.path_idx = torch.randperm(self.size_per_epoch)

        if rate_newpath <= 0.:
            num_newpath = 0
        elif rate_newpath >= 1.0:
            num_newpath = self.size_per_epoch
        else:
            num_newpath = int(self.size_per_epoch * rate_newpath)
            num_newpath = max(1, num_newpath)
        self.num_newpath = num_newpath

        self.num_dt = num_dt
        if dt is not None and (not isinstance(pde, BVP)):
            raise ValueError(
                "The 'dt' argument should only be provided for BVP problems.")

        if not isinstance(pde, BVP):
            te = pde.te
        else:
            te = dt
        self.te = te
        self.dt = (te - pde.t0) / num_dt
        t_path = torch.linspace(pde.t0, te, self.num_dt + 1)
        self.t_path = t_path.unsqueeze(-1).unsqueeze(-1)

        self.t_path = t_path
        self.ctr_func = ctr_func

        self.rate_limiter = make_rate_limiter(print_gap=ip_time_gap)

    def gen_xtpath(self, num_path, ctr_func=None):
        x0 = self.pde.x0_points(num_path)
        x0 = x0[torch.randperm(x0.shape[0])]
        if ctr_func is None:
            xt = self.pde.gen_pilpath(x0, self.te, self.num_dt)
        else:
            xt, _ = self.pde.gen_syspath(x0, self.te, self.num_dt, ctr_func)
        # xt: [t0 to tN, path, dim of x]
        return xt

    def gen_sgmsys_offline(self, t_path, xt_path: torch.Tensor):
        dwt_shape = (xt_path.shape[0], xt_path.shape[1], self.pde.dim_w)
        dwt = torch.normal(mean=0.,
                           std=torch.sqrt(self.dt),
                           size=dwt_shape,
                           device=self.rank)
        sgmt_path = self.pde.sgm_sys(t_path, xt_path, None, dwt)
        return sgmt_path

    def gen_musys_offline(self, t_path, xt_path):
        mut_path = self.pde.mu_sys(t_path, xt_path, None) * self.dt
        return mut_path

    def gen_ft_offline(self, t_path, xt_path):
        return self.pde.f(t_path, xt_path, None)

    def gen_xtpil_offlines(self, num_path, ctr_func=None):
        with torch.no_grad():
            xt_pil = self.gen_xtpath(num_path, ctr_func=ctr_func)

        path_print = ['Xt_pilot']
        xtsys_offline = xt_pil[:-1]
        if self.pde.sgmsys_online is False:
            sgmt_path = self.gen_sgmsys_offline(self.t_path[:-1], xt_pil[:-1])
            xtsys_offline = xtsys_offline + sgmt_path
            path_print.append('mu_offline')
        if self.pde.musys_online is False:
            mut_path = self.gen_musys_offline(self.t_path[:-1], xt_pil[:-1])
            xtsys_offline = xtsys_offline + mut_path
            path_print.append('sigma_offline')

        if self.pde.f_online is False:
            ft_offline = self.gen_ft_offline(self.t_path, xt_pil)
        else:
            ft_offline = None

        if self.rate_limiter():
            print(
                f'Rank {self.rank}: new paths are generated for {path_print}, \nnum_newpath={self.num_newpath}/{self.size_per_epoch}'
            )
        return xt_pil, xtsys_offline, ft_offline

    def get_pathbat(self, bat_size):

        if self.epoch == 0.:
            offlines = self.gen_xtpil_offlines(self.size_per_epoch,
                                               ctr_func=None)
            self.xt_pil = offlines[0]
            self.xtsys_offline = offlines[1]
            self.ft_offline = offlines[2]
        elif (self.epoch - self.epoch_finished) >= 1.:
            # If an entire epoch has be finished, then update the paths
            offlines = self.gen_xtpil_offlines(self.num_newpath,
                                               ctr_func=self.ctr_func)
            self.xt_pil = update_paths(self.xt_pil, offlines[0])
            self.xtsys_offline = update_paths(self.xtsys_offline, offlines[1])
            self.ft_offline = update_paths(self.ft_offline, offlines[2])
            self.epoch_finished += 1.

        bat_idx = self.path_idx[:bat_size]
        xt_bat = self.xt_pil[:, bat_idx]
        if self.xtsys_offline is not None:
            xtsysoff_bat = self.xtsys_offline[:, bat_idx]
        else:
            xtsysoff_bat = None
        if self.ft_offline is not None:
            ftoff_bat = self.ft_offline[:, bat_idx]
        else:
            ftoff_bat = None
        self.epoch += min(bat_size / self.size_per_epoch, 1.)
        self.path_idx = self.path_idx.roll(-bat_size)

        # xt_bat, ftoff_bat: [t_0 to t_N, batch, dim of x]
        # xtsysoff_bat: [t_0 to t_{N-1}, batch, dim of self]
        return xt_bat, xtsysoff_bat, ftoff_bat
