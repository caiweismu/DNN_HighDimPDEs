# core components for loss functions and training functions
import abc
import math
import time
from abc import abstractmethod
from collections.abc import Callable
from typing import Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from ex_meta import PDE, BVP, isin_ddp


def append_hist(hist: dict, hist_keys, hist_vals):
    if len(hist) == 0:
        hist.update({k: [v] for k, v in zip(hist_keys, hist_vals)})
    else:
        for k, v in zip(hist_keys, hist_vals):
            hist[k].append(v)


def make_gpumem_recoder(device_type: str) -> Callable[[], dict]:
    # produce a function "recoder()" that records the peak gpu memory usage,
    # and resets the peak memory stats after recording.

    if device_type == 'cuda':
        gpu_module = torch.cuda
    elif device_type == 'xpu':
        gpu_module = torch.xpu
    else:
        return lambda: {}

    if isin_ddp():

        num_gpus = gpu_module.device_count()
        current_rank = dist.get_rank()
        gpu_module.reset_peak_memory_stats(current_rank)

        def recoder():
            mem_list = [torch.tensor(0.) for _ in range(num_gpus)]
            mem_max = gpu_module.max_memory_allocated(current_rank) / (1024**2)
            mem_max = torch.tensor(mem_max)
            dist.all_gather(mem_list, mem_max)
            mem_dict = {}
            for i in range(num_gpus):
                key = f'peak_memory_{device_type}{i}_MB'
                mem_dict[key] = mem_list[i].cpu().item()
            gpu_module.reset_peak_memory_stats(current_rank)
            return mem_dict
    else:
        gpu_module.reset_peak_memory_stats()

        def recoder():
            mem_dict = {}
            mem_max = gpu_module.max_memory_allocated() / (1024**2)
            mem_dict[f'peak_memory_{device_type}0_MB'] = mem_max
            gpu_module.reset_peak_memory_stats()
            return mem_dict

    return recoder


def make_rate_limiter(print_gap=1.0) -> Callable[[], bool]:
    """
    Creates a rate limiter that returns True only if enough time (print_gap, in seconds) has elapsed since the last print. 
    This helps control how often certain operations (like logging) happen. If distributed training is in use, 
    it synchronizes the print decision and timestamp across all processes by broadcasting a flag and the updated timestamp.
    """
    last_time = None
    use_dist = isin_ddp()

    def rate_limiter():
        nonlocal last_time
        now = time.time()
        if last_time is None or now - last_time >= print_gap:
            last_time = now
            to_print = True
        else:
            to_print = False
        if use_dist:
            flag = torch.tensor(int(to_print))
            dist.broadcast(flag, src=0)
            to_print = bool(flag.item())
            synced = torch.tensor([last_time if last_time else 0.0],
                                  dtype=torch.float64)
            dist.broadcast(synced, src=0)
            last_time = float(synced.item())
        return to_print

    return rate_limiter


class LossCollection(metaclass=abc.ABCMeta):
    name = 'MetaClassLossCollection'

    def __init__(self,
                 problem: PDE,
                 net_dict: dict[str, nn.Module],
                 num_dt: int = 100,
                 dt=None,
                 use_dist: bool = False,
                 rank: int = 0) -> None:
        self.problem = problem
        if use_dist is True:
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1
        self.use_dist = use_dist
        self.rank = rank
        self.net_dict = net_dict

        self.num_dt = num_dt
        if dt is not None and (not isinstance(problem, BVP)):
            raise ValueError(
                "The 'dt' argument should only be provided for BVP problems.")

        if not isinstance(problem, BVP):
            te = self.problem.te
        else:
            te = dt
        self.dt = (te - self.problem.t0) / num_dt
        t_path = torch.linspace(self.problem.t0, te, self.num_dt + 1)
        self.t_path = t_path.unsqueeze(-1).unsqueeze(-1)

        self.device = self.check_devices()

    def check_devices(self):
        net_devices = [
            net.parameters().__next__().device
            for net in self.net_dict.values()
        ]
        assert all(device == net_devices[0] for device in
                   net_devices), "All networks must be on the same device."
        return net_devices[0]

    def log_func(self) -> dict:
        return {}

    @abstractmethod
    def init_train(self) -> None:
        pass

    def finalize_train(self) -> None:
        for net in self.net_dict.values():
            net.eval()

    def init_desc(self) -> None:
        pass

    def init_asc(self) -> None:
        pass

    @abstractmethod
    def loss_desc(self,
                  xt_pil,
                  xtsys_offline=None,
                  ft_offline=None) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_asc(self,
                 xt_pil,
                 xtsys_offline=None,
                 ft_offline=None) -> Union[torch.Tensor, str]:
        # If there is no asc loss, return 'no asc loss'
        pass


def train(
    loss_collection: LossCollection,
    pathsamp_func: Callable[[int], tuple],
    optim_desc: Union[Optimizer, Sequence[Optimizer]],
    optim_asc: Union[Optimizer, Sequence[Optimizer]],
    batsize: int = 64,
    max_iter: int = 1000,
    step_desc: int = 2,
    step_asc: int = 1,
    schs: Optional[Sequence[LRScheduler]] = None,
    log_func: Optional[Callable[[int], dict]] = None,
    ip_time_gap: float = 0.,
    enable_scaler: bool = True,
    factor_clip_grad: float = 10.,
) -> dict:

    schs = () if schs is None else schs
    if isinstance(optim_desc, Optimizer):
        optim_desc = (optim_desc, )
    if isinstance(optim_asc, Optimizer):
        optim_asc = (optim_asc, )

    log_func = (lambda _it, _t, _x: {}) if log_func is None else log_func
    hist_dict = {}

    # Ensure that the batch size is even for unbiased gradient estimation
    batsize = math.ceil(batsize / 2) * 2
    rt0 = time.time()
    loss_collection.init_train()
    rate_limiter = make_rate_limiter(print_gap=ip_time_gap)
    mem_recoder = make_gpumem_recoder(loss_collection.device.type)
    scaler = GradScaler(device=loss_collection.device, enabled=enable_scaler)

    ema_gradnorm = 1.
    for it in range(max_iter + 1):
        xt_pil, xtsys_offline, ft_offline = pathsamp_func(batsize)

        loss_collection.init_desc()
        for _ in range(step_desc):
            loss_desc = loss_collection.loss_desc(xt_pil,
                                                  xtsys_offline=xtsys_offline,
                                                  ft_offline=ft_offline)

            # if hasattr(loss_collection.problem, 'additional_loss'):
            #     loss_desc = loss_desc + self.problem.additional_loss(
            #         self.t_path, xt_pil, self.vnn)

            if torch.isnan(loss_desc).any():
                raise ValueError(
                    f"NaN detected in loss_asc at iteration {it}. Stopping training."
                )

            scaler.scale(loss_desc).backward()
            for opt in optim_desc:
                scaler.unscale_(opt)
                grad_norm = clip_grad_norm_(
                    opt.param_groups[0]['params'],
                    ema_gradnorm * factor_clip_grad,
                )
                ema_gradnorm = 0.99 * ema_gradnorm + 0.01 * grad_norm
                scaler.step(opt)
                opt.zero_grad()
            scaler.update()

        loss_collection.init_asc()
        for _ in range(step_asc):
            loss_asc = loss_collection.loss_asc(
                xt_pil,
                xtsys_offline=xtsys_offline,
                ft_offline=ft_offline,
            )
            if loss_asc == 'no asc loss':
                continue
            else:
                loss_asc.backward()
                for opt in optim_asc:
                    opt.step()
                    opt.zero_grad()

        for sch in schs:
            sch.step()

        # Log the training information
        rt = time.time() - rt0
        newlog_dict = {}
        with torch.no_grad():
            newlog_dict = log_func(it, loss_collection.t_path, xt_pil)
        newlog_dict.update(loss_collection.log_func())

        # newlog_dict = {k: 1.0 for k, v in newlog_dict.items()}

        mem_dict = mem_recoder()
        new_histkeys = ['it', 'rt'] + list(newlog_dict.keys()) + list(
            mem_dict.keys())
        new_histvals = [it, rt] + list(newlog_dict.values()) + list(
            mem_dict.values())
        append_hist(hist_dict, new_histkeys, new_histvals)

        # Print the log information
        should_print = rate_limiter()
        if (should_print is True) or (it in (0, max_iter)):
            lr = optim_desc[0].param_groups[0]['lr']
            mem_peak = mem_dict.get(
                f'peak_memory_{loss_collection.device.type}{loss_collection.rank}_MB',
                float('nan'))
            pr_str = f"rank: {loss_collection.rank}, peak_gpu_memory: {mem_peak:.5g}\niter step: [{it}/{max_iter}], rt: {rt:.2f}, lr: {lr:.5}\n"
            pr_str += "\n".join([f'{k}: {v}' for k, v in newlog_dict.items()])
            pr_str += '\n'
            print(pr_str)

    loss_collection.finalize_train()
    return hist_dict
