# Networks
from typing import Sequence

import torch
from torch import nn
from torch.amp import autocast
from torch.nn.parameter import Parameter

from ex_meta import isin_ddp


def make_multi_scale_net(
    subnet_generator: callable,
    num_subnets: int,
    step_scale=1.0,
):

    class MultiScaleNN(nn.Module):

        def __init__(self):
            super().__init__()
            subnets = [subnet_generator() for _ in range(num_subnets)]
            self.subnets = nn.ModuleList(subnets)
            self.scales = 1. + step_scale * torch.arange(0, num_subnets)

        def call(self, t, x):
            subnet_output = torch.stack(
                [
                    subnet(t, s * x)
                    for s, subnet in zip(self.scales, self.subnets)
                ],
                dim=-1,
            )
            return subnet_output.mean(dim=-1)

        def forward(self, t, x):
            return self.call(t, x)

    return MultiScaleNN()


def make_var_scale_net(
    backbone_net: callable,
    scale_net: callable,
):

    class VarScaleNN(nn.Module):

        def __init__(self):
            super().__init__()
            self.backbone_net = backbone_net
            self.scale_net = scale_net

        def call(self, t, x):
            scale = self.scale_net(t, x)
            output = self.backbone_net(t, scale * x)
            return output

        def forward(self, t, x):
            return self.call(t, x)

    return VarScaleNN()


def make_inputlayer(dim_in, dim_out, fourier_frequency=None):
    if fourier_frequency is not None:
        min_freq, max_freq = fourier_frequency
        ff_layer = FourierFeatures(min_frequency=min_freq,
                                   max_frequency=max_freq)
        outdim_fflayer = 2 * ff_layer.num_freqs * dim_in
        xlayer = nn.Sequential(
            ff_layer,
            nn.Linear(outdim_fflayer, dim_out),
        )
    else:
        xlayer = nn.Linear(dim_in, dim_out)
    return xlayer


def make_hidden_layers(
    xdims,
    act_func,
    layer_norm: bool = False,
    batch_norm: bool = False,
):
    layers = []
    if layer_norm is True:
        layers.append(nn.LayerNorm(xdims[1]))
    num_hidden = len(xdims) - 2

    if num_hidden > 0:
        layers.append(act_func())
        for i in range(1, num_hidden):
            layers.append(nn.Linear(xdims[i], xdims[i + 1]))
            if batch_norm is True:
                layers.append(nn.BatchNorm1d(xdims[i + 1]))
            if layer_norm is True:
                layers.append(nn.LayerNorm(xdims[i + 1]))
            layers.append(act_func())
        layers.append(nn.Linear(xdims[-2], xdims[-1]))
    else:
        layers.append(nn.Identity())

    nn_seq = nn.Sequential(*layers)
    if isin_ddp() and batch_norm:
        nn_seq = nn.SyncBatchNorm.convert_sync_batchnorm(nn_seq)

    return nn.Sequential(*layers)


class FourierFeatures(nn.Module):

    def __init__(
        self,
        min_frequency: int = 1,
        max_frequency: int = 10,
    ):
        super().__init__()
        self.freqs = nn.Parameter(
            1.0 * torch.arange(min_frequency, max_frequency + 1))
        self.num_freqs = self.freqs.shape[0]

    def forward(self, x):
        x_proj = torch.einsum('...i,j->...ij', x, self.freqs)
        sin_val = torch.sin(x_proj).flatten(start_dim=-2, end_dim=-1)
        cos_val = torch.cos(x_proj).flatten(start_dim=-2, end_dim=-1)
        return torch.cat([sin_val, cos_val], dim=-1)


class DNNx(nn.Module):

    def __init__(self,
                 xdims: Sequence,
                 shell_func=None,
                 act_func=nn.ReLU,
                 layer_norm=False,
                 batch_norm=False,
                 scale_ub=1.0,
                 scale_lb=1.0,
                 fourier_frequency=None,
                 enable_autocast=False,
                 autocast_dtype=torch.float16):
        # xdims: Sequence of ints, or [int, int, ..., int, int].
        # shell_func: (x, layer_output) -> dnn_output

        super().__init__()
        dim_x = xdims[0]
        dim_out = xdims[-1]
        self.dim_x = dim_x
        self.dim_out = dim_out

        self.enable_autocast = enable_autocast
        if enable_autocast:
            self.autocast_dtype = autocast_dtype
        else:
            self.autocast_dtype = None

        self.xlayer = make_inputlayer(
            xdims[0],
            xdims[1],
            fourier_frequency=fourier_frequency,
        )
        self.hidden_layers = make_hidden_layers(
            xdims,
            act_func=act_func,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
        )

        if (scale_ub != 1.0) or (scale_lb != 1.0):
            w = torch.linspace(scale_lb, scale_ub, dim_out)
            self.scale_layer = lambda z: w * z
        else:
            self.scale_layer = lambda z: z

        if shell_func is None:
            self.shell_func = self.__default_shellfunc
        else:
            self.shell_func = shell_func

    def __default_shellfunc(self, _x, y):
        return y

    def call(self, _t, x):
        
        if x.ndim > 1:
            # flatten for the using of BatchNorm1d
            batch_dim = x.shape[:-1]
            flat_x = x.flatten(end_dim=-2)
        else:
            batch_dim = ()
            flat_x = x
        y = self.xlayer(flat_x)
        y = self.hidden_layers(y)
        y = self.scale_layer(y)
        y = self.shell_func(flat_x, y)
        if x.ndim > 1:
            y = y.unflatten(0, batch_dim)
        return y

    def forward(self, _t, x):
        with autocast(x.device.type,
                      enabled=self.enable_autocast,
                      dtype=self.autocast_dtype):
            y = self.call(_t, x)
        return y.float()

    @property
    def module(self):
        return self


class EigenFuncValue(DNNx):

    def __init__(self, xdims, init_lamb=1.0, **kwargs):
        super().__init__(xdims, **kwargs)
        self.lamb = Parameter(torch.tensor(init_lamb))

    def eigenfunc_parameters(self):
        params = [p for n, p in self.named_parameters() if n != "lamb"]
        return params

    def eigenval_parameters(self):
        return [self.lamb]


class DNNtx(DNNx):

    def __init__(self, xdims: list, **kwargs):
        super().__init__(xdims, **kwargs)
        self.tlayer = nn.Linear(1, xdims[1])

    def call(self, t, x):
        tx = self.tlayer(t) + self.xlayer(x)
        if tx.ndim > 1:
            # flatten for the using of BatchNorm1d
            flat_tx = tx.flatten(end_dim=-2)
            batch_dim = tx.shape[:-1]
            flat_x = x.flatten(end_dim=-2)
        else:
            flat_tx = tx
            batch_dim = ()
            flat_x = x
        ltx = self.hidden_layers(flat_tx)
        y = self.shell_func(flat_x, self.scale_layer(ltx))
        if tx.ndim > 1:
            y = y.unflatten(0, batch_dim)
        return y
