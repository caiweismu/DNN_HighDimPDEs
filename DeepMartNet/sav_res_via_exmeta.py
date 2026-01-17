from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


def is_save_res():
    # determine whether to save results in the current process
    if dist.is_available() and dist.is_initialized():
        is_save = (dist.get_rank() == 0)
    else:
        is_save = True
    return is_save


def tensor2ndarray(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def plot_on_curve(s, v_true, v_approx, sav_prefix, labels=None, ylim=None):
    if labels is None:
        labels = ['True', 'Predicted']
    plt.plot(s, v_true, label=labels[0], color='blue')
    plt.scatter(s, v_approx, label=labels[1], color='red')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(sav_prefix)
    plt.close()


def save_res_oncurve(sav_prefix, curve_name, s, vtrue_x0s, vapprox_x0s):
    res_header = []
    res_col = []
    for (cname, si, vtrue, vappr) in \
            zip(curve_name, s, vtrue_x0s, vapprox_x0s):
        res_header.extend(
            [f'coord_{cname}', f'vtrue_{cname}', f'vappr_{cname}'])
        res_col.extend(
            [tensor2ndarray(si),
             vtrue.squeeze(-1),
             vappr.squeeze(-1)])
    res_arr = np.stack(res_col, axis=1)
    Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
    np.savetxt(sav_prefix + 'res_on_line.csv',
               res_arr,
               delimiter=',',
               header=','.join(res_header))


def plot_2d(x,
            y,
            z,
            sav_prefix,
            xlabel=None,
            ylabel=None,
            zlabel=None,
            ylim=None):
    plt.figure()
    plt.scatter(x, y, c=z, cmap='viridis')
    plt.colorbar(label=zlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(f'{sav_prefix}_2dplot.pdf')
    plt.close()


def modify_ylim(y_min, y_max, rate_edge=0.2, include_zero=True, min_high=1.):
    if y_max - y_min < min_high:
        shift = min_high - (y_max - y_min)
        y_min = y_min - shift / 2
        y_max = y_max + shift / 2

    edge = rate_edge * (y_max - y_min)
    y_min = y_min - edge
    y_max = y_max + edge

    if include_zero is True:
        y_min = min(0, y_min)
        y_max = max(0, y_max)
    return y_min, y_max


def res_on_curve(t,
                 vappr_func,
                 vtrue_func,
                 s,
                 x_s,
                 curve_name,
                 sav_prefix,
                 y_min=None,
                 y_max=None):
    assert len(s) == len(x_s)
    assert len(x_s) == len(curve_name)
    if type(t) is float:
        t_uns = torch.tensor(t).unsqueeze(-1)
    else:
        t_uns = t.unsqueeze(-1)
    vtrue_xs = [tensor2ndarray(vtrue_func(t_uns, x)) for x in x_s]
    vapprox_xs = [tensor2ndarray(vappr_func(t_uns, x)) for x in x_s]

    if (y_min is None) or (y_max is None):
        v_max = np.nanmax([vtrue_xs, vapprox_xs])
        v_min = np.nanmin([vtrue_xs, vapprox_xs])
        v_min, v_max = modify_ylim(v_min, v_max)
        y_min = v_min if y_min is None else y_min
        y_max = v_max if y_max is None else y_max

    if is_save_res():
        save_res_oncurve(sav_prefix, curve_name, s, vtrue_xs, vapprox_xs)
        for i in range(len(s)):
            plot_on_curve(s[i].cpu().numpy(),
                          vtrue_xs[i],
                          vapprox_xs[i],
                          sav_prefix + curve_name[i] + '.pdf',
                          ylim=(y_min, y_max))


def res_on_curve_ver2(vappr_func,
                      vtrue_func,
                      xcurve_gen,
                      sav_prefix,
                      t=0.,
                      y_min=None,
                      y_max=None,
                      type_vappr='plot',
                      type_vtrue='scatter'):
    assert type_vappr in ('plot', 'scatter')
    assert type_vtrue in ('plot', 'scatter')

    if type(t) is float:
        t_uns = torch.tensor(t).unsqueeze(-1)
    else:
        t_uns = t.unsqueeze(-1)
    curve_name, s_true, xs_true = xcurve_gen(num_points=51)
    num_pts = 1001 if type_vappr == 'plot' else 51
    _, s_appr, xs_appr = xcurve_gen(num_points=num_pts)

    s_true = [tensor2ndarray(s) for s in s_true]
    s_appr = [tensor2ndarray(s) for s in s_appr]
    vtrue_xs = [tensor2ndarray(vtrue_func(t_uns, x)) for x in xs_true]
    vapprox_xs = [tensor2ndarray(vappr_func(t_uns, x)) for x in xs_appr]

    if (y_min is None) or (y_max is None):
        v_max = np.nanmax([np.nanmax(v) for v in vtrue_xs + vapprox_xs])
        v_min = np.nanmin([np.nanmin(v) for v in vtrue_xs + vapprox_xs])
        v_min, v_max = modify_ylim(v_min, v_max, include_zero=False)
        y_min = v_min if y_min is None else y_min
        y_max = v_max if y_max is None else y_max

    res_header = []
    res_col = []
    for (cname, strue, vtrue, sappr, vappr) in \
            zip(curve_name, s_true, vtrue_xs, s_appr, vapprox_xs):
        res_header.extend([
            f's_of_{cname}_for_vtrue',
            f'vtrue_{cname}',
            f's_of_{cname}_for_vappr',
            f'vappr_{cname}',
        ])
        res_col.extend([strue, vtrue.squeeze(-1), sappr, vappr.squeeze(-1)])

    if is_save_res():
        Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(res_col).transpose().to_csv(sav_prefix + 'data.csv',
                                                 index=False,
                                                 header=res_header)
        for strue, vtrue, sappr, vappr, cname in zip(
                s_true,
                vtrue_xs,
                s_appr,
                vapprox_xs,
                curve_name,
        ):
            if type_vappr == 'plot':
                plt.plot(sappr, vappr, label='Predicted', color='blue')
            else:
                plt.scatter(sappr, vappr, label='Predicted', color='red')
            if type_vtrue == 'plot':
                plt.plot(strue, vtrue, label='True', color='blue')
            else:
                plt.scatter(strue, vtrue, label='True', color='red')
            plt.ylim(y_min, y_max)
            plt.legend()
            Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(sav_prefix + cname + '.pdf')
            plt.close()


def v_histogram(v_func,
                xcurve_gen,
                sav_prefix,
                t=0.,
                num_pts=1000,
                xlabel='v(t, x)'):
    if type(t) is float:
        t_uns = torch.tensor(t).unsqueeze(-1)
    else:
        t_uns = t.unsqueeze(-1)
    curve_name, _, xs = xcurve_gen(num_points=num_pts)

    v_vals = [tensor2ndarray(v_func(t_uns, x)) for x in xs]

    if is_save_res():
        Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
        for v_on_si, cname in zip(
                v_vals,
                curve_name,
        ):
            plt.hist(v_on_si, bins=30, density=True)
            plt.xlabel(xlabel)
            plt.ylabel('Density')
            plt.savefig(sav_prefix + cname + '.pdf')
            plt.close()


def plot_landscape(
    t0,
    te,
    func,
    t_scatter,
    x_scatter,
    sav_prefix,
    cmap='viridis',
    cbar_label=None,
):
    x_lim0 = torch.min(x_scatter)
    x_lim1 = torch.max(x_scatter)
    x_contourf = torch.linspace(x_lim0, x_lim1, 40)
    t_contourf = torch.linspace(t0, te, 50)
    v_val = torch.stack(
        [func(t.unsqueeze(-1), x_contourf.unsqueeze(-1)) for t in t_contourf])
    v_contourf = v_val.squeeze(-1).transpose(1, 0)
    tcont_np = tensor2ndarray(t_contourf)
    xcont_np = tensor2ndarray(x_contourf)
    vcont_np = tensor2ndarray(v_contourf)
    tscat_np = tensor2ndarray(t_scatter.squeeze())
    xscat_np = tensor2ndarray(x_scatter[..., 0])

    plt.figure(figsize=(5, 3))
    plt.contourf(tcont_np, xcont_np, vcont_np, levels=30, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.plot(
        tscat_np,
        xscat_np,
        marker='o',
        markersize=5,
        color='white',
        markeredgecolor='black',
        linestyle='none',
        alpha=0.6,
        label=r'Samples of $X_t$',
    )
    plt.ylabel(r'$x$')
    plt.xlabel(r'$t$')
    xlim0 = float(t0) - 0.025 * float(te - t0)
    xlim1 = float(te) + 0.025 * float(te - t0)
    plt.xlim(xlim0, xlim1)
    plt.tight_layout()
    if is_save_res():
        plt.savefig(f'{sav_prefix}landscape.pdf')
    plt.close()
