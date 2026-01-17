from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_error_path(t_grid, path_on_t, labels, sav_name):
    cmap = plt.get_cmap('tab10')(np.arange(len(labels)))
    markers = cycle(('s', 'D', '^', 'v', 'x', '+'))
    for path_i, lab, cm in zip(path_on_t, labels, cmap):
        plt.plot(t_grid, path_i, color=cm, label=lab, marker=next(markers))
    if len(labels) > 1:
        plt.legend()
    # plt.xlabel('$t$')
    # plt.ylabel('Relative error')
    ymax = max([pathi.max() for pathi in path_on_t])
    plt.ylim(top=min(2 * ymax, 1.))
    Path(sav_name).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(f'{sav_name}path.pdf')
    plt.close()


def plot_vtx_path(t_grid, vt_true, vt_appr, sav_name):

    c_list = plt.get_cmap('tab10')(np.arange(2))
    for i in range(vt_true.shape[1]):
        if (i == 0) and (vt_appr is not None):
            labels = ("Predicted $v(t, X_t)$", "Exact $v(t, X_t)$")
        else:
            labels = (None, None)
        if vt_appr is not None:
            plt.plot(t_grid, vt_appr[:, i], color=c_list[1], label=labels[0])
        plt.plot(t_grid, vt_true[:, i], color=c_list[0], label=labels[1])
        plt.scatter(t_grid[0],
                    vt_true[0, i],
                    color='black',
                    label=None,
                    marker='s')
        plt.scatter(t_grid[-1],
                    vt_true[-1, i],
                    color='black',
                    label=None,
                    marker='o')
    if any(labels):
        plt.legend()
    Path(sav_name).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(f'{sav_name}vtx_path.pdf')
    plt.close()


def plot_hist(sav_name,
              x_arr_list,
              y_arr_list,
              std_arr_list=None,
              lable_list=None,
              color_list=None,
              xlabel=None,
              ylabel=None,
              alpha=1.0,
              yscal='log',
              linewidth=None,
              ylim0=None,
              ylim1=None,
              two_side_std=False,
              figsize=None):
    num_arrs = len(x_arr_list)

    if lable_list is None:
        lable_list = [None] * num_arrs
    if all(lable_list) is False:
        plot_legend = False
    else:
        plot_legend = True
    if color_list is None:
        color_list = plt.get_cmap('tab10')(np.arange(num_arrs))

    if std_arr_list is None:
        std_arr_list = [None] * num_arrs
    #     hatches = cycle([None] * num_arrs)
    # else:
    #     hatches = cycle(['/', '\\', '|', '-', '+', 'x'])
    plt.figure(figsize=figsize)
    plt.yscale(yscal)  # affects mem. leak
    plt.xlabel(xlabel)  # affects mem. leak
    plt.ylabel(ylabel)  # affects mem. leak
    for x_arr, y_arr, std_arr, label, color in zip(
            x_arr_list,
            y_arr_list,
            std_arr_list,
            lable_list,
            color_list,
    ):
        plt.plot(x_arr,
                 y_arr,
                 label=label,
                 alpha=alpha,
                 color=color,
                 linewidth=linewidth)
        if std_arr is not None:
            yup_arr = y_arr + 2 * std_arr
            if two_side_std is True:
                ylow_arr = y_arr - 2 * std_arr
            else:
                ylow_arr = y_arr
            plt.fill_between(x_arr,
                             ylow_arr,
                             yup_arr,
                             facecolor=color,
                             alpha=0.5 * alpha)
    plt.ylim(ylim0, ylim1)  # affects mem. leak
    plt.tight_layout()
    if plot_legend:
        plt.legend()
    Path(sav_name).parent.mkdir(exist_ok=True, parents=True)
    plt.grid(True)
    plt.savefig(sav_name)
    plt.close()


def plot_l1linf_error(hist_df, sav_path):
    l1err = hist_df['rel_l1err']
    linf_err = hist_df['rel_linferr']
    plot_hist(sav_path + 'error_hist.pdf', [hist_df['it'].to_numpy()] * 2,
              (linf_err, l1err),
              lable_list=(r"RE$_{\infty}$", r"RE$_1$"),
              alpha=1.0)


def plot_l1_error(hist_df, sav_path):
    l1_err = hist_df['rel_l1err']
    plot_hist(sav_path + 'error_hist.pdf', (hist_df['it'].to_numpy(), ),
              (l1_err, ),
              lable_list=None,
              alpha=1.0)


def ylim_for_g1loss(g1_hist):
    y_median = np.median(g1_hist)
    iqr = np.quantile(g1_hist, 0.75) - np.quantile(g1_hist, 0.25)
    ylim0 = y_median - 10 * max(iqr, 0.05)
    ylim1 = y_median + 10 * max(iqr, 0.05)
    return ylim0, ylim1


def save_hist(hist_dict, sav_prefix):
    Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)

    # save and load to avoid memory leak
    pd.DataFrame(hist_dict).to_csv(f'{sav_prefix}log.csv', index=False)
    hist_df = pd.read_csv(f'{sav_prefix}log.csv')
    it_arr = np.array(hist_df['it'])

    if 'pde_loss' in hist_df:
        plot_hist(
            sav_prefix + 'pde_loss_hist.pdf',
            (it_arr, ),
            (hist_df['pde_loss'], ),
        )

    if 'ctr_loss' in hist_df:
        g1_hist = hist_df['ctr_loss'].to_numpy()
        ylim0, ylim1 = ylim_for_g1loss(g1_hist)
        plot_hist(sav_prefix + 'ctr_loss.pdf', (it_arr, ), (g1_hist, ),
                  yscal='linear',
                  ylim0=ylim0,
                  ylim1=ylim1)

    if ("cost" in hist_df) and ("vtrue_on_x0cost" in hist_df):
        it_cost = hist_df[['it', 'cost', 'vtrue_on_x0cost']].dropna()
        if len(it_cost) > 0:
            plot_hist(sav_prefix + 'cost_hist.pdf',
                      (it_cost['it'], it_cost['it']),
                      (it_cost['cost'], it_cost['vtrue_on_x0cost']),
                      lable_list=(r'$J(u_{\theta})$', '$J(u^*)$'),
                      yscal='linear')

    if ("rel_l1err" in hist_df) and ("rel_linferr" in hist_df):
        plot_l1linf_error(hist_df, sav_prefix)
    elif "rel_l1err" in hist_df:
        plot_l1_error(hist_df, sav_prefix)

    if "ev_error" in hist_df:
        plot_hist(sav_prefix + 'ev_error_hist.pdf',
                  (hist_df['it'].to_numpy(), ), (hist_df['ev_error'], ),
                  lable_list=None,
                  alpha=1.0)


def save_pathres(path_res, sav_prefix):
    res_col = []
    res_header = []
    # path_res contains:
    # {'t_path': t_path,
    #  'rel_l1err_path': re1_path,
    #  'rel_linferr_path': reinf_path,
    #  'vtrue_path': vtrue_path,
    #  'vappr_path': vappr_path}
    for k1, v1 in path_res.items():
        res_col.append(v1.squeeze(-1))
        if k1 in ['vtrue_path', 'vappr_path']:
            # v2 shape: [t, path, dim of v]
            idx_path = range(v1.shape[-2])
            res_header.extend([f'{k1}{i}' for i in idx_path])
        else:
            res_header.append(k1)
    res_col = np.concatenate(res_col, axis=-1)
    np.savetxt(sav_prefix + 'res_on_path.csv',
               res_col,
               delimiter=',',
               header=','.join(res_header))

    err_path = []
    err_labels = []
    t_grid = path_res['t_path'].squeeze()
    if "rel_l1err_path" in path_res:
        err_path.append(path_res['rel_l1err_path'].squeeze())
        err_labels.append(r"RE$_1$")
    if "rel_linferr_path" in path_res:
        err_path.append(path_res['rel_linferr_path'].squeeze())
        err_labels.append(r"RE$_\infty$")
    if len(err_path) > 0:
        plot_error_path(t_grid, err_path, err_labels, f'{sav_prefix}error_')

    # if ("vtrue_path" in path_res) and ("vappr_path" in path_res):
    #     plot_vtx_path(
    #         t_grid,
    #         path_res['vtrue_path'],
    #         path_res['vappr_path'],
    #         sav_prefix,
    #     )
    if "vtrue_path" in path_res:
        plot_vtx_path(t_grid, path_res['vtrue_path'], None, sav_prefix)


def summary_hist(hist_list) -> pd.DataFrame:

    df_list = [pd.DataFrame(hist) for hist in hist_list]
    hist_df = pd.concat(df_list,
                        axis=0,
                        keys=range(len(df_list)),
                        names=['round', 'index'])
    hist_gp = hist_df.groupby('it')
    mean_df = hist_gp.mean()
    std_df = hist_gp.std()
    summ_df = pd.concat([mean_df, std_df], axis=1, keys=['mean', 'std'])
    summ_df = summ_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return summ_df


def summary_repath(res_list) -> pd.DataFrame:
    # res_list: list of of res_dict, indexed by round
    # res_dict contains:
    # {'t_path': t_path,
    #  'rel_l1err_path': re1_path,
    #  'rel_linferr_path': reinf_path,
    #  'vtrue_path': vtrue_path,
    #  'vappr_path': vappr_path}

    if all(['rel_linferr_path' in res for res in res_list]):
        cols1 = ['t_path', 'rel_l1err_path', 'rel_linferr_path']
    else:
        cols1 = ['t_path', 'rel_l1err_path']

    reslist_filtered = [{
        c: res[c].squeeze()
        for c in cols1
    } for res in res_list]
    df_list = [pd.DataFrame(res_i) for res_i in reslist_filtered]
    hist_df = pd.concat(df_list,
                        axis=0,
                        keys=range(len(df_list)),
                        names=['round', 'index'])
    hist_gp = hist_df.groupby('index')
    mean_df = hist_gp.mean()
    std_df = hist_gp.std()
    summ_df = pd.concat([mean_df, std_df], axis=1, keys=['mean', 'std'])
    summ_df = summ_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return summ_df


def plot_hist_summary(hist_df: pd.DataFrame, sav_name: str):
    it_arr = np.array(hist_df.index)

    if 'pde_loss' in hist_df.columns:
        plot_hist(sav_name + 'summary_pdeloss.pdf', (it_arr, ),
                  (hist_df['pde_loss', 'mean'], ),
                  std_arr_list=(hist_df['pde_loss', 'std'], ),
                  lable_list=None,
                  yscal='log',
                  alpha=1.0,
                  two_side_std=False)

    if 'ctr_loss' in hist_df.columns:
        ctr_loss = hist_df['ctr_loss', 'mean'].to_numpy()
        ylim0, ylim1 = ylim_for_g1loss(ctr_loss)
        plot_hist(sav_name + 'summary_ctrloss.pdf', (it_arr, ), (ctr_loss, ),
                  std_arr_list=(hist_df['ctr_loss', 'std'], ),
                  lable_list=None,
                  yscal='linear',
                  alpha=1.0,
                  two_side_std=False,
                  ylim0=ylim0,
                  ylim1=ylim1)

    record_l1err = ('rel_l1err', 'mean') in hist_df.columns \
        and ('rel_l1err', 'std') in hist_df.columns
    # record_l2err = ('rel_l2err', 'mean') in hist_df.columns \
    #     and ('rel_l2err', 'std') in hist_df.columns
    record_linferr = ('rel_linferr', 'mean') in hist_df.columns \
        and ('rel_linferr', 'std') in hist_df.columns

    if record_l1err:
        mean_l1err = hist_df['rel_l1err', 'mean'].to_numpy()
        std_l1err = hist_df['rel_l1err', 'std'].to_numpy()
    # if record_l2err:
    #     mean_l2err = hist_df['rel_l2err', 'mean'].to_numpy()
    #     std_l2err = hist_df['rel_l2err', 'std'].to_numpy()
    if record_linferr:
        mean_linferr = hist_df['rel_linferr', 'mean'].to_numpy()
        std_linferr = hist_df['rel_linferr', 'std'].to_numpy()

    if record_l1err and record_linferr:
        plot_hist(sav_name + 'summary_error.pdf', (it_arr, it_arr),
                  (mean_linferr, mean_l1err),
                  std_arr_list=(std_linferr, std_l1err),
                  lable_list=(r"RE$_{\infty}$", r"RE$_1$"),
                  yscal='log',
                  alpha=1.0,
                  two_side_std=False)
    elif record_l1err:
        plot_hist(sav_name + 'summary_error.pdf', (it_arr, ), (mean_l1err, ),
                  std_arr_list=(std_l1err, ),
                  lable_list=None,
                  yscal='log',
                  alpha=1.0,
                  two_side_std=False)
    elif record_linferr:
        plot_hist(sav_name + 'summary_error.pdf', (it_arr, ), (mean_linferr, ),
                  std_arr_list=(std_linferr, ),
                  lable_list=None,
                  yscal='log',
                  alpha=1.0,
                  two_side_std=False)

    if (('rel_everr', 'mean') in hist_df.columns) \
            and (('rel_everr', 'mean') in hist_df.columns):
        mean_everr = hist_df['rel_everr', 'mean'].to_numpy()
        std_everr = hist_df['rel_everr', 'std'].to_numpy()
        plot_hist(sav_name + 'summary_ev_error.pdf', (it_arr, ),
                  (mean_everr, ),
                  std_arr_list=(std_everr, ),
                  lable_list=None,
                  yscal='log',
                  alpha=1.0,
                  two_side_std=False)

    if (('cost', 'mean') in hist_df.columns) \
            and (('vtrue_on_x0cost', 'mean') in hist_df.columns):
        cost_df = hist_df[[('cost', 'mean'), ('cost', 'std'),
                           ('vtrue_on_x0cost', 'mean'),
                           ('vtrue_on_x0cost', 'std')]].dropna()
        if len(cost_df) > 0:
            it_cost = cost_df.index.to_numpy()
            mean_cost = cost_df['cost', 'mean'].to_numpy()
            std_cost = cost_df['cost', 'std'].to_numpy()
            mean_vtrue_t0 = cost_df['vtrue_on_x0cost', 'mean'].to_numpy()
            std_vtrue_t0 = cost_df['vtrue_on_x0cost', 'std'].to_numpy()
            plot_hist(
                sav_name + 'summary_cost.pdf',
                (it_cost, it_cost),
                (mean_cost, mean_vtrue_t0),
                std_arr_list=(std_cost, std_vtrue_t0),
                lable_list=(r'$J(u_{\theta})$', '$J(u^*)$'),
                yscal='linear',
            )


def plot_path_summary(path_df: pd.DataFrame, sav_name: str):

    t_path = path_df['t_path', 'mean'].to_numpy()
    mean_l1err = path_df['rel_l1err_path', 'mean'].to_numpy()
    std_l1err = path_df['rel_l1err_path', 'std'].to_numpy()
    x_arr_list = [t_path]
    y_arr_list = [mean_l1err]
    std_arr_list = [std_l1err]
    lable_list = None

    record_linferr = ('rel_linferr_path', 'mean') in path_df.columns \
        and ('rel_linferr_path', 'std') in path_df.columns
    if record_linferr:
        x_arr_list.insert(0, t_path)
        mean_linferr = path_df['rel_linferr_path', 'mean'].to_numpy()
        std_linferr = path_df['rel_linferr_path', 'std'].to_numpy()
        y_arr_list.insert(0, mean_linferr)
        std_arr_list.insert(0, std_linferr)
        lable_list = (r"RE$_{\infty}$", r"RE$_1$")

    ymean_max = np.max(y_arr_list)
    ystd_max = np.max(std_arr_list)
    if np.isnan(ystd_max):
        ystd_max = 0.
    ylim1 = min(1.5 * (ymean_max + 2 * ystd_max), 1.)
    plot_hist(
        sav_name + 'summary_repath.pdf',
        x_arr_list,
        y_arr_list,
        std_arr_list=std_arr_list,
        lable_list=lable_list,
        yscal='linear',
        alpha=1.0,
        two_side_std=False,
        ylim1=ylim1,
        ylim0=0.,
    )
