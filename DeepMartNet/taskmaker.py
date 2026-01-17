# Produce task files, which will be used by runtask.py

from itertools import product

from runtask import DEFAULT_CONFIG, TASK_PATH, get_config, sav_config


def maxit_default(dim_x):
    if dim_x <= 1000:
        maxit = 2000
    elif dim_x <= 5000:
        maxit = 4000
    else:
        maxit = 6000
    return maxit


def maxit_long(dim_x):
    if dim_x <= 1000:
        maxit = 3000
    elif dim_x <= 5000:
        maxit = 6000
    else:
        maxit = 9000
    return maxit


def batsize_default(dim_x, _rs):
    if dim_x <= 1000:
        bsz = 256
    elif dim_x <= 5000:
        bsz = 128
    elif dim_x <= 10000:
        bsz = 64
    else:
        bsz = 32
    return bsz


def eposize_default(dim_x):
    if dim_x <= 5000:
        esz = 10000
    elif dim_x <= 10000:
        esz = 5000
    else:
        esz = 2000
    return esz


def default_lr0func(dim_x):
    if dim_x >= 1000:
        expo = 0.8
    else:
        expo = 0.5
    lr0_u = 3 * 1e-3 / dim_x**expo
    lr0_v = 3 * 1e-3 / dim_x**expo
    lr0_rho = 3 * 1e-2 / dim_x**expo
    return lr0_u, lr0_v, lr0_rho


def small_lrfunc(dim_x):
    lr0_u, lr0_v, lr0_rho = default_lr0func(dim_x)
    return lr0_u * 0.1, lr0_v * 0.1, lr0_rho * 0.1


def fix_lrfunc(_dim_x):
    return 1e-3, 1e-3, 1e-3


def ev_lrfunc(dim_x):
    if dim_x >= 1000:
        expo = 0.8
    else:
        expo = 0.5
    lr0_u = 3 * 1e-4 / dim_x**expo
    lr0_v = 3 * 1e-4 / dim_x**expo
    lr0_rho = 3 * 1e-3 / dim_x**expo
    return lr0_u, lr0_v, lr0_rho


def default_width(dim_x):
    if dim_x <= 100:
        wid = 5 * dim_x
    elif dim_x <= 1000:
        wid = 2 * dim_x
    else:
        wid = dim_x
    wid = max(wid, 100)
    return wid


def set_method(config, meth):

    if meth == 'MartNet':
        method_pde = 'QuasiMartNet'
        method_soc = 'SocMartNet'
    elif meth == 'DfMartNet':
        method_pde = 'DfQuasiMartNet'
        method_soc = 'DfSocMartNet'
    elif meth == 'RdmStrForm':
        method_pde = 'QuasiRdmStrForm'
        method_soc = 'SocRdmStrForm'
    elif meth == 'Pinn':
        method_pde = 'QuasiPinn'
        method_soc = 'SocPinn'
    else:
        raise ValueError(f'Unknown method {meth}.')

    config.set('Example', 'method_pde', method_pde)
    config.set('Example', 'method_soc', method_soc)
    return config


def make_task(
    dimx_list,
    example_list,
    numdt_list,
    meth_list,
    num_hidd=4,
    start_idx=0,
    width_rho=600,
    lr0_func=default_lr0func,
    rate_lr0_lrend=1e-2,
    maxit_func=None,
    width_func=default_width,
    repeat_time=1,
    syspath_as_pilpath=True,
    batsize_func=batsize_default,
    eposize_func=eposize_default,
    enable_autocast=False,
    use_multiscale=False,
    scale_ub_rho=10.,
    use_varscale=False,
    optimizer='Adam',
    num_rdmsamp_list=None,
):
    idx = start_idx
    tot_idx = start_idx - 1 + len(dimx_list) * len(example_list)
    length = len(str(tot_idx))

    if maxit_func is None:
        maxit_func = maxit_default

    for meth, dimx, exmaple, num_dt in product(
            meth_list,
            dimx_list,
            example_list,
            numdt_list,
    ):
        default_desc_step = get_config(DEFAULT_CONFIG).getint(
            'Training', 'inner_step_descend')
        if meth != 'RdmStrForm':

            actual_nrdms_list = [None]
            actual_desc_step = default_desc_step
            max_it = maxit_func(dimx)
        else:
            actual_nrdms_list = num_rdmsamp_list
            actual_desc_step = 1
            max_it = maxit_func(dimx) * default_desc_step

        for rs in actual_nrdms_list:
            bsz = batsize_func(dimx, rs)
            esz = eposize_func(dimx)
            wid = width_func(dimx)
            config = get_config(DEFAULT_CONFIG)
            config.set('Environment', 'enable_autocast', str(enable_autocast))
            config.set('Example', 'dim_x', str(dimx))

            config = set_method(config, meth)
            config.set('Network', 'width_u', str(wid))
            config.set('Network', 'width_v', str(wid))
            config.set('Network', 'width_rho', str(width_rho))
            config.set('Network', 'num_hidden_u', str(num_hidd))
            config.set('Network', 'num_hidden_v', str(num_hidd))
            config.set('Network', 'multiscale_u', str(use_multiscale))
            config.set('Network', 'multiscale_v', str(use_multiscale))
            config.set('Network', 'varscale', str(use_varscale))
            config.set('Network', 'scale_ub_rho', str(scale_ub_rho))

            config.set('Example', 'name', exmaple)
            config.set('Example', 'repeat_time', str(repeat_time))

            config.set('Optimizer', 'optimizer_u', optimizer)
            config.set('Optimizer', 'optimizer_v', optimizer)
            config.set('Optimizer', 'optimizer_lamb', optimizer)
            config.set('Optimizer', 'optimizer_rho', optimizer)

            lr0_u, lr0_v, lr0_rho = lr0_func(dimx)
            config.set('Optimizer', 'lr0_u', str(lr0_u))
            config.set('Optimizer', 'lr0_v', str(lr0_v))
            config.set('Optimizer', 'lr0_rho', str(lr0_rho))

            config.set(
                'Optimizer',
                'decay_rate_u',
                str(rate_lr0_lrend) +
                '**(${decay_stepgap_u} / ${Training:max_iter})',
            )
            config.set(
                'Optimizer',
                'decay_rate_v',
                str(rate_lr0_lrend) +
                '**(${decay_stepgap_v} / ${Training:max_iter})',
            )
            config.set(
                'Optimizer',
                'decay_rate_rho',
                str(rate_lr0_lrend) +
                '**(${decay_stepgap_rho} / ${Training:max_iter})',
            )

            config.set('Training', 'num_dt', str(num_dt))
            config.set('Training', 'max_iter', str(max_it))
            config.set('Training', 'batsize', str(bsz))
            config.set('Training', 'epochsize', str(esz))
            config.set('Training', 'syspath_as_pilpath',
                       str(syspath_as_pilpath))
            config.set('Training', 'inner_step_descend', str(actual_desc_step))
            if rs is not None:
                config.set('RdmStrForm', 'num_rdmsamp', str(rs))

            idx_str = str(idx).zfill(length)

            if use_multiscale and use_varscale:
                ScaleNw = '_MVscale'
            elif use_multiscale:
                ScaleNw = '_Mscale'
            elif use_varscale:
                ScaleNw = '_Vscale'
            else:
                ScaleNw = ''

            sav_name = f'{idx_str}_{meth}_{exmaple}_d{dimx}_W{wid}_H{num_hidd}_N{num_dt}{ScaleNw}_BZ{bsz}'
            if meth == 'RdmStrForm':
                sav_name = f'{sav_name}_RS{rs}'

            sav_config(config, f'{TASK_PATH}/{sav_name}')
            idx = idx + 1


def main():

    # maxit_func = maxit_long
    # maxit_func = lambda d: 5000 if d <= 10000 else 12000
    maxit_func = lambda _d: 3000
    # maxit_func = maxit_default
    # width_func = lambda d: int(d // 100) + 100
    width_func = lambda _d: 1024
    # width_func = default_width
    width_rho = 1024
    lr0_func = default_lr0func
    # lr0_func = small_lrfunc
    # lr0_func = lambda _d: (1e-3, 1e-3, 1e-2)

    rate_lr0_lrend = 1e-2
    num_rdmsamp_list = [128]

    # batsize_func = batsize_default
    # batsize_func = lambda _d, rs: int(512 // rs)
    batsize_func = lambda _d, _rs: 64

    num_hidd = 4
    # lr0_func = fix_lrfunc
    eposize_func = eposize_default

    syspath_as_pilpath = True
    enable_autocast = False
    use_multiscale = False
    use_varscale = False

    scale_ub_rho = 10.

    meth_list = ['DfMartNet']
    # 'Pinn', 'MartNet', 'DfMartNet', 'RdmStrForm'

    # optimizer = 'Adam'
    optimizer = 'RMSprop'
    repeat_time = 1
    dimx_list = [100, 1000]
    # dimx_list = [int(k * 10**5) for k in [2, 5, 10, 20, 50]]
    numdt_list = [100]
    # numdt_list = [int(8 * (128 / 8)**(i / 10)) for i in range(1, 11)]

    exa_list = [
        # 'Shock1a',
        # 'Shock1b',
        # 'Shock1aBmPil',
        # 'Shock1bBmPil',
        # 'Shock1aNoOsc',
        # 'Shock1bNoOsc',
        'BlackScholes',
        'QLP_HJB0_Rosenbrock'
        # 'QLP_HJB0a',
        # 'QLP_HJB0b',
        # 'HJB2v',
        # 'Shock1aBmPil',
        # 'Shock1bBmPil',
        # 'HJB1a',
        # 'HjbHopfColeSumG',
        # 'HjbHopfCole1aBmPil',
        # 'HJB1b',
        # 'HJB2',
        # 'QLP1',
        # 'QLP2a',
        # 'QLP2b',
        # 'AllenCahnBVP',
        # 'SineGordonBVP',
        # 'AllenCahnBVPLessTestPt',
        # 'SineGordonBVPLessTestPt',
    ]

    make_task(
        dimx_list,
        exa_list,
        numdt_list,
        meth_list,
        num_hidd=num_hidd,
        width_rho=width_rho,
        maxit_func=maxit_func,
        width_func=width_func,
        repeat_time=repeat_time,
        lr0_func=lr0_func,
        rate_lr0_lrend=rate_lr0_lrend,
        syspath_as_pilpath=syspath_as_pilpath,
        batsize_func=batsize_func,
        eposize_func=eposize_func,
        enable_autocast=enable_autocast,
        use_multiscale=use_multiscale,
        scale_ub_rho=scale_ub_rho,
        use_varscale=use_varscale,
        optimizer=optimizer,
        num_rdmsamp_list=num_rdmsamp_list,
    )


if __name__ == "__main__":
    main()
