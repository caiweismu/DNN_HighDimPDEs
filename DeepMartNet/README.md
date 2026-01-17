# DeepMartNet

This folder contains the DeepMartNet implementation used for the numerical results (HJB and Black-Scholes equations with d = 100 and 1000) reported in the paper "Deep Neural networks for solving high-dimensional parabolic partial differential equations". 
Here, DeepMartNet refers to the derivative-free SOC-MartNet method proposed in arXiv:2408.14395v4, which is equivalent to the deep random difference method (DRDM) proposed in arXiv:2506.20308. 

The parameter settings are provided in the `taskfiles/` folder. 
Running `runtask.py` reproduces the 4 test cases from the paper.
To run a single test case, place only one `.ini` file in the `taskfiles/` folder. 

Below is a summary of the repository structure and instructions for installation and quick start.

## Installation

Python dependencies are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Quick start

### Run with the default config

If `taskfiles/` contains no `.ini` files, `runtask.py` will run using `default_config.ini`.

```bash
python runtask.py
```

Outputs are written to the directory specified by `[Environment] output_dir` (default: `./outputs`).

### Run a batch of tasks

Place one or more `*.ini` config files under `taskfiles/` (or generate them via `taskmaker.py`).
Then run:

```bash
python runtask.py
```

The runner iterates through all `taskfiles/*.ini` files in sorted order.

**Example usage:** Running `taskmaker.py` generates an `.ini` file under `./taskfiles/`. 

If you delete the `./taskfiles/` folder, `runtask.py` falls back to `default_config.ini`.


## Repository structure (module summary)

### Core runner

- `runtask.py`
	- Main entrypoint.
	- Loads configs, instantiates an example problem, builds networks, selects a loss/method, samples training paths, runs training, and saves results.
	- Supports CPU and GPU; uses torch.distributed (DDP) when `world_size > 1`.

- `default_config.ini`
	- Default configuration template.
	- Used when `taskfiles/` has no task configs.

- `taskfiles/`
	- Directory scanned by `runtask.py` for `*.ini` task configs.
	- Empty by default.

- `taskmaker.py`
	- Generates many task `*.ini` files (parameter sweeps) under `taskfiles/`.
	- Encodes common heuristics for max iterations, batch size, network width, learning rates, etc.

### Problem definitions (examples)

- `ex_meta.py`
	- Defines base problem classes and shared utilities.
	- Includes helpers for automatic differentiation (time/space derivatives) and Monte-Carlo evaluation of reference solutions.
	- Provides curve/point generators used for training and evaluation.

- `ex_hjb_hopfcole.py`
	- HJB examples where a Hopf–Cole type transform is used to obtain a reference solution via an auxiliary linear PDE.
	- Contains multiple HJB variants (e.g., `HJB0`, `HJB1a`, etc.).

- `ex_hjb_constantceff.py`
	- Constant-coefficient HJB examples, including cases where the reference solution is computed by Monte Carlo.

- `ex_pde_linear.py`
	- Linear PDE examples (e.g., shock-type drift/diffusion, Black–Scholes-style problems).
	- Some examples provide closed-form `v(t,x)` for direct error evaluation.

- `ex_qlp.py`
	- Quasi-linear PDE examples (`QLP*`) including variants with different diffusion magnitudes.

- `ex_evp.py`
	- Eigenvalue problem examples (`EVP*`).
	- Uses a network that includes an explicit learnable eigenvalue parameter.

- `ex_bvp.py`
	- Boundary-value style problems and utilities for simulating paths with absorbing boundaries (e.g., unit ball).

- `ex_miscellany.py`
	- Additional PDE test problems (e.g., Allen–Cahn, Black–Scholes equation variants).

- `ex_invest.py`
	- Stochastic control / investment HJB example(s), such as CRRA investment.

### Neural networks

- `networks.py`
	- Network components used by the solver.
	- Includes time-dependent and time-independent MLPs (`DNNtx`, `DNNx`), Fourier feature inputs, and wrappers for multiscale and variable-scaling networks.

### Sampling and training

- `sampling.py`
	- `PathSampler` for generating pilot paths and (optionally) offline components used by different losses.
	- Implements path refreshing across epochs and returns mini-batches to the training loop.

- `loss_meta.py`
	- Abstract loss interface (`LossCollection`) and the generic training loop (`train`).
	- Includes logging helpers and GPU memory recording.

### Loss functions / methods

- `loss_martnet.py`
	- MartNet-family objectives implemented as `LossCollection` subclasses.
	- Includes:
		- `DfSocMartNet`: derivative-free MartNet/deep random-difference method for HJB equations.
		- `SocMartNet`: Soc-MartNet for HJB equations.
		- `QuasiMartNet`: Soc-MartNet for quasi-linear parabolic PDEs.
		- `DfQuasiMartNet`: derivative-free MartNet/deep random-difference method for quasi-linear parabolic PDEs.

- `loss_martnet_strf.py`
	- Strong-form random difference method (RDM) variants: `SocRdmStrForm` and `QuasiRdmStrForm`.
	- Uses `num_rdmsamp` (even) and optional antithetic sampling.

- `loss_pinn.py`
	- PINN baselines (`QuasiPinn`, `SocPinn`) using strong-form residual minimization via automatic differentiation.
	- Currently does not support DDP.

### Results, plotting, and post-processing

- `sav_res_via_runtask.py`
	- Utilities to save training logs and summarize repeats (mean/std).
	- Plot helpers for error curves and path-based diagnostics.

- `sav_res_via_exmeta.py`
	- Utilities to evaluate/plot solutions on curves and to visualize 2D landscapes.
	- Used by `ex_meta.py` and example problems when saving results.

### Misc

- `*.sh` (e.g., `srun_*.sh`)
	- Cluster/HPC helper scripts (typically for Linux/SLURM environments).


## Notes

- DDP is used automatically when `[Environment] world_size` is greater than 1 and GPUs are available.
- Some reference-solution evaluations use Monte Carlo and can be expensive (see `nsamp_mc` in example classes).
