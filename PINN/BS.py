import haiku as hk
import jax
import optax
import jax.numpy as jnp
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

class MLP(hk.Module):
    def __init__(self, layers, T, r):
        super().__init__()
        self.layers = layers
        self.T = T
        self.r = r

    def phi(self, x):
        return jnp.exp(-self.r * self.T) * jax.nn.relu(jnp.max(x) - 100)

    def __call__(self, x, t):
        X = jnp.hstack([t, x])
        for dim in self.layers[:-1]:
            X = hk.Linear(dim)(X)
            X = jnp.tanh(X)
        X = hk.Linear(self.layers[-1])(X)
        X = X[0]
        return X

parser = argparse.ArgumentParser(description='GBM')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--epochs', type=int, default=300001)
parser.add_argument('--dim', type=int, default=11)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--PINN_h', type=int, default=128)
parser.add_argument('--PINN_L', type=int, default=4)
parser.add_argument('--N_r', type=int, default=100, help="Number of residual points")
parser.add_argument('--T', type=float, default=1)
args = parser.parse_args()
print(args)

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

x, t, u = np.loadtxt("GBM_Data/x_"+str(args.dim)+".txt"), np.loadtxt("GBM_Data/t_"+str(args.dim)+".txt"), np.loadtxt("GBM_Data/u_"+str(args.dim)+".txt")
r = 1.0 / 20.0
mu = -1.0 / 20.0
T = args.T
sigma = 0.4 * jnp.arange(1, args.dim) / (args.dim - 1) + 0.1

class PINN:
    def __init__(self):
        self.epoch = args.epochs
        self.adam_lr = args.lr
        self.X, self.Y, self.U = x, t, u
        self.r, self.T = r, T

        layers = [args.PINN_h] * (args.PINN_L - 1) + [1]
        @hk.transform
        def network(x, t):
            temp = MLP(layers=layers, T=T, r=r)
            return temp(x, t)
 
        self.u_net = hk.without_apply_rng(network)
        self.u_pred_fn = jax.vmap(self.u_net.apply, (None, 0, 0)) # consistent with the dataset
        self.r_pred_fn = jax.vmap(self.GBM, (None, 0, 0))
        self.b_pred_fn = jax.vmap(self.boundary, (None, 0, 0))

        self.params = self.u_net.init(key, self.X[0], self.Y[0])
        lr = optax.exponential_decay(init_value=self.adam_lr, transition_steps=5000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)
    
    def GBM(self, params, x, y): 
        u_t = jax.grad(self.u_net.apply, argnums=2)(params, x, y)
        u_x = jax.jacrev(self.u_net.apply, argnums=1)(params, x, y)
        u_hess = jax.jacfwd(jax.jacrev(self.u_net.apply, argnums=1), argnums=1)(params, x, y)
        u_xx = jnp.diag(u_hess)
        return jnp.sum(0.5 * sigma**2 * x**2 * u_xx + mu * x * u_x) - u_t
    
    def boundary(self, params, x, y):
        pred_b = self.u_net.apply(params, x, y)
        ub = jnp.exp(-self.r * self.T) * jax.nn.relu(jnp.max(x) - 100)
        return pred_b - ub
    
    def get_loss_pinn(self, params, xf, yf, xb, yb):
        f = self.r_pred_fn(params, xf, yf)
        b = self.b_pred_fn(params, xb, yb)
        mse_f = jnp.mean(f**2) + 20 * jnp.mean(b ** 2)
        return mse_f
    
    @partial(jax.jit, static_argnums=(0,))
    def step_pinn(self, params, opt_state, xf, yf, xb, yb):
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn)(params, xf, yf, xb, yb)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state
    
    def resample(self):
        N_r = args.N_r
        D = args.dim - 1

        t = np.random.rand(N_r, 1) * args.T
        W = np.random.randn(N_r, D) * np.sqrt(t)
        x = np.random.rand(N_r, D) * 20 + 90
        x = x * np.exp(sigma.reshape(1, D) * W + (mu - sigma.reshape(1, D)**2 / 2) * t)
        t = args.T - t.reshape(-1)

        tb = np.ones((N_r, 1)) * args.T
        Wb = np.random.randn(N_r, D) * np.sqrt(tb)
        xb = np.random.rand(N_r, D) * 20 + 90
        xb = xb * np.exp(sigma.reshape(1, D) * Wb + (mu - sigma.reshape(1, D)**2 / 2) * tb)
        tb = args.T - tb.reshape(-1)

        return x, t, xb, tb

    def train_adam(self):
        for n in tqdm(range(self.epoch)):
            xf, yf, xb, yb = self.resample()
            current_loss, self.params, self.opt_state = self.step_pinn(self.params, self.opt_state, xf, yf, xb, yb)
            if n%1000==0: print('epoch %d, loss: %e, L2: %e'%(n, current_loss, self.L2_pinn(self.params, self.X, self.Y, self.U)))

    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn(self, params, x, y, u):
        pinn_u_pred_20 = self.u_pred_fn(params, x, y).reshape(-1)
        pinn_error_u_total_20 = jnp.linalg.norm(u - pinn_u_pred_20, 2) / jnp.linalg.norm(u, 2)
        return (pinn_error_u_total_20)

model = PINN()
model.train_adam()