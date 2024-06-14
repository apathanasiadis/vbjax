import os
import sys
import jax
import jax.numpy as np
import vbjax as vb
# internal functions
from pathlib import Path
cwd = os.getcwd()
i = 0; head_tail = '';
while head_tail != 'neuromodulation':
    head = Path(cwd).parents[i]
    head_tail = os.path.split(head)[1]
    i += 1
sys.path.append(os.path.join(head, 'vbjax/vbjax/'))
##############################################

def simulate(init, cy, p, dopa_dfun, sigma=0.0, T=1, dt = 1e-3, seed=42):
    key = jax.random.PRNGKey(seed)
    nt = int(T / dt)
    dw = jax.random.normal(key, (nt, init.size))
    f = lambda x, p: dopa_dfun(x, cy, p)
    _, loop = vb.make_sde(dt, f, sigma)
    ys = loop(init, dw, p)
    return ys


def sweep_node(init, cy, params, dopa_dfun, T=10.0, dt=0.01, sigma=1e-3, seed=42, cores=4):
    "Run sweep for single dopa node on params matrix"

    # setup grid for parameters
    pkeys, pgrid = vb.tuple_meshgrid(params)
    pshape, pravel = vb.tuple_ravel(pgrid)

    # distribute params for cpu; doesn't work for now
    if vb.is_cpu:
        pravel = vb.tuple_shard(pravel, cores)

    # setup model
    f = lambda x, p: dopa_dfun(x, cy, p)
    _, loop = vb.make_sde(dt, f, sigma)

    # assume same inits and noise for all params
    key = jax.random.PRNGKey(seed)
    nt = int(T / dt)
    dw = jax.random.normal(key, (nt, init.size))
    
    # run sweep
    runv = jax.vmap(lambda p: loop(init, dw, p))
    run_params = jax.jit(jax.vmap(runv) if vb.is_cpu else runv)
    ys = run_params(pravel)

    # reshape the resulting time series
    # assert ys.shape == (pravel[0].size, nt, 6)
    ys = ys.reshape(pshape + (nt, init.size))

    return pkeys, ys


##################################################################################

def dopa_dfun_2args(y, p):
    "Adaptive QIF model with dopamine modulation --modified." 

    r, V, u, Sa, Sg, Dp = y
    c_inh, c_exc, c_dopa, node_params = p
    a, b, c, ga, gg, Eta, Delta, Iext, Ea, Eg, Sja, Sjg, tauSa, tauSg, alpha, beta, ud, k, Vmax, Km, Bd, Ad, tau_Dp, *_ = node_params

    dr = 2. * a * r * V + b * r - ga * Sa * r - gg * Sg * r + (a * Delta) / np.pi
    dV = a * V**2 + b * V + c + Eta - (np.pi**2 * r**2) / a + (Ad * Dp + Bd) * ga * Sa * (Ea - V) + gg * Sg * (Eg - V) + Iext - u
    du = alpha * (beta * V - u) + ud * r
    dSa = -Sa / tauSa + Sja * (c_exc + r)
    dSg = -Sg / tauSg + Sjg * c_inh
    dDp = (k * c_dopa - Vmax * Dp / (Km + Dp)) / tau_Dp
    return np.array([dr, dV, du, dSa, dSg, dDp])

def sweep_node_cy(init, cy, params, T=10, dt=1e-2, sigma=0, seed=42):
    '''sweep the dopa_fun node, where
    cy: (c_inh, c_exc, c_dopa) and
    e.g. c_inh is an array that you want to sweep over
    '''
    cym = np.meshgrid(*cy)
    cyf = [ci.flatten() for ci in cym]
    c_inh, c_exh, c_dopa = cyf
    # setup model
    _, loop = vb.make_sde(dt, dopa_dfun_2args, sigma)
    # assume same inits and noise for all params
    key = jax.random.PRNGKey(seed)
    nt = int(T / dt)
    dw = jax.random.normal(key, (nt, init.size))
    # run sweep
    runv = lambda c_inh, c_exh, c_dopa: loop(init, dw,
                                            (c_inh, c_exh, c_dopa, params))
    run_params = jax.jit(jax.vmap(runv))
    ys = run_params(c_inh, c_exh, c_dopa)
    ys = ys.reshape(cy[0].size, cy[1].size, cy[2].size, -1, init.size)
    return ys