import jax
import jax.numpy as np
import vbjax as vb
# internal modules
import nm

def simulate(init, cy, p, dopa_dfun, adhoc=None, sigma=0.0, T=1, dt = 1e-3, seed=42):
    key = jax.random.PRNGKey(seed)
    nt = int(T / dt)
    dw = jax.random.normal(key, (nt, init.size))
    f = lambda x, p: dopa_dfun(x, cy, p)
    _, loop = vb.make_sde(dt, f, sigma, adhoc=adhoc)
    ys = loop(init, dw, p)
    return ys


def sweep_node(init, cy, params, dopa_dfun, adhoc=None, T=10.0, dt=0.01, sigma=1e-3, seed=42, cores=4):
    "Run sweep for single dopa node on params matrix"

    # setup grid for parameters
    pkeys, pgrid = vb.tuple_meshgrid(params)
    pshape, pravel = vb.tuple_ravel(pgrid)

    # distribute params for cpu; doesn't work for now
    if vb.is_cpu:
        pravel = vb.tuple_shard(pravel, cores)

    # setup model
    f = lambda x, p: dopa_dfun(x, cy, p)
    _, loop = vb.make_sde(dt, f, sigma, adhoc=adhoc)

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


# the `make_run` function just needs dimensions of the problem 
# check the version on jlab to get the old version.
def make_run(init, adhoc=nm.dopa_r_positive,
                       dt=1e-2, tavg_period=5.0, total_time=100.0):
    nt = int(tavg_period / dt)
    nta = int(np.ceil(total_time / tavg_period))
    n_nodes = init.shape[1]
    def gfun(x, p):
        *_, dopa_theta = p
        # change noise model here
        return dopa_theta.sigma
    _, loop = vb.make_sde(dt, nm.dopa_net_dfun, gfun, adhoc)
    # tavg monitor
    ta_buf, ta_step, ta_sample = vb.make_timeavg((init.shape[0], n_nodes))
    ta_sample = vb.make_offline(ta_step, ta_sample)
    # bold monitor
    bold_buf, bold_step, bold_samp = vb.make_bold(
            shape=init[0].shape, # only r
            dt=tavg_period/1e3,
            p=vb.bold_default_theta)
    
    def run(Ci, Ce, Cd, params, key):
    
        sim = {
            'ta': ta_buf,
            'bold': bold_buf,
            'init': init,
            'p': (Ci, Ce, Cd, params),
        }
        
        def sim_step(sim, dw):
            ys = loop(sim['init'], dw, sim['p'])
            sim['ta'], ta_y = ta_sample(sim['ta'], ys)
            sim['bold'] = bold_step(sim['bold'], ta_y[0])
            _, bold_t = bold_samp(sim['bold'])
            sim['init'] = ys[-1]
            return sim, (ta_y, bold_t)
    
        ts = np.r_[:nta]*tavg_period
        dw = jax.random.normal(key, (ts.size * nt, init.shape[0], n_nodes)).reshape(ts.size,nt,init.shape[0],n_nodes)
        sim, (ta_y, bold) = jax.lax.scan(sim_step, sim, dw)
        return ts, ta_y, bold

    return jax.jit(run)


#  
def make_run_sn(init, cy, dopa_dfun, tavg_period, adhoc=None, total_time=100.0, dt=1e-1):
    ''' `make_run` single node function:
    returns the run function;
    so it basically serves as a factory function that creates
    and configures the run function based on the provided parameters.
    '''
    nt = int(tavg_period / dt) # chunk size
    nta = int(np.ceil(total_time / tavg_period)) # number of chunks

    def gfun(x, p):
        dopa_theta = p
        # change noise model here
        return dopa_theta.sigma

    dfun = lambda x, p: dopa_dfun(x, cy, p)
    _, loop = vb.make_sde(dt, dfun, gfun, adhoc)

    # setup tavg monitor
    ta_buf, ta_step, ta_sample = vb.make_timeavg(init.shape)
    ta_sample = vb.make_offline(ta_step, ta_sample)

    # now setup bold
    bold_buf, bold_step, bold_samp = vb.make_bold(
        shape=(1,),  # only r
        dt=tavg_period/1e3,
        p=vb.bold_default_theta)

    # run function actually does the simulation based on inputs
    # that we might want to sweep over or change
    @jax.jit
    def run(params, key=jax.random.PRNGKey(42)):
 
        sim = {
            'ta': ta_buf,
            'bold': bold_buf,
            'init': init,
            'p': params,
            'key': key,
        }

        def sim_step(sim, t_key):
            t, key = t_key
            
            # sim['key'], key = jax.random.split(sim['key'])

            # generate randn and run simulation from initial conditions
            dw = jax.random.normal(key, (nt, init.size))
            raw = loop(sim['init'], dw, sim['p'])

            # monitor results
            sim['ta'], ta_y = ta_sample(sim['ta'], raw)
            sim['bold'] = bold_step(sim['bold'], ta_y[0])
            _, bold_t = bold_samp(sim['bold'])
            sim['init'] = raw[-1]
            return sim, (ta_y, bold_t)

        ts = np.r_[:nta]*tavg_period
        keys = jax.random.split(key, ts.size)
        
        sim, (ta_y, bold) = jax.lax.scan(sim_step, sim, (ts, keys))
        return ts, ta_y, bold
        
    return run

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