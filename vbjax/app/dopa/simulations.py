import jax
import jax.numpy as np
import vbjax as vb

# Simulating for the tavg and bold monitors
def make_run(init, tavg_period, total_time=100.0, dt=1e-1):
    '''returns the run function;
    so it basically serves as a factory function that creates
    and configures the run function based on the provided parameters.
    '''
    nt = int(tavg_period / dt) # chunk size
    nta = int(np.ceil(total_time / tavg_period)) # number of chunks

    def gfun(x, p):
        *_, dopa_theta = p
        # change noise model here
        return dopa_theta.sigma

    _, loop = vb.make_sde(dt, vb.dopa_net_dfun, gfun)

    # setup tavg monitor
    n_nodes = init.shape[1]
    ta_buf, ta_step, ta_sample = vb.make_timeavg((init.shape[0], n_nodes))
    ta_sample = vb.make_offline(ta_step, ta_sample)

    # now setup bold
    bold_buf, bold_step, bold_samp = vb.make_bold(
        shape=init[0].shape,  # only r
        dt=tavg_period/1e3,
        p=vb.bold_default_theta)

    # run function actually does the simulation based on inputs
    # that we might want to sweep over or change
    @jax.jit
    def run(Ci, Ce, Cd, params, key=jax.random.PRNGKey(42)):
 
        sim = {
            'ta': ta_buf,
            'bold': bold_buf,
            'init': init,
            'p': (Ci, Ce, Cd, params),
            'key': key,
        }

        def sim_step(sim, t_key):
            t, key = t_key
            
            # sim['key'], key = jax.random.split(sim['key'])

            # generate randn and run simulation from initial conditions
            dw = jax.random.normal(key, (nt, 6, n_nodes))
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

# Simulating for raw timeseries
def make_raw(init, t_period, total_time=100.0, dt=1e-1):
    nt = int(t_period / dt)  # chunk size
    nta = int(np.ceil(total_time / t_period))  # number of chunks

    def gfun(x, p):
        *_, dopa_theta = p
        # change noise model here
        return dopa_theta.sigma  # for now just additive noise
    _, loop = vb.make_sde(dt, vb.dopa_net_dfun, gfun)
    n_state_variables, n_nodes = init.shape[0], init.shape[1]

    @jax.jit
    def run(Ci, Ce, Cd, params, key=jax.random.PRNGKey(42)):
        '''input: Ci, Ce, Cd, params
        output: ts, y
        '''
        sim = {
            'init': init,
            'p': (Ci, Ce, Cd, params),
            'key': key,
        }

        def sim_step(sim, t_key):
            t, key = t_key
            # generate randn and run simulation from initial conditions
            dw = jax.random.normal(key, (nt, 6, n_nodes))
            raw = loop(sim['init'], dw, sim['p'])
            sim['init'] = raw[-1]
            return sim, raw

        ts = np.r_[:nta]*t_period  # get ts
        keys = jax.random.split(key, ts.size)  # generate as many keys as ts

        sim, raw = jax.lax.scan(sim_step, sim, (ts, keys))
        return ts, raw.reshape(-1, n_state_variables, n_nodes)

    return run