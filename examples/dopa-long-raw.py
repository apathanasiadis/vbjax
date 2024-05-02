import time
import sys
import os
import numpy
import jax
from jax.lib import xla_bridge
from jax import config
import os
jax.config.update('jax_platform_name','gpu')
# os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={vb.cores}' # for CPU usage
print(xla_bridge.get_backend().platform)
import jax.numpy as np
import vbjax as vb
from vbjax.app.dopa import simulations
from pathlib import Path

cwd = os.getcwd()
head = Path(cwd).parents[2]
print('head path:', head)
sys.path.append(os.path.join(head, "mymodules"))
import functions
#%%
# load the connectome -- Spase
path_connectome = os.path.join(head, "connectome/Material_connectome_implementation")
logconn, connectome_matrix, factor, Ce, Ci, Cd, SC, rois = functions.get_connectome(os.path.join(head, 'connectome'), log=True)
n_nodes = logconn.shape[0]
# Parameters Definition
y0 = np.array([0.1, -70, 0, 0, 0, 0.05], dtype = np.float32)
y0 = np.outer(y0, np.ones(n_nodes))
## old settings for bistability ##
standard_params = {
    "we": 7.2e-2,  #5e-2
    "wi": 8e-3,
    "wd": 1e-1,
    "sigma": 4.75e-5
}
# input parameters
params = vb.dopa_default_theta
params = params._replace(
    wi=standard_params["wi"]*factor,
    we=standard_params["we"]*factor,
    wd=standard_params["wd"]*factor,
    sigma=standard_params["sigma"]
)
# Simulation time
total_time = 1e3
t_period = 1e2
dt = 0.01

# functions
run = simulations.make_raw(y0, t_period=t_period,  # 5ms or 200Hz
                             total_time=total_time, dt=dt)

# RUN SIMULATION!
tik = time.time()
ts, y = run(Ci, Ce, Cd, params)
y.block_until_ready()
tok = time.time()
print(tok - tik, "seconds for the simulation")

r = y[:,0,0]
V = y[:,1,0]
del y

# save into the output_dict
output_dict = {
    "params" : params,
    "ts"     : ts,
    "r"      : r,
    "V"      : V,
}