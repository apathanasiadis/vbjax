#%%
import jax
import jax.numpy as np
import numpy
from scipy.optimize import fsolve # if substituted by jax function ill be able to vmap
import utilities
import numerics
import vbjax as vb
from jax.tree_util import Partial
#%%
import nm
utilities.add_modules_folder(parent_folder_name='neuromodulation',
                                 modules_folder_name='vbjax/vbjax/')
from neural_mass import dopa_dfun, dopa_dr, dopa_dV, dopa_du, dopa_dSa, dopa_dSg, dopa_dDp
#%%
def equations(y0, cy, params):
        return np.array([dopa_dr(y0, cy, params), dopa_dV(y0, cy, params), dopa_du(y0, cy, params),
                         dopa_dSa(y0, cy, params), dopa_dSg(y0, cy, params), dopa_dDp(y0, cy, params)])

def find_intersections(y0, cy, params, equations_fun=equations, dopa_dfun=dopa_dfun, jaximus=False):
    '''returns eq point and associated jacobian matrix'''
    # Define the intersection function as a system of equations
    if jaximus:
        pequations = Partial(equations_fun, cy=cy, params=params)
        intersection = numerics.newton(pequations, y0, tol=1e-16)
    else:
        intersection = fsolve(equations_fun, y0, args=(cy, params), xtol=1e-16)    
    def jac(ystar, cy, params):
        A = np.vstack(jax.jit(jax.jacfwd(dopa_dfun))(ystar, cy, params))
        return A
    A = jac(intersection, cy, params)
    return intersection, A

def sweep2bif(y0, cy, params, equations_fun=equations, dopa_dfun=dopa_dfun, sweep_var='params'):
    '''uses the `find_intersections function to sweep over params and get jacobian eigenvalues`'''
    if sweep_var == 'params':
        pkeys, pgrid = vb.tuple_meshgrid(params)
        pshape, pravel = vb.tuple_ravel(pgrid)
        runv = jax.jit(jax.vmap(lambda p: find_intersections(y0, cy, p,
                                                             equations_fun, dopa_dfun, True)))
        inters, As = runv(pravel)
        eivals, _ = numpy.linalg.eig(As)
        if len(pkeys)>1:
            eivals = eivals.reshape(pshape[0], pshape[1], y0.size)
    elif sweep_var == 'cy':
        cys = tuple([np.meshgrid(*cy)[i].flatten() for i in range(len(cy))])
        runv = jax.jit(jax.vmap(lambda cys: find_intersections(y0, cys, params, True)))
        inters, As = runv(cys)
        eivals, _ = numpy.linalg.eig(As)
    else:
        raise ValueError('sweep_var is either "params" or "cy"')    
    return eivals


# based on nm.dopaMF_dfun
def get_nullcline_r(V0, y0, cy, p):
    '''y0 now contains V, shape = (6,)
    r0 is the initial guess (or initial condition) for r.
    '''
    r0 = y0[0]
    def dr_fun(r0, V0, y0, cy, p):
        r = r0
        V = V0
        _, _, u, Sa, Sg, Dp, M = y0
        tup = (r, V, u, Sa, Sg, Dp, M)
        dr = nm.dopa_dr(tup, cy, p)
        return dr
    pdr_fun = Partial(dr_fun, V0=V0, y0=y0, cy=cy, p=p)  # or functools.partial
    return numerics.newton(pdr_fun, r0)


def get_nullcline_V(r0, y0, cy, p):
    '''y0 of shape (7,)'''
    _, _, u, Sa, Sg, Dp, M = y0
    c_inh, c_exc, c_dopa = cy # not used -- for convention
    a, b, c, alpha, beta, uj, Bd, ga, ea, gg, eg, Iext, Sja, Sjg, tauSa, tauSg, k, Km, Vmax, tauDe, tauM, Rd, Sp, Eta, *_ = p    
    B = b -  (M + Bd)*ga*Sa - gg*Sg
    C = c + Eta + (M + Bd)*ga*Sa*ea + gg*Sg*eg + Iext - u - ((np.pi**2*r0**2)/(a))
    discriminant = B**2 - 4*a*C
    V1 = jax.lax.select(discriminant >= 0, (-B+np.sqrt(discriminant))/(2*a), np.nan)
    V2 = jax.lax.select(discriminant >= 0, (-B-np.sqrt(discriminant))/(2*a), np.nan)
    Nan_val = -B / (2 * a)
    return np.c_[V1, V2].squeeze(), Nan_val


def correct_nullclines(nullcline_Vs, Nan_val):
    '''no vmap compatible'''
    # Nan_val = -B / (2 * a)
    V1, V2 = nullcline_Vs.T
    X_index = np.where(np.isnan(V1))
    if (len(X_index[0]) != 0):
        # 1st curve
        V1 = V1.at[X_index[0][0]].set(Nan_val)
        V2 = V2.at[X_index[0][0]].set(Nan_val)
        # 2nd curve
        V1 = V1.at[X_index[0][-1]].set(Nan_val)
        V2 = V2.at[X_index[0][-1]].set(Nan_val)
    return np.c_[V1, V2]

def dopaMF_rV_dfun(y0, cy, p):
    dr = nm.dopa_dr(y0, cy, p)
    dv = nm.dopa_dv(y0, cy, p)
    return np.array([dr, dv])