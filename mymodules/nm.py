import collections
import jax
import jax.numpy as np
import vbjax as vb

# parameters
DopaParams = collections.namedtuple(typename='DopaParams',
                                    field_names='a, b, c, alpha, beta, uj, Bd, ga, ea, gg, eg, Iext, Sja, Sjg, tauSa, tauSg, k, Km, Vmax, tauDe, tauM, Rd, Sp, Eta, Delta, wi, we, wd, sigma')


dopa_default_params = DopaParams(
  a=0.04, b=5., c=140., alpha=0.013, beta=.4, uj=12,Sja=0.8, Sjg=1.2,
  ga=12., gg=12., ea=0., eg=-80., tauSa=2.6, tauSg=2.6,
  Eta=12., Iext=0., Delta=5., #cdopa=1e-4,
  k=10e4, Vmax=1300.,
  Km=150., Bd=0.2, tauDe=500., tauM=500., Sp=-1., Rd=1.,
  wi=1.e-4, we=1.e-4, wd=1.e-4, sigma=1e-3,
)

DopaState = collections.namedtuple(typename='DopaState',
                                   field_names='r, v, u, Sa, Sg, De, M')

dopa_default_initial_state = DopaState(
 r=2, v=-60.0, u=7.9, Sa=0.04,
 Sg=1e-5, De=0.001, M=0.7,
 )

dopa_default_origin = DopaState(
 r=.0, v=.0, u=.0, Sa=.0,
 Sg=.0, De=.0, M=.0,
 )

# partial functions
def dopa_dr(y0, cy, p):
    r, v, u, Sa, Sg, De, M = y0
    c_inh, c_exc, c_dopa = cy
    dr = 2.*p.a*r*v + p.b*r - (M + p.Bd)*p.ga*Sa*r - p.gg*Sg*r + (p.a*p.Delta)/np.pi
    return dr
def dopa_dv(y0, cy, p):
    r, v, u, Sa, Sg, De, M = y0
    dv = p.a*(v**2) + p.b*v + p.c + p.Iext + (M + p.Bd)*p.ga*Sa*(p.ea - v) + p.gg*Sg*(p.eg - v) - u + p.Eta - (np.pi**2 * r**2) /p.a
    return dv
def dopa_du(y0, cy, p):
    r, v, u, Sa, Sg, De, M = y0    
    du = p.alpha*(p.beta*v - u) + p.uj*r
    return du
def dopa_dSa(y0, cy, p):
    r, v, u, Sa, Sg, De, M = y0
    c_inh, c_exc, c_dopa = cy
    dSa = - Sa/p.tauSa + p.Sja*(r + c_exc)
    return dSa
def dopa_dSg(y0, cy, p):
    r, v, u, Sa, Sg, De, M = y0
    c_inh, c_exc, c_dopa = cy
    dSg = - Sg/p.tauSg + p.Sjg*c_inh
    return dSg
def dopa_dDe(y0, cy, p):
    r, v, u, Sa, Sg, De, M = y0
    c_inh, c_exc, c_dopa = cy
    dDe =  (p.k*c_dopa - p.Vmax*De/(p.Km + De)) / p.tauDe
    return dDe
def dopa_dM(y0, cy, p):
    r, v, u, Sa, Sg, De, M = y0
    dM = (-M + p.Rd/(1 + np.exp(p.Sp*(De + 1)))) / p.tauM
    return dM

# System
def dopaMF_explicit_dfun(y0, cy, p: DopaParams):
    r, v, u, Sa, Sg, De, M = y0
    c_inh, c_exc, c_dopa = cy    
    dr = 2.*p.a*r*v + p.b*r - (M + p.Bd)*p.ga*Sa*r - p.gg*Sg*r + (p.a*p.Delta)/np.pi
    dv = p.a*(v**2) + p.b*v + p.c + p.Iext + (M + p.Bd)*p.ga*Sa*(p.ea - v) + p.gg*Sg*(p.eg - v) - u + p.Eta - (np.pi**2 * r**2) /p.a
    du = p.alpha*(p.beta*v - u) + p.uj*r
    dSa = - Sa/p.tauSa + p.Sja*(r + c_exc)
    dSg = - Sg/p.tauSg + p.Sjg*c_inh
    dDe = (p.k*c_dopa - p.Vmax*De/(p.Km + De)) / p.tauDe
    dM = (-M + p.Rd/(1 + np.exp(p.Sp*(De + 1)))) / p.tauM
    return np.array([dr, dv, du, dSa, dSg, dDe, dM])

def dopaMF_dfun(y0, cy, p: DopaParams):
    dr = dopa_dr(y0, cy, p)
    dv = dopa_dv(y0, cy, p)
    du = dopa_du(y0, cy, p)
    dSa = dopa_dSa(y0, cy, p)
    dSg = dopa_dSg(y0, cy, p)
    dDe = dopa_dDe(y0, cy, p)
    dM = dopa_dM(y0, cy, p)
    return np.array([dr, dv, du, dSa, dSg, dDe, dM])

# Fast Subsystem
def dopaMF_explicit_fast_dfun(y0, cy, params):
    r, V = y0
    c_inh, c_exc, c_dopa = cy    
    sv, p = params
    dr = 2. * p.a * r * V + p.b * r - p.ga * sv.Sa * r - p.gg * sv.Sg * r + (p.a * p.Delta) / np.pi
    dV = p.a* V**2 + p.b* V + p.c + p.Eta - (np.pi**2*r**2) / p.a + (sv.M + p.Bd) *p.ga *sv.Sa *(p.ea - V) + p.gg *sv.Sg * (p.eg - V) + p.Iext - sv.u
    return np.array([dr, dV])


def dopaMF_fast_dfun(y0, cy, p: DopaParams):
    dr = dopa_dr(y0, cy, p)
    dv = dopa_dv(y0, cy, p)
    return np.array([dr, dv])


# Network
def dopa_net_dfun(y, p):
    "Canonical form for network of dopa nodes."
    Ci, Ce, Cd, node_params = p
    r = y[0]
    c_inh = node_params.wi * Ci @ r
    c_exc = node_params.we * Ce @ r
    c_dopa = node_params.wd * Cd @ r
    return dopaMF_dfun(y, (c_inh, c_exc, c_dopa), node_params)

