import numpy as np

def Vu_reparam(C, k, v_r, v_t, I, alpha, beta, ci, delta, vpeak, ea, ga, y0):
    '''
    Reparameterization based on the `V` and `u` variables
    for the mean field to match the book equations (8.5), (8.6).
    This reparameterization considers: V_new = V - v_r; u_new = u/C;
    Inputs:
    -------
    C, k, v_r, v_t, I, alpha, beta, ci, delta, vpeak: floats
        book parameters
    y0: list,
        initial conditions for `V` and `u`
    Returns:
    --------
    y0: list,
        rescaled initial conditions
    args: tuple,
        (a, b, c, I, alpha, beta)
        Use `I` as the `c` parameter of the mean field.
    ci, delta, vpeak: floats
        only delta is actually used in the mean field,
        as the parameter for `uj`. The rest is handled
        internally by the firing rate `r`.
    '''
    a = k / C; b = k * (v_r - v_t) / C; c = 0; I = I / C; 
    alpha = alpha; beta = beta / C; 
    ci = ci - v_r; delta = delta / C; 
    vpeak = vpeak - v_r;
    ea = ea - v_r; ga = ga / C;
    y0 = [y0[0] - v_r, y0[1] / C];
    args = (a, b, c, I, alpha, beta)
    return y0, args, ci, delta, vpeak, ea, ga


def Vt_reparam(C, k, v_r, v_t, I, alpha, beta, ci, delta, vpeak, y0):
    '''
    Reparameterization based on `V` and time `t`
    for the mean field to match the book equations (8.5), (8.6).
    This reparameterization considers: v_new = v - v_r; t_new = t/C;
    however the time of the simulation is not reparameterized here,
    and thus it needs to be done separately.
    Inputs:
    -------
    C, k, v_r, v_t, I, alpha, beta, ci, delta, vpeak: floats
        book parameters
    y0: list,
        initial conditions for `V` and `u`
    Returns:
    --------
    y0: list,
        rescaled initial conditions
    args: tuple,
        (a, b, c, I, alpha, beta)
        Use `I` as the `c` parameter of the mean field.
    ci, delta, vpeak: floats
        only delta is actually used in the mean field,
        as the parameter for `uj`. The rest is handled
        internally by the firing rate `r`.
    '''
    a = k; b = k * (v_r - v_t); c = 0; I = I;
    alpha = alpha * C; beta = beta;
    ci = ci - v_r; delta = delta;
    y0 = [y0[0] - v_r, y0[1]]; vpeak = vpeak - v_r
    args = (a, b, c, I, alpha, beta)
    return y0, args, ci, delta, vpeak


def ChCa_reparam(C, k, v_r, v_t, I, alpha, beta, ci, delta, ea, ga, Sjump, tauSa, vpeak):
    t_w     = 1/alpha
    v_ra    = np.abs(v_r)
    vpeak   = 1 + vpeak/v_r
    a       = 1 + v_t/v_ra
    alpha   = (t_w*k*v_ra/C)**(-1)
    sjump   = Sjump * C/(k*v_ra)
    er      = 1 + (ea/v_ra)
    tau_s   = tauSa*k*v_ra/C
    v_reset = 1 + ci/v_ra
    ga      = ga/(k*v_ra)
    beta    = beta/(k*v_ra)
    uj      = delta/(k*v_ra**2)
    I       = I/(k*v_ra**2)
    v_dim   = lambda v, v_ra: v*v_ra - 1
    u_dim   = lambda u, k, v_ra: u*(k*v_ra**2)
    T_dim   = lambda C, t, k, v_ra: C * t / (k*v_ra)  
    return vpeak, a, alpha, sjump, er, tau_s, v_reset, ga, beta, uj, I, v_dim, u_dim, T_dim