import matplotlib.pyplot as plt
import os
import utilities
from decimal import Decimal
import numpy as np

def get_if(xs, ip, fp=None):
    '''given xs and a timestamp in [ms],
    get initial (i) and final (f) timepoints
    for plotting.
    '''
    i = np.where(np.isclose(xs, ip))[0][0]
    if isinstance(fp, type(None)):
        return i
    else:
        f = np.where(np.isclose(xs, fp))[0][0]
        return i, f

def format_float(numbers, e=0, decimals = 8):
    '''if you want scientific notation keep e>0
    and play with that number depending on the precision
    you need.
    Otherwise play with decimals.
    '''
    format_string = f'%.{decimals}f'
    out = [format_string % number for number in numbers]
    if e > 0:
        out = [f"{Decimal(string):.{e}e}" for string in out]
    return out

def bif_eivals_plot(eivals, Var_range, var_str, rylim=[-40,20]):
    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.plot(Var_range, eivals.real, '-o', ms=2)
    plt.hlines(0, Var_range[0], Var_range[-1], color='k')
    plt.xlabel(var_str)
    plt.ylabel(r'$max(\Re(\lambda))$')
    plt.xscale("log")
    plt.ylim(rylim)
    plt.grid()
    plt.subplot(212)
    plt.plot(Var_range, eivals.imag, '-o', ms=2)
    plt.hlines(0, Var_range[0], Var_range[-1], color='k')
    plt.xlabel(var_str)
    plt.ylabel(r'$max(\Im(\lambda))$')
    plt.xscale("log")
    plt.grid()


def plot_tr_rV(xs, ys):
    r = ys[:,0]
    V = ys[:,1]
    plt.subplot(121)
    plt.plot(xs, r)
    plt.xlabel('t [ms]'); plt.ylabel('r')
    plt.subplot(122)
    plt.plot(r, V)
    plt.xlabel('r'); plt.ylabel('V')

def save_sweeps(xs, ys, Var_range, var_string,
                 head_path, folder_rel_path, figsize=(7,3), e=2):
    fpath = os.path.join(head_path, folder_rel_path)
    utilities.create_folder(fpath)
    var = format_float(Var_range, e=e)
    for v in range(len(Var_range)):
        plt.figure(figsize=figsize)
        y = ys[v]
        plot_tr_rV(xs, y)
        plt.suptitle(rf'${var_string}={var[v]}$')
        plt.tight_layout()
        plt.savefig(os.path.join(fpath, f'{v}-{var_string}{var[v]}.png'), dpi=200)
        plt.close()