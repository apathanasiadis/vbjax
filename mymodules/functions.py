import jax
import jax.numpy as np
import numpy
import scipy.io
import os
import pandas as pd

def zscore(x):
    return (x-x.mean())/x.std()
def maxscore(x):
    return (x-x.mean())/x.max()

def get_CeCiCd(path_connectome, conmat):
    n_nodes = conmat.shape[0]
    Cd = np.zeros((n_nodes, n_nodes))
    Cd = Cd.at[42, 35].set(1)
    Cd = Cd.at[45, 48].set(1)
    # get Ci
    inh_matrix_scheme = os.path.join(path_connectome, "inhibitory_matrix_scheme.mat")
    inh_matrix = scipy.io.loadmat(inh_matrix_scheme)
    key = "I"
    Ci = np.array(inh_matrix[key])
    # get Ce
    exc_matrix_scheme = os.path.join(path_connectome, "excitatory_matrix_scheme.mat")
    exc_matrix = scipy.io.loadmat(exc_matrix_scheme)
    key = "A"
    Ce = np.array(exc_matrix[key])
    Cd *= conmat
    Ce *= conmat
    Ci *= conmat
    return Ce, Ci, Cd

def get_connectome(path_connectome, option = 'max'):
    '''
    Return
    ------
    if log=False: connectome_matrix, Ce, Ci, Cd, SC, rois
    else: logconn, connectome_matrix, factor, Ce, Ci, Cd, SC, rois
    '''
    options = ['max', 'max-log', 'log-max']
    num_files = 10
    matrices = []
    for i in range(num_files):
        file_name = os.path.join(path_connectome, f'weights{i}.mat')
        mat = scipy.io.loadmat(file_name)
        key = f'weights{i}'
        if key in mat:
            data = np.array(mat[key])
            matrices.append(data)
    connectome_matrix = np.asarray(matrices).mean(0)
    unconsidered_regions = np.array([39, 43, 47, 53])
    connectome_matrix_rows = np.delete(connectome_matrix,unconsidered_regions, axis=0)
    connectome_matrix = np.delete(connectome_matrix_rows,unconsidered_regions, axis=1) 
    numpy.testing.assert_allclose(connectome_matrix, connectome_matrix.T)  # asserting that the connectome_matrix is symmetric
    n_nodes = connectome_matrix.shape[0]
    if option=='max':
        data = np.load(os.path.join(path_connectome,'connectome3.npz'))
        conn_inh = data['conn_inh']
        conn_dop = data['conn_dop']
        conn_exc = data['conn_exc']
        maxx = np.stack((conn_dop, conn_exc, conn_inh)).max()
        Ci = conn_inh / maxx
        Cd = conn_dop / maxx
        Ce = conn_exc / maxx
        SC = np.abs(Cd) + np.abs(Ce) + np.abs(Ci)
    elif option == 'max-log':
        normconn = connectome_matrix/connectome_matrix.max()
        logconn = np.log10(normconn + 1)
        factor = logconn.sum()/normconn.sum()
        Ce, Ci, Cd = get_CeCiCd(path_connectome, logconn)
        SC = np.abs(Cd) + np.abs(Ce) + np.abs(Ci)
    elif option == 'log-max':
        logconn = np.log10(connectome_matrix+1)
        factor = logconn.sum()/connectome_matrix.sum()
        logconn /= logconn.max()
        Ce, Ci, Cd = get_CeCiCd(path_connectome, logconn)
        SC = np.abs(Cd) + np.abs(Ce) + np.abs(Ci)
    else:
        raise ValueError(f'option {option} not in {options}')
    rois = pd.read_fwf( os.path.join(path_connectome, "list_regions.txt"), index_col=False)
    idx, counts = np.unique(rois["0"].values, return_counts=True)
    rois = rois.drop(idx[np.where(counts==2)])
    rois = rois.reset_index(drop=True)
    rois = rois.drop(columns="0")
    rois = rois.dropna(axis=1, how='any')
    rois = rois.drop(unconsidered_regions).reset_index(drop=1)
    if option=='max':
        return connectome_matrix, Ce, Ci, Cd, SC, n_nodes, rois
    else:
        return logconn, connectome_matrix, factor, Ce, Ci, Cd, SC, rois