import numpy as np
import jax
import jax.numpy as jnp

def get_fluidity(x,
             TR=1.0,
             window_length=30,
             positive=True,
             masks={},
             metrics=['variance', 'mean', 'range'],
             get = None):
    """
    calculates FCD and subsequently fluidity metrics for different networks.
    Input

    Parameters
    ----------

    x: np.ndarray [n_regions, n_samples]
        input array
    TR: float
        time per scanning.
    window_length: int
    positive: bool
        If True, only positive correlations are considered, in consistence with Lavagna et al. 2023, Stumme et al., 2020.
    masks: dictionary

    metrics: list
        valid_metrics = ['variance', 'mean'], with output metric(fcd_ut)
    get: str
        select from "FCDs", "FCDs_ut", "FC_streams", "FCDs_metrics".
        Default value is None, and it calculates for "FCDs_metrics"
    Return
    ----------------------------
    
    """
    ts = x.T
    nt, nn = ts.shape    
    mask_full = jnp.ones((nn, nn))
    # calculate windowed FC ### calculate FCD ###
    windowed_data = np.lib.stride_tricks.sliding_window_view(
        ts, (int(window_length/TR), nn), axis=(0, 1)).squeeze()
    n_windows = windowed_data.shape[0]
    fc_stream = jnp.asarray(
        [jnp.corrcoef(windowed_data[i, :, :], rowvar=False) for i in range(n_windows)])
    if len(masks) == 0: #! TODO check getting length of dictionary
        masks["full"] = mask_full
    ut_idx = jnp.triu_indices_from(mask_full, k=1)
    ut_fc_stream = fc_stream[:,ut_idx[0], ut_idx[1]]
    fcd = jnp.corrcoef(ut_fc_stream, rowvar=True)
    ut_idx = jnp.triu_indices_from(fcd, k=int(window_length/TR))
    # print(ut_idx[0].size)
    ut_fcd = fcd[ut_idx[0], ut_idx[1]]
    valid_metrics = ['variance', 'mean', 'range']
    for j, key in enumerate(masks.keys()):
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Please choose from {valid_metrics}.")
            elif metric == 'variance':
                fluidity_var = jnp.var(ut_fcd)
            elif metric == 'mean':
                fluidity_mean = jnp.mean(ut_fcd)
            elif metric == 'range':
                fluidity_range = ut_fcd.max()-ut_fcd.min()
    return fcd, fluidity_var, fluidity_mean, fluidity_range


def get_FC_SC_corr(r, SC):
    '''r.shape = samples x features
    returns coef scalar.
    '''
    FC = jnp.corrcoef(r, rowvar=False)
    ut_idx = jnp.triu_indices_from(FC, k=1)
    FC_ut = FC[ut_idx]
    SC_ut = SC[ut_idx]
    return jnp.corrcoef(FC_ut, SC_ut)[1, 0]
    

get_fs = lambda dt: 1/(dt*1e-3)
freq_size = lambda d, ts: int((1/d/2)/(1/(ts.size*d)))
def get_ps(d, ts):
    '''d is the sampling density'''
    d = jnp.array(d)
    N = ts.shape[0]                       # Define the total number of data points
    T = N * d                            # Define the total duration of the data
    
    xf = jnp.fft.fft(ts - ts.mean())           # Compute Fourier transform of x
    Sxx = 2 * d ** 2 / T * (xf * xf.conj())  # Compute spectrum
    Sxx = Sxx[:int(len(ts) / 2)]          # Ignore negative frequencies
    
    df = 1 / T.max()                      # Determine frequency resolution
    fNQ = 1 / d / 2                      # Determine Nyquist frequency
    faxis = jnp.arange(0, fNQ, df)        # Construct frequency axis
    return faxis, Sxx.real