import jax
import jax.numpy as np
import numpy
import analytics

class PhasePlane():
    '''class to plot for the phase plane'''
    def __init__(self, r_min, r_max, V_min, V_max, n_samples):
        '''specify the ranges of r and V, as well as the n_samples
        inputs: r_min, r_max, V_min, V_max, n_samples
        '''
        self.r_min = r_min
        self.r_max = r_max
        self.V_min = V_min
        self.V_max = V_max
        self.n_samples = n_samples
        self.r_range = numpy.linspace(self.r_min, self.r_max, n_samples)
        self.V_range = numpy.linspace(self.V_min, self.V_max, n_samples)

    def fit(self, y, cy, params, dfun):
        self.y = y
        self.cy = cy
        self.params = params
        self.X, self.Y = numpy.meshgrid(self.r_range, self.V_range)
        self.dfun = dfun
        return self
        
    def get_grid_ys(self):
        '''calclulate the system over the meshgrid'''
        y0_grid = (self.X, self.Y, *tuple(self.y[2:]))
        None_tuple = tuple([None for _ in range(self.y.size-2)])
        in_axes_tuple = ((0, 0, *None_tuple), None, None)
        fun = jax.jit(jax.vmap(jax.vmap(self.dfun, in_axes=in_axes_tuple), in_axes=in_axes_tuple))
        self.ys= fun(y0_grid, self.cy, self.params).transpose(-1,0,1)
        return self.ys
    
    def get_rV_nullclines(self):
        '''given the r_range and V_range and the point y,
        whose coordinates correspond to the combination of 
        the values that the state variables take, and given also
        the cy and params arguments, it returns the nullcline_r,
        and nullcline_V.
        '''
        # r nullcline
        fun = jax.vmap(analytics.get_nullcline_r, in_axes=(0,None,None,None))
        self.nullcline_r = fun(self.V_range, self.y, self.cy, self.params)
        # V nullcline
        fun = jax.vmap(analytics.get_nullcline_V, in_axes=(0,None,None,None))
        nullcline_V, Nan_value = fun(self.r_range, self.y, self.cy, self.params)
        Nan_value = Nan_value[0]
        self.nullcline_V = analytics.correct_nullclines(nullcline_V, Nan_value)
        return self.nullcline_r, self.nullcline_V
    
    
    def plot_phaseplane(self, nullclines='contours'):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        dr, dV, *_ = self.ys
        plt.streamplot(self.X, self.Y, dr, dV, linewidth=1, color='#6B6B6B', density=1.8)
        handles, labels = [], []
        if nullclines == 'contours':
            plt.contour(self.X, self.Y, dr, levels=[0], linewidths=1.5, colors='tab:blue')
            plt.contour(self.X, self.Y, dV, levels=[0], linewidths=1.5, colors='tab:orange')
            # Create custom legend handles, as the contour plots dont accept `label` arg
            r_nullcline_handle = Line2D([0], [0], color='tab:blue', label=r'$r$-nullcline')
            V_nullcline_handle = Line2D([0], [0], color='tab:orange', label=r'$V$-nullcline')
            handles.extend([r_nullcline_handle, V_nullcline_handle])
            labels.extend([r'$r$-nullcline', r'$V$-nullcline'])
        elif nullclines == 'lines':
            V_rdiv_max = np.where(self.nullcline_r == np.max(self.nullcline_r))[0][0]
            line1, = plt.plot(self.nullcline_r[:V_rdiv_max + 1], self.V_range[:V_rdiv_max + 1], color='tab:blue', label='$r$-nullcline', lw=2)
            line2, = plt.plot(self.nullcline_r[V_rdiv_max + 1:], self.V_range[V_rdiv_max + 1:], color='tab:blue', lw=2)
            line3, = plt.plot(self.r_range, self.nullcline_V, color='tab:orange', label='$V$-nullcline')
            handles.extend([line1, line3])
            labels.extend([r'$r$-nullcline', r'$V$-nullcline'])
        else:
            pass
        # Add the legend to the plot
        plt.xlim([self.r_range.min(), self.r_range.max()])
        plt.ylim([self.V_range.min(), self.V_range.max()])
        plt.xlabel('$r$')
        plt.ylabel('$V$') 
        return handles, labels