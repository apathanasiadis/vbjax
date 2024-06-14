import jax
import jax.numpy as np
import numpy
import os
import nm

# from marmaduke: https://github.com/ipython/ipython/issues/10045
def show_gif(fname):
    import base64
    from IPython import display
    with open(fname, 'rb') as fd:
        b64 = base64.b64encode(fd.read()).decode('ascii')
    return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')

def show_gif_grid(directory, output_file='gif_grid.html'):
    import os
    import base64
    from IPython.display import HTML, display
    
    # Get a list of all files in the directory
    gif_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.gif')]
    gif_files.sort()

    # Calculate the number of columns for the grid layout
    num_files = len(gif_files)
    num_columns = np.ceil(num_files/2).astype('i')  # min(num_files, max_columns)  # Maximum 4 columns

    # Generate HTML code to display GIFs in a grid layout
    html_code = f'<div style="display: grid; grid-template-columns: repeat({num_columns}, 1fr); grid-gap: 0px;">'
    for gif_file in gif_files:
        with open(os.path.join(directory, gif_file), 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        html_code += f'<img src="data:image/gif;base64,{b64}" style="width:100%;">'
    html_code += '</div>'

    # Display the HTML code
    display(HTML(html_code))

    # Save the HTML code to a file
    with open(output_file, 'w') as f:
        f.write(html_code)

    return output_file


def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    except FileExistsError:
        print(f"Folder already exists at {folder_path}")


def animate_stream(fig, ys, cy, p,
                   x_min, x_max, y_min, y_max, gif_name, path_folder=os.getcwd(), 
                   n_frames=100, n_samples=256, init_tp=0, fps=60,
                   display_nullclines=True):
    import analytics
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import FancyArrowPatch
    # from matplotlib.lines import Line2D
    from functools import partial

    path = os.path.join(path_folder, gif_name)
    name = gif_name.split('.')[:-1]
    name = '.'.join(name).split('_')[1]
    u, Sa, Sg, Dp, M = ys[init_tp, 2:]
    x_range = numpy.linspace(x_min, x_max, n_samples)
    y_range = numpy.linspace(y_min, y_max, n_samples)
    X, Y = numpy.meshgrid(x_range, y_range)
    # initialize
    # y0_sweep = (X, Y, u, Sa, Sg, Dp, M)
    y0 = ys[0, :]
    plt.xlabel('dr'); plt.ylabel('dV');
    plt.ylim(y_range.min(), y_range.max()); plt.xlim(x_range.min(), x_range.max())
    pbar = tqdm(total=n_frames)

    def animate(iter, n_frames=n_frames, y0=y0):  # undefined: X, Y, cy, p, y0
        '''iter = n_frames
        ys = numerically calculated system, shape (n_samples, n_state_variables)

        '''
        # Clear previous lines and collections
        for line in plt.gca().lines:
            line.remove()
        for coll in plt.gca().collections:
            coll.remove()
        for artist in plt.gca().get_children():
            if isinstance(artist, FancyArrowPatch):
                artist.remove()
        N = ys.shape[0]
        d = np.ceil(N/n_frames).astype('i')
        idx_array = np.r_[0: N: d]
        idx = idx_array[iter]
        u, Sa, Sg, Dp, M = ys[idx, 2:]
        y0 = y0.at[2:].set([u, Sa, Sg, Dp, M])
        y0_sweep = (X, Y, u, Sa, Sg, Dp, M)
        # dr, dV, du, dSa, dSg, dDp, dM = nm.dopaMF_dfun(y0_sweep, cy, p)
        in_axes_tuple = ((0,0,None,None,None,None,None),None,None)
        tmp = jax.vmap(jax.vmap(nm.dopaMF_dfun, in_axes=in_axes_tuple), in_axes=in_axes_tuple)(y0_sweep, cy, p)
        dr, dV, du, dSa, dSg, dDp, dM = tmp.transpose(-1,-2,-3)
        color = numpy.array(np.sqrt(dr**2 + dV**2))
        stream = plt.streamplot(X, Y, dr, dV, color=color, cmap='rainbow_r',
                                density=1.4, arrowsize=1)
        plt.contour(X, Y, dr, levels=[0], linewidths=1.5, colors='green', linestyles='dashed')
        if display_nullclines:
            nullcline_r = jax.vmap(analytics.get_nullcline_r, in_axes=(0, None, None, None))(y_range, y0, cy, p)
            nullcline_V, Nan_value = jax.vmap(analytics.get_nullcline_V, in_axes=(0, None, None, None))(x_range, y0, cy, p)
            nullcline_V = nullcline_V.squeeze()
            Nan_value = Nan_value[0]
            nullcline_V = analytics.correct_nullclines(nullcline_V, Nan_value)
            plt.plot(nullcline_r, y_range, color='green', label='nullcline r')
            plt.plot(x_range, nullcline_V, color='k', label='nullcline V')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        plt.title(f'Phase Diagram for {name}: \n u={u:.3f}, Sa={Sa:.3f}, Sg={Sg:.3f}, Dp={Dp:.3f}, M={M:.3f}')
        pbar.update(1)
        return stream

    # with tqdm(total=n_frames) as pbar:
    anim = animation.FuncAnimation(fig, partial(animate, n_frames=n_frames, y0=y0),
                                       frames=n_frames, interval=50, blit=False, repeat=False)
    create_folder(path_folder)
    anim.save(path, writer='imagemagick', fps=fps)
    return anim