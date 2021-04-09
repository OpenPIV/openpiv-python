"""
functions to plot 3D-deformation fields and simple 3D-structures
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D


def set_axes_equal(ax):

    """
    Following https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

      Parameters
    ----------
     ax: matplotlib.axes object
       
     
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def scatter_3D(a, cmap="jet", sca_args=None, control="color", size=60):

    # default arguments for the quiver plot. can be overwritten by quiv_args
    if not isinstance(sca_args, dict):
        sca_args = {}
    scatter_args = {"alpha": 1}
    scatter_args.update(sca_args)

    x, y, z = np.indices(a.shape)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', rasterized=True)

    if control == "color":
        # make cmap
        cbound = [np.nanmin(a), np.nanmax(a)]
        # create normalized color map for arrows
        norm = matplotlib.colors.Normalize(
            vmin=cbound[0], vmax=cbound[1]
        )  # 10 ) #cbound[1] ) #)
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # different option
        cm = matplotlib.cm.get_cmap(cmap)
        colors = cm(norm(a)).reshape(a.shape[0] * a.shape[1] * a.shape[2], 4)  #
        # plotting
        nan_filter = ~np.isnan(a.flatten())
        ax.scatter(
            x[nan_filter],
            y[nan_filter],
            z[nan_filter],
            c=colors[nan_filter],
            s=size,
            **scatter_args
        )
        plt.colorbar(sm)

    if control == "alpha":
        # untested #
        colors = [(0, 0, 1, x / np.max(z)) for x in np.ravel(z)]
        ax.scatter(x, y, z, c=colors, s=size, **scatter_args)
        plt.show()

    if control == "size":
        sizes = (a - a.min()) * size / a.ptp()
        ax.scatter(x, y, z, a, s=sizes, **scatter_args)
        ax_scale = plt.axes([0.88, 0.1, 0.05, 0.7])
        # ax_scale.set_ylim((0.1,1.2))
        nm = 5
        ax_scale.scatter(
            [0] * nm,
            np.linspace(a.min(), a.max(), nm),
            s=sizes.max() * np.linspace(0, 1, nm),
        )
        ax_scale.spines["left"].set_visible(False)
        ax_scale.spines["right"].set_visible(True)
        ax_scale.spines["bottom"].set_visible(False)
        ax_scale.spines["top"].set_visible(False)
        ax_scale.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labelleft=False,
            labelright=True,
            bottom=False,
            left=False,
            right=True,
        )

    ax.set_xlim(0, a.shape[0])
    ax.set_ylim(0, a.shape[1])
    ax.set_zlim(0, a.shape[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return fig


def explode(data):
    # following "https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_numpy_logo.html"

    if len(data.shape) == 3:
        size = np.array(data.shape) * 2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
    if len(data.shape) == 4:  ## color data
        size = np.array(data.shape)[:3] * 2
        data_e = np.zeros(np.concatenate([size - 1, np.array([4])]), dtype=data.dtype)
        data_e[::2, ::2, ::2, :] = data

    return data_e


def plot_3D_alpha(data):
    # plotting each voxel as a slightly smaller block with transparency depending
    # on the data value
    # following "https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_numpy_logo.html"

    col = np.zeros((data.shape[0], data.shape[1], data.shape[2], 4))

    data_fil = data.copy()
    data_fil[(data == np.inf)] = np.nanmax(data[~(data == np.inf)])
    data_fil = (data_fil - np.nanmin(data_fil)) / (
        np.nanmax(data_fil) - np.nanmin(data_fil)
    )
    data_fil[np.isnan(data_fil)] = 0

    col[:, :, :, 2] = 1
    col[:, :, :, 3] = data_fil

    col_exp = explode(col)
    fill = explode(np.ones(data.shape))

    x, y, z = np.indices(np.array(fill.shape) + 1).astype(float) // 2

    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.voxels(x, y, z, fill, facecolors=col_exp, edgecolors=col_exp)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def quiver_3D(
    u,
    v,
    w,
    x=None,
    y=None,
    z=None,
    mask_filtered=None,
    filter_def=0,
    filter_reg=(1, 1, 1),
    cmap="jet",
    quiv_args=None,
    vmin=None,
    vmax=None,
    arrow_scale=0.15,
    equal_ax=True,
):
    """ 
    Displaying 3D deformation fields vector arrows

    Parameters
    ----------
     u,v,w: 3d ndarray or lists
         arrays or list with deformation in x,y and z direction

     x,y,z: 3d ndarray or lists
          Arrays or list with deformation the coordinates of the deformations.
          Must match the dimensions of the u,v qnd w. If not provided x,y and z are created
          with np.indices(u.shape)

     mask_filtered, boolean 3d ndarray or 1d ndarray
          Array, or list with same dimensions as the deformations. Defines the area where deformations are drawn
          
     filter_def: float
          Filter that prevents the display of deformations arrows with length < filter_def
          
     filter_reg: tuple,list or int
          Filter that prevents the display of every i-th deformations arrows separatly alon each axis.
          filter_reg=(2,2,2) means that only every second arrow along x,y z axis is displayed leading to
          a total reduction of displayed arrows by a factor of 8. filter_reg=3 is interpreted
          as (3,3,3).
          
     cmap: string
          matplotlib colorbar that defines the coloring of the arrow
          
     quiv_args: dict
         Dictionary with kwargs passed on to the matplotlib quiver function.

     vmin,vmax: float
         Upper and lower bounds for the colormap. Works like vmin and vmax in plt.imshow().
         
    arrow_scale: float
        Automatic scaling of the quiver arrows so that the longest arrow has the 
        length axis length * arrow_scale. Arrow length can alternatively be set by
        passing a "lenght" argument in quiv_args. 
    
    equal_axes: bool
        resize the figure axis so that they are have equal scaling.
    

    Returns
    -------
     fig: matploltib figure object

     ax: mattplotlib axes object
         the holding the main 3D quiver plot

    """

    # default arguments for the quiver plot. can be overwritten by quiv_args
    quiver_args = {
        "normalize": False,
        "alpha": 0.8,
        "pivot": "tail",
        "linewidth": 1,
        "length": 1,
    }
    if isinstance(quiv_args, dict):
        quiver_args.update(quiv_args)
    # overwriting length if an arrow scale and a "length" argument in quiv_args
    # is provided at the same
    if arrow_scale is not None:
        quiver_args["length"] = 1

    # convert filter ot list if proveided as int
    if not isinstance(filter_reg, (tuple, list)):
        filter_reg = [filter_reg] * 3

    # generating coordinates if not provided
    if x is None:
        # if you provide deformations as a list
        if len(u.shape) == 1:
            x, y, z = [np.indices(u.shape)[0] for i in range(3)]
        # if you provide deformations as an array
        elif len(u.shape) == 3:
            x, y, z = np.indices(u.shape)
        else:
            raise ValueError(
                "displacement data has wrong number of dimensions (%s). Use 1d array, list, or 3d array."
                % str(len(u.shape))
            )

    # conversion to array
    x, y, z = np.array([x, y, z])

    deformation = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    if not isinstance(mask_filtered, np.ndarray):
        mask_filtered = deformation > filter_def
        if isinstance(filter_reg, list):
            show_only = np.zeros(u.shape).astype(bool)
            # filtering out every x-th
            show_only[:: filter_reg[0], :: filter_reg[1], :: filter_reg[2]] = True
            mask_filtered = np.logical_and(mask_filtered, show_only)

    xf = x[mask_filtered]
    yf = y[mask_filtered]
    zf = z[mask_filtered]
    uf = u[mask_filtered]
    vf = v[mask_filtered]
    wf = w[mask_filtered]
    df = deformation[mask_filtered]

    # make cmap
    # create normalized color map for arrows
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)  # 10 ) #cbound[1] ) #)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # different option
    colors = matplotlib.cm.jet(norm(df))  #

    colors = [c for c, d in zip(colors, df) if d > 0] + list(
        chain(*[[c, c] for c, d in zip(colors, df) if d > 0])
    )
    # colors in ax.quiver 3d is really fucked up/ will probably change with updates:
    # requires list with: first len(u) entries define the colors of the shaft, then the next len(u)*2 entries define
    # the color ofleft and right arrow head side in alternating order. Try for example:
    # colors = ["red" for i in range(len(cf))] + list(chain(*[["blue", "yellow"] for i in range(len(cf))]))
    # to see this effect.
    # BUT WAIT THERS MORE: zeor length arrows are apparently filtered out in the matplolib with out filtering
    # the color list appropriately so we have to do this our selfs as well

    # scale arrows to axis dimensions:
    ax_dims = [(x.min(), x.max()), (y.min(), y.max()), (z.min(), z.max())]
    if arrow_scale is not None:
        max_length = df.max()
        max_dim_length = np.max([(d[1] - d[0] + 1) for d in ax_dims])
        scale = max_dim_length * arrow_scale / max_length
    else:
        scale = 1

    # plotting
    fig = plt.figure()
    ax = fig.gca(projection="3d", rasterized=True)
    ax.quiver(
        xf, yf, zf, vf * scale, uf * scale, wf * scale, colors=colors, **quiver_args
    )
    plt.colorbar(sm)

    ax.set_xlim(ax_dims[0])
    ax.set_ylim(ax_dims[1])
    ax.set_zlim(ax_dims[2])

    if equal_ax:
        set_axes_equal(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.w_xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))

    return fig
