'''
functions to plot 3D-deformation fields and simple 3D-structures
'''


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D


def scatter_3D(a, cmap="jet", sca_args={}, control="color", size=60):
    # default arguments for the quiver plot. can be overwritten by quiv_args
    scatter_args = {"alpha": 1}
    scatter_args.update(sca_args)

    x, y, z = np.indices(a.shape)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    fig = plt.figure()
    ax = fig.gca(projection='3d', rasterized=True)

    if control == "color":
        # make cmap
        cbound = [0, np.nanmax(a)]
        # create normalized color map for arrows
        norm = matplotlib.colors.Normalize(vmin=cbound[0], vmax=cbound[1])  # 10 ) #cbound[1] ) #)
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # different option
        cm = matplotlib.cm.get_cmap(cmap)
        colors = cm(norm(a)).reshape(a.shape[0] * a.shape[1] * a.shape[2], 4)  #
        # plotting
        nan_filter = ~np.isnan(a.flatten())
        ax.scatter(x[nan_filter], y[nan_filter], z[nan_filter], c=colors[nan_filter], s=size, **scatter_args)
        plt.colorbar(sm)

    if control == "alpha":
        # untested####
        col = [(0, 0, 1, x / np.max(z)) for x in np.ravel(z)]
        ax.scatter(x, y, z, c=colors, s=size, **scatter_args)
        plt.show()

    if control == "size":
        sizes = (a - a.min()) * size / a.ptp()
        ax.scatter(x, y, z, a, s=sizes, **scatter_args)
        ax_scale = plt.axes([0.88, 0.1, 0.05, 0.7])
        # ax_scale.set_ylim((0.1,1.2))
        nm = 5
        ax_scale.scatter([0] * nm, np.linspace(a.min(), a.max(), nm), s=sizes.max() * np.linspace(0, 1, nm))
        ax_scale.spines["left"].set_visible(False)
        ax_scale.spines["right"].set_visible(True)
        ax_scale.spines["bottom"].set_visible(False)
        ax_scale.spines["top"].set_visible(False)
        ax_scale.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, labelright=True,
                             bottom=False, left=False, right=True)

        # implement marker scale bar

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
    # following "https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_numpy_logo.html"

    col = np.zeros((data.shape[0], data.shape[1], data.shape[2], 4))

    data_fil = data.copy()
    data_fil[(data == np.inf)] = np.nanmax(data[~(data == np.inf)])
    data_fil = (data_fil - np.nanmin(data_fil)) / (np.nanmax(data_fil) - np.nanmin(data_fil))
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
    ax = fig.gca(projection='3d')
    ax.voxels(x, y, z, fill, facecolors=col_exp, edgecolors=col_exp)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def quiver_3D(u, v, w, x=None, y=None, z=None, image_dim=None, mask_filtered=None, filter_def=0, filter_reg=(1, 1, 1),
              cmap="jet", quiv_args={}, cbound=None):
    # filter_def filters values with smaler absolute deformation
    # nans are also removed
    # setting the filter to <0 will probably mess up the arrow colors
    # filter_reg filters every n-th value, separate for x, y, z axis
    # you can also provide your own mask with mask_filtered !!! make sure to filter out arrows with zero total deformation!!!!
    # other wise the arrows are not colored correctly
    # use indices for x,y,z axis as default - can be specified by x,y,z

    # default arguments for the quiver plot. can be overwritten by quiv_args
    quiver_args = {"normalize": False, "alpha": 0.8, "pivot": 'tail', "linewidth": 1, "length": 20}
    quiver_args.update(quiv_args)
    if not isinstance(image_dim, (list, tuple, np.ndarray)):
        image_dim = np.array(u.shape)

    if x is None:
        x, y, z = np.indices(u.shape) * (np.array(image_dim) / np.array(u.shape))[:, np.newaxis, np.newaxis, np.newaxis]
    else:
        x, y, z = np.array([x, y, z]) * (np.array(image_dim) / np.array(u.shape))[:, np.newaxis]

    deformation = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    if not isinstance(mask_filtered, np.ndarray):
        mask_filtered = deformation > filter_def
        if isinstance(filter_reg, tuple):
            mask_filtered[::filter_reg[0], ::filter_reg[1], ::filter_reg[2]] *= True

    xf = x[mask_filtered]
    yf = y[mask_filtered]
    zf = z[mask_filtered]
    uf = u[mask_filtered]
    vf = v[mask_filtered]
    wf = w[mask_filtered]
    df = deformation[mask_filtered]

    # make cmap
    if not cbound:
        cbound = [0, np.nanmax(df)]
    # create normalized color map for arrows
    norm = matplotlib.colors.Normalize(vmin=cbound[0], vmax=cbound[1])  # 10 ) #cbound[1] ) #)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cm = matplotlib.cm.get_cmap(cmap)
    colors = cm(norm(df))  #

    colors = [c for c, d in zip(colors, df) if d > 0] + list(chain(*[[c, c] for c, d in zip(colors, df) if d > 0]))
    # colors in ax.quiver 3d is really fucked up/ will probably change with updates:
    # requires list with: first len(u) entries define the colors of the shaft, then the next len(u)*2 entries define
    # the color ofleft and right arrow head side in alternating order. Try for example:
    # colors = ["red" for i in range(len(cf))] + list(chain(*[["blue", "yellow"] for i in range(len(cf))]))
    # to see this effect
    # BUT WAIT THERS MORE: zeor length arrows are apparently filtered out in the matplolib with out filtering the color list appropriately
    # so we have to do this our selfs as well

    # plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d', rasterized=True)

    ax.quiver(xf, yf, zf, vf, uf, wf, colors=colors, **quiv_args)
    plt.colorbar(sm)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.w_xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    return fig
