import os
import matplotlib.pyplot as plt
import numpy as np
from typing import TypedDict

class custom_zoom(TypedDict):
    x: list[float]
    y: list[float]

def scatter_contour_at_z_level(fun, z_level=0, res=100, custom_axis = None,
                               eval_area = (-1,1), scale=(1,1),
                               custom_zoom: custom_zoom={"x": [0.25, 0.75],
                                                         "y": [-0.25, -0.75]},
                                clim = None, flip_axes=False):
    if custom_axis:
        ax = [custom_axis]
        plt_show = False
    elif custom_zoom is not None:
        _, ax = plt.subplots(1, 2)
    else:
        _, ax = plt.subplots(1, 1)
        ax = [ax]

    x = np.linspace(eval_area[0], eval_area[1], num=res)
    y = np.linspace(eval_area[0], eval_area[1], num=res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) + z_level
    sdf = fun(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T).reshape(X.shape)
    if flip_axes:
        tmp = Y
        Y = X
        X = tmp
    
    # cbar = ax[0].scatter(X, Y, c=sdf, cmap="seismic")c
    cbar = ax[0].contourf(X*scale[0], Y*scale[1], sdf, cmap="seismic")
    ax[0].contour(X*scale[0], Y*scale[1], sdf, levels=[0], color="black", linewidths=0.5)
    if clim:
        cbar.set_clim(clim[0], clim[1])
    else:
        cbar.set_clim(-1,1)
    ax[0].set_aspect(1)

    if custom_zoom is not None and not custom_axis:
        x2 = np.linspace(custom_zoom["x"][0], custom_zoom["x"][1], num=res)
        y2 = np.linspace(custom_zoom["y"][0], custom_zoom["y"][1], num=res)
        X2, Y2 = np.meshgrid(x2, y2)
        Z2 = np.zeros_like(X2) + 0
        sdf2 = fun(np.vstack([X2.flatten(), Y2.flatten(), Z2.flatten()]).T).reshape(X2.shape)

        # cbar = ax[1].scatter(X2, Y2, c=sdf2, cmap="seismic")
        cbar = ax[1].contour(X2*scale[0], Y2*scale[0], sdf2, levels=[0], colors="seismic")
        if clim:
            cbar.set_clim(clim[0], clim[1])
        else:
            cbar.set_clim(-1,1)
        ax[1].set_aspect(1)
    if plt_show:
        plt.show()