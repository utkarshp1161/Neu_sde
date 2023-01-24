""" Visualization utilities to visualize drift and diffuion fields. """

import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.ticker import ScalarFormatter
import numpy as np

from matplotlib import font_manager

locations = ['/home/ece/utkarsh/Helvetica'] 

font_files = font_manager.findSystemFonts(fontpaths = locations)
for file in font_files:
    font_manager.fontManager.addfont(file)

CMAP = 'inferno'
plt.rcParams.update(
    {
        'font.family': 'sans-serif',
        'font.sans-serif': 'Helvetica',
        'font.size': 32,
    }
)


def plot_drift_field(ax, fx, fy):
    """
    Visualize the drift field as a vector field.

    Args:
        ax: Axis on to which to plot to.
        fx: The (callable) drift function, that map mx, my -> f(mx, my).
    """

    #     r, theta = np.meshgrid(np.linspace(0, 1, 1), np.linspace(0, 2 * np.pi, 90))
    #     x, y = r * np.cos(theta), r * np.sin(theta)
    #x_, y_ = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21)) 
    x_, y_ = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15)) 
    x = x_[x_ ** 2 + y_ ** 2 <= 1]
    y = y_[x_ ** 2 + y_ ** 2 <= 1]
    qv = ax.quiver(x, y, fx(x, y), fy(x, y),
                   np.sqrt(fx(x, y) ** 2 + fy(x, y) ** 2),
                   width=0.008, cmap='inferno')
    ax.set(xlabel='$m_x$', ylabel='$m_y$', title='Drift Field')
    ax.set(xticks=[-1, -.5, 0, .5, 1], yticks=[-1, -.5, 0, .5, 1])
    ax.set_aspect('equal', 'box')

    plt.colorbar(qv, ax=ax, fraction=0.0453)

def plot_diffusion_field(ax, gxx, gyy, gxy, scale=1):
    """
    Visualize the diffusion field

    Args:
        ax: Axis on to which to plot to
        gxx, gyy: Diffusion functions
        gxy: Cross diffusion
        scale: Scale factor to scale the ellipses with
    """

    x_, y_ = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
    xs = x_[x_ ** 2 + y_ ** 2 <= 1]
    ys = y_[x_ ** 2 + y_ ** 2 <= 1]
    xy = np.column_stack((xs.ravel(), ys.ravel()))
    maj_axis = np.empty(xs.size)
    min_axis = np.empty(xs.size)
    angles = np.empty(xs.size)
    colors = np.empty(xs.size)
    for i, (x, y) in enumerate(zip(xs.ravel(), ys.ravel())):
        diff = [[gxx(x, y), gxy(x, y)],
                [gxy(x, y), gyy(x, y)]]

        # Eigendecomposition is computed using the lower triangular part, assuming symmetry
        eigval, eigvec = np.linalg.eigh(diff, UPLO='L')
        maj_axis[i] = scale * eigval[0]
        min_axis[i] = scale * eigval[1]
        angles[i] = np.arctan2(eigvec[0, 1], eigvec[0, 0]) * 180 / np.pi
        colors[i] = np.linalg.det(diff)
    
    ec = EllipseCollection(maj_axis, min_axis, angles,
                           offsets=xy, offset_transform=ax.transData,
                           linewidths=1.5,
                           # edgecolors=,
                           # edgecolors=plt.cm.inferno(plt.Normalize()(colors)),
                           facecolors=(0, 0, 0, 0),
                           cmap='inferno')
    ec.set_array(colors)
    ax.add_collection(ec)
    ax.autoscale_view()
    # ax.scatter(xs.ravel(), ys.ravel(), marker='+', color=(0.8, 0.8, 0.8))
    ax.set(xlabel='$m_x$', ylabel='$m_y$', title='Diffusion Field')
    ax.set(xticks=[-1, -.5, 0, .5, 1], yticks=[-1, -.5, 0, .5, 1])
    ax.set_aspect('equal', 'box')

    plt.colorbar(ec, ax=ax, fraction=0.0453)

if __name__ == "__main__":
    print ("Executed when invoked directly")
else:
    print ("Executed when imported")
