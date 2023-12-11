import argparse
import pathlib
import typing

import h5py
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy
from scipy.ndimage import zoom

# TODO:
# - Merge this with write_hdf5.py
# - Add optional input file to CLI
# - Use '-n0' to shift density when reading from a file

def main(
    infile: str,
    outfile: str,
    factor: typing.Optional[typing.Union[float, typing.Tuple[float]]]=None,
    verbose: bool=False,
) -> None:
    """Extract the density array from HDF5 EPPIC output."""
    inpath = pathlib.Path(infile).resolve().expanduser()
    if verbose:
        print(f"Reading from {inpath} ...")
    h5 = h5py.File(inpath)
    inden = 1.0 + numpy.transpose(h5['den1'][:])
    if factor:
        if verbose:
            print(f"Scaling dimensions by {factor} ...")
        outden = zoom(inden, factor)
    else:
        outden = inden
    outpath = pathlib.Path(outfile).resolve().expanduser()
    if verbose:
        print(f"Writing to {outpath} ...")
    with h5py.File(outpath, 'w') as f:
        f.create_dataset('density-ijk', data=outden)
        f.create_dataset('density-kji', data=numpy.transpose(outden))
    plotpath = outpath.with_suffix('.png')
    create_figure(outden)
    plt.savefig(plotpath)
    if verbose:
        print(f"Saved {plotpath}\n")
    plt.close()


def create_figure(array: numpy.ndarray):
    """Plot 2D planes from the given array."""
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharex='col',
        sharey='row',
        squeeze=False,
        figsize=(6, 6),
    )
    axs[0, 1].remove()
    nx, ny, nz = array.shape
    axes = {
        ('x', 'y'): {'axis': axs[0, 0], 'data': array[:, :, nz//2]},
        ('x', 'z'): {'axis': axs[1, 0], 'data': array[:, ny//2, :]},
        ('y', 'z'): {'axis': axs[1, 1], 'data': array[nx//2, :, :]},
    }
    for (xstr, ystr), current in axes.items():
        add_plot(current['axis'], current['data'], xstr, ystr)
    fig.tight_layout()


def add_plot(ax: matplotlib.axes.Axes, data, xstr: str, ystr: str):
    """Add a 2-D figure from `data` to `ax`.
    
    Notes
    -----
    This function inverts the logical y axis in order to set the origin in the
    bottom left corner.
    """
    ax.pcolormesh(numpy.transpose(data))
    ax.set_xlabel(xstr)
    ax.set_ylabel(ystr)
    ax.label_outer()
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(tck.MaxNLocator(5))
    ax.yaxis.set_major_locator(tck.MaxNLocator(5))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        'infile',
        help="path to HDF5 file containing the target density array",
    )
    p.add_argument(
        'outfile',
        help="path to HDF5 file that will contain the result",
    )
    p.add_argument(
        '--scale',
        dest='factor',
        help="numeric scaling factor; see scipy.ndimage.zoom",
        type=float,
    )
    p.add_argument(
        '-v',
        '--verbose',
        help="print runtime messages",
        action='store_true',
    )
    args = p.parse_args()
    main(**vars(args))
