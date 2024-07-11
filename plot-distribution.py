import argparse
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import petsc


def add_plot(ax, vecs, xstr: str, ystr: str, show_np: bool=False, **kwargs):
    """Add a 2-D figure from `data` to `ax`.
    
    Notes
    -----
    This function inverts the logical y axis in order to set the origin in the
    bottom left corner.
    """
    x, y = vecs
    ax.scatter(x, y, marker='.')
    if xlim := kwargs.get('xlim'):
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(tck.MaxNLocator(xlim[1]-xlim[0]))
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    if ylim := kwargs.get('ylim'):
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(tck.MaxNLocator(ylim[1]-ylim[0]))
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.set_xlabel(xstr)
    ax.set_ylabel(ystr)
    if show_np:
        # NOTE: This assumes that len(x) == len(y), but I think pyplot.scatter
        # will have already failed if that's true.
        ax.set_title(f"N = {len(x)}")
    ax.label_outer()
    ax.set_aspect('equal')


def create(vec: petsc.Vec, **kwargs):
    """Plot the planes in the given dataset."""
    x = vec.array[:, 0]
    y = vec.array[:, 1]
    xy = (x, y)
    if vec.ndim > 2:
        z = vec.array[:, 2]
        xz = (x, z)
        yz = (y, z)
    else:
        xz = None
        yz = None
    planes = {
        (0, 0): {'vecs': xy, 'xstr': 'x', 'ystr': 'y'},
        (1, 0): {'vecs': xz, 'xstr': 'x', 'ystr': 'z'},
        (1, 1): {'vecs': yz, 'xstr': 'y', 'ystr': 'z'},
    }
    axes = {k: v.copy() for k, v in planes.items() if v['vecs'] is not None}
    nrows = 1 + (len(axes) == 3)
    ncols = 2
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex='col',
        sharey='row',
        squeeze=False,
        figsize=(6, 6),
    )
    axs[0, 1].remove()
    for i, this in axes.items():
        ax = axs[i[0], i[1]]
        add_plot(ax, **this, **kwargs)
    fig.tight_layout()


def main(
    filebase: str,
    source: str=None,
    outdir: str=None,
    show_np: bool=False,
    verbose: bool=False,
    **plotkw
) -> None:
    """Plot 2-D planes of particle-distribution quantities."""
    # NOTE: Copied from `plot_planes.main` => could be refactored.
    srcdir = pathlib.Path(source or '.').expanduser().resolve()
    figdir = (
        pathlib.Path(outdir).expanduser().resolve()
        if outdir else srcdir
    )
    figdir.mkdir(parents=True, exist_ok=True)
    filepaths = list(srcdir.glob(f"{filebase}*.bin"))
    for filepath in filepaths:
        tag = filepath.stem.lstrip(filebase)
        create(petsc.Vec(filepath), show_np=show_np, **plotkw)
        if verbose:
            print(f"Plotting {filebase}")
            figname = f"{filebase.replace(' ', '_')}{tag}.png"
            if verbose:
                print(f"Saving {figname}")
            plt.savefig(figdir / figname)
            plt.close()
        print()
    if verbose:
        print(f"Figure directory: {figdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'filebase',
        help="the base name of the file(s) containing output vectors",
    )
    parser.add_argument(
        '-i',
        '--indir',
        dest="source",
        help=(
            "the directory containing the data to plot"
            "\n(default: current working directory)"
        ),
    )
    parser.add_argument(
        '-o',
        '--outdir',
        help=(
            "the directory to which to save the figure"
            "\n(default: SOURCE)"
        ),
    )
    parser.add_argument(
        '--np',
        dest='show_np',
        help="print the number of particles",
        action='store_true',
    )
    parser.add_argument(
        '--xlim',
        help="the x-axis bounds",
        nargs=2,
        type=float,
        metavar=('LO', 'HI'),
    )
    parser.add_argument(
        '--ylim',
        help="the y-axis bounds",
        nargs=2,
        type=float,
        metavar=('LO', 'HI'),
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help="print runtime messages",
        action='store_true',
    )
    args = parser.parse_args()
    main(**vars(args))


