"""
Support for working with PETSc objects.
"""

import collections.abc
import pathlib
import sys
import typing

import numpy
import matplotlib.colors as clr

PETSC_PATH = pathlib.Path('~/petsc/lib/petsc/bin').expanduser()
sys.path.append(str(PETSC_PATH))

import PetscBinaryIO


class Info(collections.abc.Mapping):
    """A mapping view of binary-file metadata."""

    def __init__(self, filepath: pathlib.Path) -> None:
        super().__init__()


class Vec:
    """A custom representation of a PETSc Vec object."""

    def __init__(self, filepath: pathlib.Path) -> None:
        io = PetscBinaryIO.PetscBinaryIO()
        fh = open(filepath)
        io.readObjectType(fh)
        self._data = io.readVec(fh)
        with filepath.with_suffix('.bin.info').open() as fp:
            line = fp.readline()
            self._ndim = int(line.split()[1])
        self._array = None

    @property
    def array(self):
        """The equivalent numpy array."""
        if self._array is None:
            array = numpy.array(self._data)
            if self._ndim is None:
                self._array = array
            else:
                np = array.size // self._ndim
                self._array = array.reshape(np, self._ndim)
        return self._array

    @property
    def ndim(self):
        """The number of axes represented by the data."""
        return self._ndim


class Mat:
    """A custom representation of a PETSc Mat object."""

    def __init__(self, filepath: pathlib.Path) -> None:
        io = PetscBinaryIO.PetscBinaryIO()
        fh = open(filepath)
        io.readObjectType(fh)
        self._data = io.readMatDense(fh)
        self._array = None
        self._normalized = None
        self._nr = None
        self._nc = None

    def mask(self, value: float, normalized: bool=False):
        """Mask the data array at `value`."""
        array = self.normalized if normalized else self.array
        return numpy.ma.masked_values(array, value)

    @property
    def nr(self):
        """The number of rows in the matrix."""
        if self._nr is None:
            self._nr = self.array.shape[0]
        return self._nr

    @property
    def nc(self):
        """The number of columns in the matrix."""
        if self._nc is None:
            self._nc = self.array.shape[1]
        return self._nc

    @property
    def array(self):
        """The corresponding numpy array."""
        if self._array is None:
            self._array = numpy.array(self._data)
        return self._array

    @property
    def normalized(self):
        """Alias for `self.normalize(vcenter=0.0)`."""
        if self._normalized is None:
            self._normalized = self.normalize(vcenter=0.0)
        return self._normalized

    @typing.overload
    def normalize(
        self,
        vmin: float=None,
        vcenter: float=None,
        vmax: float=None,
    ) -> numpy.ndarray: ...

    @typing.overload
    def normalize(
        self,
        vmin: float=None,
        vmax: float=None,
        clip: bool=False,
    ) -> numpy.ndarray: ...

    def normalize(self, **kwargs) -> numpy.ndarray:
        """Normalized the data array."""
        normalize = (
            clr.TwoSlopeNorm(**kwargs) if 'vcenter' in kwargs
            else clr.Normalize(*kwargs)
        )
        return normalize(self._array)


