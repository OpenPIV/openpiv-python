"""Frame access for the stereo-PIV example.

DaVis camera streams (``.ims``) and images (``.im7``) are read with
`readim7 <https://github.com/alexlib/readim7>`_, which is an *optional*
dependency: it is only imported when a frame is actually requested.

``.ims`` streams are raw, header-less 12-bit packed frame pairs.  Their
geometry normally comes from the small index file that DaVis writes next
to the stream (``Camera<N>-0.ims``).  When that index is not a sibling of
the stream, the sensor size has to be supplied by the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["ims_info", "read_image", "read_pair"]


def _readim7() -> Any:
    try:
        import readim7
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading DaVis files needs the readim7 package: "
            'pip install "readim7 @ git+https://github.com/alexlib/readim7.git"'
        ) from exc
    return readim7


def ims_info(path: str | Path, nx: int | None = None, ny: int | None = None) -> dict:
    """Describe a DaVis camera stream.

    Parameters
    ----------
    path:
        The ``.ims`` stream or the run folder holding it.
    nx, ny:
        Sensor size in pixels.  Only needed when no index file sits next
        to the stream; otherwise the geometry is read from the index.

    Returns
    -------
    dict
        The ``readim7.ims_info`` record.
    """
    readim7 = _readim7()
    if nx is None or ny is None:
        return readim7.ims_info(str(path))
    return readim7.ims_info(str(path), nx=int(nx), ny=int(ny))


def read_pair(
    path: str | Path,
    pair_index: int = 0,
    nx: int | None = None,
    ny: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one dual-frame pair from a DaVis camera stream.

    Parameters
    ----------
    path:
        The ``.ims`` stream.
    pair_index:
        Zero based index of the frame pair.
    nx, ny:
        Sensor size in pixels, see :func:`ims_info`.

    Returns
    -------
    (pulse_a, pulse_b)
        Two ``uint16`` arrays of shape ``(ny, nx)``.
    """
    readim7 = _readim7()
    info = ims_info(path, nx=nx, ny=ny)
    return readim7.read_ims_pair(info, int(pair_index))


def read_image(path: str | Path) -> np.ndarray:
    """Read a DaVis ``.im7`` image (or image sequence) as an array.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` for a single frame, ``(nf, ny, nx)`` for a sequence.
    """
    readim7 = _readim7()
    buffer_, _ = readim7.get_Buffer_andAttributeList(str(path))
    array, _ = readim7.buffer_as_array(buffer_)
    return np.asarray(array)
