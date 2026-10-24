"""Runnable examples shipped with OpenPIV.

The sub-packages here are thin, documented drivers around the library
functionality.  They are importable (so they can be reused and tested) and
also runnable as scripts, e.g.::

    python -m openpiv.examples.stereo --data-root /data/run calibrate
"""

__all__ = ["stereo"]
