"""
=======================
Polynomial Camera Model
=======================

This module contains an implementation of a polynomial camera model. This
model is implemented using 3rd order polynomials in the x and y axis and a
2nd order polynomial along the z-axis. This model can handle a multiplitude
of different distortions and is usually preferred if processing algorithms
later on do not heavily utilize triangulation. Additionally, it is important
that the calibration markers cover as much of the image(s) as possible to
limit artifacts from extrapolation.

Functions
=========
    camera - Create an instance of a Soloff camera model
    multi_line_intersect - Using multiple lines, approximate their intersection

Note
====
It is important to only import the submodule and not the functions that
are in the submodules. Explicitly importing a function from this submodule
could cause conflicts between other camera models due to similar naming
conventions that are normally protected behind namespaces.

"""
from ._camera import *
from ._epipolar_geom import *


__all__ = [s for s in dir() if not s.startswith("_")]