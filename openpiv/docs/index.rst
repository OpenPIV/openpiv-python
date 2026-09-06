.. OpenPIV documentation master file, created by
   sphinx-quickstart on Mon Apr 18 23:22:32 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenPIV: a python package for PIV image analysis.
=================================================

OpenPIV is a effort of scientists to deliver a tool for the analysis of PIV images
using state-of-the-art algorithms. OpenPIV is released under the 
`GPL Licence <http://en.wikipedia.org/wiki/GNU_General_Public_License>`_,
which means that the source code is freely available for users to study, copy, modify
and improve. Because of its permissive licence, you are welcome to download and try 
OpenPIV for whatever need you may have. Furthermore, you are encouraged to contribute
to OpenPIV, with code, suggestions and critics.

OpenPIV exists in three forms: Matlab, C++ and Python. This is the home page of the **Python** implementation.

High-Performance Dual-Backend Engine
------------------------------------

OpenPIV features an integrated dual-backend architecture:

* **Parallel Rust Backend** (``openpiv_rust``): Multi-core Rayon parallelization delivering up to **580x** faster outlier validation, **19x** faster peak interpolation & SNR estimation, and **4-6x** faster cross-correlation. Pre-compiled wheels are published to PyPI for Linux, macOS, and Windows.
* **Pure Python / SciPy Reference**: Zero-compiler portable fallback ensuring 100% numerical parity across platforms.

=========
Contents:
=========

.. toctree::
   :maxdepth: 2
   :titlesonly:
   

   src/piv_basics
   src/installation_instruction
   src/tutorial1
   src/windef
   src/masking
   src/developers
   src/fft_correlation_backends
   src/api_reference
   src/faq

   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

