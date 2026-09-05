FFT Correlation Backends & Performance Study
============================================

This document details the comparative study of 2D cross-correlation algorithms,
backends, and performance characteristics in OpenPIV, focusing on both **Circular**
and **Linear** correlation, `scipy.fft` (PocketFFT), and the `openpiv_rust` compiled backend.

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
--------

In Particle Image Velocimetry (PIV), cross-correlation between interrogation windows
is the core computational step. For an image pair split into :math:`N` interrogation
windows of size :math:`(H \times W)`, the correlation maps are computed using the
Wiener-Khinchin theorem via Fast Fourier Transforms:

.. math::

   C = \mathcal{F}^{-1} \left( \mathcal{F}(I_B) \cdot \mathcal{F}^*(I_A) \right)

OpenPIV supports two primary correlation paradigms:

1. **Circular Correlation (Standard OpenPIV)**:
   Assumes periodic boundary conditions (toroidal wraparound).
   Inputs of size :math:`(N \times N)` yield correlation maps of size :math:`(N \times N)`.
   No zero-padding is required.

2. **Linear Correlation (Full / Extended)**:
   Evaluates true spatial shift without periodic wraparound.
   For windows of size :math:`s_1` and :math:`s_2`, the full linear correlation has size
   :math:`(s_1 + s_2 - 1)`. To prevent time-domain aliasing, inputs must be zero-padded
   to at least this size before applying the FFT.


The 63x63 vs 64x64 Power-of-2 Finding
-------------------------------------

In previous versions of OpenPIV, linear correlation in ``openpiv.pyprocess.fft_correlate_images``
computed the padded transform size as:

.. code-block:: python

   size = s1 + s2 - 1                                   # e.g., 32 + 32 - 1 = 63
   fsize = 2 ** np.ceil(np.log2(size)).astype(int) - 1  # 64 - 1 = 63 (BUG!)

Why 63 Disabled PocketFFT Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SciPy delegates FFT operations to **PocketFFT** (written in C++ by Martin Reinecke),
which includes hand-tuned AVX2/FMA vector kernels. However, these SIMD routines rely on
smooth radix factors (powers of 2: 2, 4, 8, 16, 32, 64).

1. Subtracting 1 produced an **odd composite size** :math:`63 = 3 \times 3 \times 7`.
2. This forced PocketFFT into slow Bluestein/composite transforms, disabling SIMD vectorization.
3. Furthermore, padding to 63 left an off-by-one boundary alignment when slicing the central window.

The Power-of-2 Solution
^^^^^^^^^^^^^^^^^^^^^^^

By keeping the clean power-of-2 transform size :math:`fsize = 2^{\lceil \log_2(size) \rceil}`
(e.g., :math:`64 \times 64` for :math:`32 \times 32` windows), PocketFFT achieves optimal
SIMD execution speed.

To recover the exact central correlation window matching ``scipy.signal.correlate``
to machine precision (:math:`\le 10^{-13}`), the centered slice is:

.. code-block:: python

   # Exact power of 2 transform size
   fsize = 2 ** np.ceil(np.log2(size)).astype(int)

   # Centered slice extracting (s1) around the zero-displacement lag
   fslice = (
       slice(0, image_a.shape[0]),
       slice(fsize[0] // 2 - s1[0] // 2, fsize[0] // 2 - s1[0] // 2 + s1[0]),
       slice(fsize[1] // 2 - s1[1] // 2, fsize[1] // 2 - s1[1] // 2 + s1[1]),
   )
   f2a = conj(rfft2(image_a, fsize, axes=(-2, -1), workers=workers))
   f2b = rfft2(image_b, fsize, axes=(-2, -1), workers=workers)
   corr = fftshift(irfft2(f2a * f2b, axes=(-2, -1)).real, axes=(-2, -1))[fslice]


Batched Multi-Threading in scipy.fft
------------------------------------

``scipy.fft`` supports a ``workers`` parameter (e.g., ``workers=-1`` for all logical cores).
However, profiling reveals:

* On large 2D or 3D volumes, PocketFFT multi-threading scales well.
* On **batches of tiny 2D windows** (:math:`16 \times 16` or :math:`32 \times 32`),
  thread synchronization overhead inside PocketFFT often negates multi-core gains
  (scaling from :math:`1.0\times` to :math:`1.1\times`), because thread splitting occurs
  per-transform rather than distributing independent windows across workers.


The Rust Acceleration Engine (openpiv_rust)
-------------------------------------------

To overcome Python GIL overhead and achieve linear CPU scaling across window batches,
OpenPIV provides an optional compiled Rust extension (``openpiv_rust``) built with
PyO3, Rayon, and RealFFT.

Key Architectural Optimizations in Rust
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Zero-Allocation Rayon Scratch Pool**:
   Window transforms use ``par_chunks_exact_mut().for_each_init(...)``, pre-allocating
   frequency buffers once per worker thread. For 961 windows, this eliminates over
   8,600 heap allocations per pass.

2. **Native Power-of-2 Real FFT**:
   Uses ``realfft`` (Real-to-Complex forward and Complex-to-Real inverse), cutting
   arithmetic and memory bandwidth by 50% compared to full complex transforms.

3. **Batched Subpixel Peak Finding**:
   In addition to cross-correlation, `openpiv_rust` provides
   ``batch_correlation_to_displacement``, executing Gaussian/Centroid/Parabolic subpixel
   peak fitting directly across all windows in parallel, yielding a **28x to 65x speedup**
   over the nested Python loop.


Benchmark Results
-----------------

Linear Correlation (225 windows of 32x32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============================================  ====================  ==================
Implementation                                Runtime               Relative Speed
============================================  ====================  ==================
``scipy.signal.correlate`` (Python loop)      70.63 ms              1.0x (baseline)
OpenPIV SciPy (legacy ``fsize=63``)           49.28 ms              1.4x
OpenPIV SciPy (power-of-2 ``fsize=64``)       39.10 ms              1.8x
**openpiv_rust** (Rayon + realfft)            **8.26 ms**           **8.5x**
============================================  ====================  ==================

Displacement Calculation (Peak Finding)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============================================  ====================  ==================
Grid / Windows                                Python Loop           openpiv_rust
============================================  ====================  ==================
225 windows (32x32)                           5.19 ms               **0.18 ms (28x)**
961 windows (16x16)                           19.84 ms              **0.30 ms (65x)**
============================================  ====================  ==================

End-to-End Multi-Pass Windef (3 passes on 256x256 image)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **SciPy backend**: 110.59 ms
* **Rust backend (with subpixel Rust engine)**: **81.66 ms** (:math:`1.35\times` speedup)

Conclusion & Usage Guidelines
-----------------------------

* For systems with compiled binary extensions available, set ``settings.backend = 'rust'``
  to maximize throughput in both single-pass and multi-pass deformation workflows.
* On pure Python / NumPy / SciPy environments, using the exact power-of-2 padding rule
  ensures optimal PocketFFT vectorization while maintaining exact numerical consistency.
