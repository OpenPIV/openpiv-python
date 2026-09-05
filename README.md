# OpenPIV
[![Python package](https://github.com/OpenPIV/openpiv-python/actions/workflows/testing.yml/badge.svg)](https://github.com/OpenPIV/openpiv-python/actions/workflows/testing.yml)
[![Wheels](https://github.com/OpenPIV/openpiv-python/actions/workflows/wheels.yml/badge.svg)](https://github.com/OpenPIV/openpiv-python/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/openpiv.svg)](https://pypi.org/project/openpiv/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.593157.svg)](https://doi.org/10.5281/zenodo.593157)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


OpenPIV consists in a Python and Cython modules for scripting and executing the analysis of 
a set of PIV image pairs. In addition, a Qt and Tk graphical user interfaces are in 
development, to ease the use for those users who don't have python skills.

## Warning

The OpenPIV python version is still in its *beta* state. This means that
it still might have some bugs and the API may change. However, testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.


## Test it without installation
Click the link - thanks to BinderHub, Jupyter and Conda you can now get it in your browser with zero installation:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openpiv/openpiv-python/master?filepath=openpiv%2Fexamples%2Fnotebooks%2Ftutorial1.ipynb)




## Installing

### Recommended: Using uv (fastest)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust:

    pip install uv
    uv pip install openpiv

### Using pip (standard)

Use PyPI: <https://pypi.python.org/pypi/OpenPIV>:

    pip install openpiv

### Or [Poetry](https://python-poetry.org/)

    poetry add openpiv

### Note on Conda/Anaconda

⚠️ **Conda packages are no longer actively maintained.** The conda-forge package may be outdated.

If you previously installed OpenPIV via conda, you can migrate to pip or uv:

    # Remove the conda package
    conda remove openpiv
    
    # Install with pip or uv
    pip install openpiv
    # or
    uv pip install openpiv
    
    
### To build from source

Clone using git:

    git clone https://github.com/OpenPIV/openpiv-python.git
    cd openpiv-python

To build the Rust acceleration extension locally (requires Rust toolchain and maturin):

    pip install maturin
    maturin develop --release -m crates/openpiv_rust/Cargo.toml
    pip install -e .


## High-Performance Dual-Backend Acceleration (Rust & SciPy)

OpenPIV features a parallel dual-backend architecture designed for high throughput without sacrificing numerical reproducibility:

* **⚡ Parallel Rust Backend (`openpiv_rust`)**: Multithreaded execution across all CPU cores using Rayon and real-to-complex FFTW/RustFFT routines. Delivers up to **580x** faster outlier validation, **19x** faster subpixel peak interpolation and SNR calculation, **4-6x** faster windowing and FFT cross-correlation, and **3.7x** faster end-to-end PIV pipelines.
* **🐍 Pure Python / SciPy Backend**: Complete, zero-dependency reference implementation that runs everywhere without a compiler.

When you install OpenPIV from PyPI via `pip` or `uv`, pre-compiled binary wheels with the Rust acceleration backend are installed automatically.

### Explicit Backend Control

All core processing functions accept a `backend` parameter:

| `backend` Option | Behavior |
| :--- | :--- |
| `"auto"` *(default)* | Automatically selects the parallel Rust backend if available; cleanly and transparently falls back to pure Python/SciPy if not. |
| `"rust"` | Enforces the parallel Rust backend. Raises an informative `ImportError` if the Rust extension is not compiled. |
| `"scipy"` (or `"python"`) | Enforces the pure Python/SciPy reference path. |

Both backends produce identical numerical results (`diff = 0.0`).

---

### Code Examples

#### 1. Quick Analysis with `simple_piv`

```python
from openpiv import piv

# Default: auto-selects fast Rust backend with fallback
x, y, u, v, s2n = piv.simple_piv("exp1_001_a.bmp", "exp1_001_b.bmp", backend="auto")

# Explicitly force the parallel Rust backend
x, y, u, v, s2n = piv.simple_piv("exp1_001_a.bmp", "exp1_001_b.bmp", backend="rust")

# Explicitly force the SciPy reference backend
x, y, u, v, s2n = piv.simple_piv("exp1_001_a.bmp", "exp1_001_b.bmp", backend="scipy")
```

#### 2. Standard PIV with `extended_search_area_piv`

```python
from openpiv import pyprocess, tools

frame_a = tools.imread("exp1_001_a.bmp")
frame_b = tools.imread("exp1_001_b.bmp")

# Run with parallel Rust acceleration
u, v, s2n = pyprocess.extended_search_area_piv(
    frame_a,
    frame_b,
    window_size=32,
    overlap=16,
    search_area_size=32,
    correlation_method="circular",
    backend="rust",  # 'auto', 'rust', or 'scipy'
)
```

#### 3. Multi-Pass Window Deformation (`windef`)

```python
from openpiv import windef, tools

settings = windef.PIVSettings()
settings.windowsizes = (64, 32, 16)
settings.overlap = (32, 16, 8)
settings.num_iterations = 3

# Choose backend in PIVSettings: 'auto', 'rust', or 'scipy'
settings.backend = "auto"

frame_a = tools.imread("exp1_001_a.bmp")
frame_b = tools.imread("exp1_001_b.bmp")

# Executes all deformation passes with the chosen backend
x, y, u, v, mask = windef.simple_multipass(frame_a, frame_b, settings)
```

---

## Documentation

The OpenPIV documentation is available on the project web page at <http://openpiv.readthedocs.org>

## Demo notebooks 

1. [Tutorial Notebook 1](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/tutorial1.ipynb)
2. [Tutorial notebook 2](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/tutorial2.ipynb)
3. [Dynamic masking tutorial](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/masking_tutorial.ipynb)
4. [Multipass with Windows Deformation](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/window_deformation_comparison.ipynb)
5. [Multiple sets in one notebook](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/all_test_cases_sample.ipynb)
6. [3D PIV](https://nbviewer.org/github/OpenPIV/openpiv-python-examples/blob/main/notebooks/PIV_3D_example.ipynb)


These and many additional examples are in another repository: [OpenPIV-Python-Examples](https://github.com/OpenPIV/openpiv-python-examples)


## Contributors

1. [Alex Liberzon](http://github.com/alexlib)
2. [Roi Gurka](http://github.com/roigurka)
3. [Zachary J. Taylor](http://github.com/zjtaylor)
4. [David Lasagna](http://github.com/gasagna)
5. [Mathias Aubert](http://github.com/MathiasAubert)
6. [Pete Bachant](http://github.com/petebachant)
7. [Cameron Dallas](http://github.com/CameronDallas5000)
8. [Cecyl Curry](http://github.com/leycec)
9. [Theo Käufer](http://github.com/TKaeufer)
10. [Andreas Bauer](https://github.com/AndreasBauerGit)
11. [David Bohringer](https://github.com/davidbhr)
12. [Erich Zimmer](https://github.com/ErichZimmer)
13. [Peter Vennemann](https://github.com/eguvep)
14. [Lento Manickathan](https://github.com/lento234)
15. [Yuri Ishizawa](https://github.com/yuriishizawa)


Copyright statement: `smoothn.py` is a Python version of `smoothn.m` originally created by D. Garcia [https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn], written by Prof. Lewis and available on Github [https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py]. We include a version of it in the `openpiv` folder for convenience and preservation. We are thankful to the original authors for releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the authors regarding their license. 

## How to cite this work
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.593157.svg)](https://doi.org/10.5281/zenodo.593157)

If you use OpenPIV in your scientific research, please cite the persistent software archive on Zenodo:

```bibtex
@software{openpiv_python,
  author       = {Liberzon, Alex and K{\"a}ufer, Theo and Bauer, Andreas and Vennemann, Peter and Zimmer, Erich and contributors},
  title        = {OpenPIV: Python and Rust Acceleration for Particle Image Velocimetry},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v0.26.1},
  doi          = {10.5281/zenodo.593157},
  url          = {https://doi.org/10.5281/zenodo.593157}
}
```




