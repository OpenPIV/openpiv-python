# OpenPIV Python

OpenPIV is a Python library for Particle Image Velocimetry (PIV) analysis of fluid flow images. It provides tools for scripting and executing PIV analysis on image pairs to extract velocity fields from particle-seeded flow visualizations. The library includes both computational algorithms and optional Qt/Tk graphical user interfaces.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Install Dependencies
- **Primary method (recommended)**: Use Poetry for development:
  - Install Poetry: `pip install poetry`
  - Install dependencies: `poetry install` -- takes ~10 seconds. NEVER CANCEL.
  - All development commands should use `poetry run <command>`

### Alternative Installation Methods
- **From PyPI**: `pip install openpiv` -- takes ~33 seconds. NEVER CANCEL.
- **From conda-forge**: `conda install -c conda-forge openpiv` -- takes ~46 seconds. NEVER CANCEL.
- **Build from source**: `python setup.py build_ext --inplace` -- takes <1 second (no Cython extensions in current setup)

### Build and Test the Repository
- **Run tests**: `poetry run pytest openpiv -v` -- takes ~10 seconds, 216 tests pass. NEVER CANCEL. Set timeout to 30+ minutes for safety.
- **Test import**: `poetry run python -c "import openpiv; print('OpenPIV imported successfully')"`
- **Test core functionality**: `poetry run python -c "import openpiv.piv as piv; import numpy as np; frame_a = np.random.rand(64, 64); frame_b = np.random.rand(64, 64); result = piv.simple_piv(frame_a, frame_b, plot=False); print('PIV analysis completed, returned:', len(result), 'outputs')"`

### Run Example Workflows
- **Tutorial 1**: `poetry run python openpiv/tutorials/tutorial1.py` -- demonstrates complete PIV analysis workflow
- **Test data location**: `openpiv/data/test1/` contains sample image pairs (`exp1_001_a.bmp`, `exp1_001_b.bmp`)

## Validation

### ALWAYS run these validation steps after making changes:
1. **Import test**: Verify basic import works: `poetry run python -c "import openpiv"`
2. **Core functionality test**: Run simple PIV analysis to ensure algorithms work
3. **Full test suite**: `poetry run pytest openpiv -v` -- NEVER CANCEL, takes ~10 seconds but allow 30+ minutes timeout
4. **Tutorial execution**: Run `poetry run python openpiv/tutorials/tutorial1.py` to test complete workflow

### Critical User Scenarios to Test
After making changes, ALWAYS test these scenarios:
- **Basic PIV Analysis**: Load two images, run PIV analysis, get velocity fields
- **Data Loading**: Import test images from `openpiv/data/test1/`
- **Validation and Filtering**: Apply signal-to-noise filtering and outlier detection
- **File I/O**: Save and load PIV results in vector field format

### CI/CD Validation
- The repository has GitHub Actions workflows in `.github/workflows/`:
  - `testing.yml`: Runs tests on Python 3.10, 3.11, 3.12, 3.13, 3.14 with Poetry
  - `build.yml`: Builds and publishes to PyPI on releases triggered by version tags
- No linting tools are configured (no black, flake8, etc.)

## Common Tasks

### Repository Structure
```
openpiv/
├── __init__.py          # Package init; exposes __version__ via importlib.metadata
├── piv.py               # High-level PIV workflows: simple_piv(), piv_example(), process_pair()
├── pyprocess.py         # Core cross-correlation algorithms: extended_search_area_piv(), get_coordinates()
├── pyprocess3D.py       # 3D PIV algorithms
├── tools.py             # I/O and visualization: imread(), save(), display_vector_field(), transform_coordinates()
├── validation.py        # Spurious vector detection: global_val(), global_std(), sig2noise_val()
├── filters.py           # Outlier replacement: replace_outliers() (calls lib.replace_nans internally)
├── lib.py               # Low-level NaN inpainting: replace_nans() (used by filters)
├── windef.py            # Window-deformation iterative PIV: multipass_img_deform(), piv()
├── settings.py          # PIVSettings dataclass -- required by windef batch processing
├── scaling.py           # Coordinate scaling and transformation
├── preprocess.py        # Image preprocessing (background subtraction, masking)
├── smoothn.py           # Smoothing algorithms (robust spline smoothing)
├── phase_separation.py  # Solid-phase / liquid tracer separation utilities
├── PIV_3D_plotting.py   # 3D vector field plotting helpers
├── data/                # Bundled sample PIV image pairs (test1/)
├── test/                # Test suite (~216 tests); conftest.py disables matplotlib GUI
├── tutorials/           # Example scripts (tutorial1.py, tutorial2.py, windef_tutorial.py, masking_tutorial.py)
└── docs/                # Sphinx documentation source
```

### Key APIs and Usage Patterns
- **Simple PIV** (quickest path): `piv.simple_piv(frame_a, frame_b, plot=False)` → `(x, y, u, v, s2n)`
- **Complete workflow**: `piv.process_pair(frame_a, frame_b)` → `(x, y, u, v, mask)`
- **Core cross-correlation**: `pyprocess.extended_search_area_piv(im1, im2, window_size, overlap, search_area_size)` → `(u, v, s2n)`
- **Coordinates**: `pyprocess.get_coordinates(image_size, search_area_size, overlap)` → `(x, y)`
- **Window deformation (batch)**: `windef.piv(settings)` where `settings` is a `PIVSettings` instance
- **File I/O**: `tools.imread(path)`, `tools.save(x, y, u, v, mask, filename)`, `tools.display_vector_field(filename)`
- **Validation**: `validation.global_val(u, v, u_thresholds, v_thresholds)`, `validation.sig2noise_val(s2n, threshold)`
- **Filtering**: `filters.replace_outliers(u, v, mask, method='localmean', kernel_size=1)`
- **Coordinate transform**: `tools.transform_coordinates(x, y, u, v)` — always call before saving/displaying

### PIVSettings (for windef batch processing)
`settings.PIVSettings` is a dataclass with defaults that point to the bundled test data. Key fields:
```python
from openpiv.settings import PIVSettings
s = PIVSettings()
s.filepath_images = pathlib.Path('path/to/images')
s.frame_pattern_a = 'frame_a_*.tif'
s.frame_pattern_b = 'frame_b_*.tif'
s.windowsizes = (64, 32, 16)   # multi-pass window sizes
s.overlap = (32, 16, 8)        # corresponding overlaps
s.num_iterations = 3
```

### Project Management
- **Dependencies**: Managed via Poetry (`pyproject.toml`); fallback setuptools config in `setup.py`
- **Package name**: `"OpenPIV"` (capital letters on PyPI, lowercase `openpiv` as import)
- **Version**: `0.25.4` (defined in both `pyproject.toml` and `setup.py`; runtime via `importlib.metadata`)
- **Python support**: 3.10, 3.11, 3.12, 3.13, 3.14
- **Key dependencies**: numpy ≥2.0, scipy ≥1.11, scikit-image ≥0.23, matplotlib ≥3.8, imageio ≥2.35, natsort, tqdm

### Development Notes
- **No Cython extensions**: Although the package description still mentions "Cython modules" (legacy), all Cython (`.pyx`) files have been removed and converted to pure Python. `python setup.py build_ext --inplace` takes <1 second with nothing to compile.
- Uses stdlib `importlib.resources.files()` (NOT the third-party `importlib_resources`) to locate bundled data
- Test configurations in `openpiv/test/conftest.py` disable matplotlib GUI (uses `Agg` backend, patches `plt.show`)
- Sample data bundled at `openpiv/data/test1/`; accessed via `files('openpiv.data').joinpath('test1/...')`
- `pyproject.toml` still uses the deprecated `[tool.poetry.dev-dependencies]` section; this generates a warning but is harmless
- Documentation built with Sphinx (source in `openpiv/docs/`)
- External examples repository: [OpenPIV-Python-Examples](https://github.com/OpenPIV/openpiv-python-examples)

### Common Command Reference
```bash
# Development setup
poetry install                                              # ~10 seconds
poetry run pytest openpiv -v                               # ~10 seconds, 216 tests pass

# Testing functionality
poetry run python openpiv/tutorials/tutorial1.py           # Complete PIV workflow
poetry run python -c "import openpiv.piv as piv; ..."      # API test

# Alternative installs
pip install openpiv                                         # ~33 seconds
conda install -c conda-forge openpiv                       # ~46 seconds

# Build from source (minimal - no Cython compilation needed)
python setup.py build_ext --inplace                        # <1 second
```

### Timing Expectations and Timeouts
- **Poetry install**: ~10 seconds (set 5+ minute timeout)
- **Test suite**: ~10 seconds (set 30+ minute timeout for safety)
- **Tutorial execution**: ~1-2 seconds
- **Pip install**: ~33 seconds (set 10+ minute timeout)
- **Conda install**: ~46 seconds (set 15+ minute timeout)
- **Build from source**: <1 second (no Cython compilation currently)

**CRITICAL: NEVER CANCEL long-running commands. PIV analysis can be computationally intensive and build systems may take longer than expected.**