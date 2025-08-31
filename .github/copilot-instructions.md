# OpenPIV Python

OpenPIV is a Python and Cython library for Particle Image Velocimetry (PIV) analysis of fluid flow images. It provides tools for scripting and executing PIV analysis on image pairs to extract velocity fields from particle-seeded flow visualizations. The library includes both computational algorithms and optional Qt/Tk graphical user interfaces.

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
- **Run tests**: `poetry run pytest openpiv -v` -- takes ~12 seconds, 198 tests pass, 12 skipped. NEVER CANCEL. Set timeout to 30+ minutes for safety.
- **Test import**: `poetry run python -c "import openpiv; print('OpenPIV imported successfully')"`
- **Test core functionality**: `poetry run python -c "import openpiv.piv as piv; import numpy as np; frame_a = np.random.rand(64, 64); frame_b = np.random.rand(64, 64); result = piv.simple_piv(frame_a, frame_b); print('PIV analysis completed, returned:', len(result), 'outputs')"`

### Run Example Workflows
- **Tutorial 1**: `poetry run python openpiv/tutorials/tutorial1.py` -- demonstrates complete PIV analysis workflow
- **Test data location**: `openpiv/data/test1/` contains sample image pairs (`exp1_001_a.bmp`, `exp1_001_b.bmp`)

## Validation

### ALWAYS run these validation steps after making changes:
1. **Import test**: Verify basic import works: `poetry run python -c "import openpiv"`
2. **Core functionality test**: Run simple PIV analysis to ensure algorithms work
3. **Full test suite**: `poetry run pytest openpiv -v` -- NEVER CANCEL, takes ~12 seconds but allow 30+ minutes timeout
4. **Tutorial execution**: Run `poetry run python openpiv/tutorials/tutorial1.py` to test complete workflow

### Critical User Scenarios to Test
After making changes, ALWAYS test these scenarios:
- **Basic PIV Analysis**: Load two images, run PIV analysis, get velocity fields
- **Data Loading**: Import test images from `openpiv/data/test1/`
- **Validation and Filtering**: Apply signal-to-noise filtering and outlier detection
- **File I/O**: Save and load PIV results in vector field format

### CI/CD Validation
- The repository has GitHub Actions workflows in `.github/workflows/`:
  - `testing.yml`: Runs tests on Python 3.10, 3.11, 3.12 with Poetry
  - `build.yml`: Builds and publishes to PyPI on releases
- No linting tools are configured (no black, flake8, etc.)

## Common Tasks

### Repository Structure
```
openpiv/
├── __init__.py          # Main package initialization
├── piv.py              # High-level PIV analysis functions
├── pyprocess.py        # Core PIV processing algorithms
├── pyprocess3D.py      # 3D PIV algorithms
├── tools.py            # Utility functions for I/O and visualization
├── validation.py       # Signal validation and filtering
├── filters.py          # Outlier detection and replacement
├── windef.py           # Window deformation PIV
├── scaling.py          # Coordinate scaling and transformation
├── preprocess.py       # Image preprocessing
├── smoothn.py          # Smoothing algorithms
├── data/               # Sample test data
├── test/               # Comprehensive test suite (210 tests)
├── tutorials/          # Example scripts
└── docs/              # Documentation source
```

### Key APIs and Usage Patterns
- **Simple PIV**: `piv.simple_piv(frame_a, frame_b)` returns `(x, y, u, v, s2n)`
- **Extended search area**: `pyprocess.extended_search_area_piv()` for higher accuracy
- **Window deformation**: `windef` module for advanced PIV with iterative refinement
- **File I/O**: `tools.imread()`, `tools.save()`, `tools.display_vector_field()`
- **Validation**: `validation.sig2noise_val()`, `validation.global_val()`
- **Filtering**: `filters.replace_outliers()` for cleaning velocity fields

### Project Management
- **Dependencies**: Managed via Poetry (`pyproject.toml`) and fallback setuptools (`setup.py`)
- **Package name**: "OpenPIV" (capital letters)
- **Version**: 0.25.3 (defined in both `pyproject.toml` and `setup.py`)
- **Python support**: 3.10, 3.11, 3.12
- **Key dependencies**: numpy, scipy, scikit-image, matplotlib, imageio

### Development Notes
- Uses `importlib_resources` for accessing package data files
- Test configurations in `openpiv/test/conftest.py` disable plotting for CI
- Sample data includes real PIV image pairs for testing workflows
- Documentation built with Sphinx (source in `openpiv/docs/`)
- External examples repository: [OpenPIV-Python-Examples](https://github.com/OpenPIV/openpiv-python-examples)

### Common Command Reference
```bash
# Development setup
poetry install                                          # ~10 seconds
poetry run pytest openpiv -v                          # ~12 seconds, 198 tests pass

# Testing functionality  
poetry run python openpiv/tutorials/tutorial1.py      # Complete PIV workflow
poetry run python -c "import openpiv.piv as piv; ..."  # API test

# Alternative installs
pip install openpiv                                    # ~33 seconds
conda install -c conda-forge openpiv                  # ~46 seconds

# Build from source (minimal - no Cython compilation needed)
python setup.py build_ext --inplace                   # <1 second
```

### Timing Expectations and Timeouts
- **Poetry install**: ~10 seconds (set 5+ minute timeout)
- **Test suite**: ~12 seconds (set 30+ minute timeout for safety)
- **Tutorial execution**: ~1-2 seconds  
- **Pip install**: ~33 seconds (set 10+ minute timeout)
- **Conda install**: ~46 seconds (set 15+ minute timeout)
- **Build from source**: <1 second (no Cython compilation currently)

**CRITICAL: NEVER CANCEL long-running commands. PIV analysis can be computationally intensive and build systems may take longer than expected.**