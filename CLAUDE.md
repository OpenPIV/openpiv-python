# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

OpenPIV is a Python library for Particle Image Velocimetry (PIV) analysis of fluid flow images: it extracts velocity fields from pairs of particle-seeded flow images via cross-correlation.

**Always reference these instructions first and fall back to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Install Dependencies
- Use Poetry: `pip install poetry` then `poetry install` (~10 seconds). All development commands should use `poetry run <command>`.

### Build and Test
- Run tests: `poetry run pytest openpiv -v` — ~10 seconds, 216 tests pass.
- Run a single test file: `poetry run pytest openpiv/test/test_process.py -v`
- Import test: `poetry run python -c "import openpiv; print('OpenPIV imported successfully')"`
- No linting tools are configured (no black, flake8, etc.).
- `python setup.py build_ext --inplace` is a legacy no-op: all Cython (`.pyx`) files have been removed and converted to pure Python, despite the package description still mentioning "Cython modules".

### Run Example Workflows
- `poetry run python openpiv/tutorials/tutorial1.py` — demonstrates a complete PIV analysis workflow.
- Sample test data lives at `openpiv/data/test1/` (`exp1_001_a.bmp`, `exp1_001_b.bmp`), accessed via `importlib.resources.files('openpiv.data').joinpath('test1/...')` (stdlib, not the third-party `importlib_resources`).

## Architecture

### Module responsibilities
- `piv.py` — high-level entry points: `simple_piv()`, `piv_example()`, `process_pair()`.
- `pyprocess.py` — core 2D cross-correlation algorithms: `extended_search_area_piv()`, `get_coordinates()`.
- `pyprocess3D.py` — 3D PIV algorithms.
- `windef.py` — window-deformation iterative (multi-pass) PIV: `multipass_img_deform()`, `piv()`; driven by a `PIVSettings` instance from `settings.py`.
- `settings.py` — `PIVSettings` dataclass; defaults point at the bundled test data. Key fields: `filepath_images`, `frame_pattern_a`/`frame_pattern_b`, `windowsizes`, `overlap`, `num_iterations` (tuples must align position-by-position across passes).
- `validation.py` — spurious vector detection: `global_val()`, `global_std()`, `sig2noise_val()`.
- `filters.py` — outlier replacement: `replace_outliers()`, which calls into `lib.replace_nans()` for the actual NaN inpainting.
- `lib.py` — low-level NaN inpainting used by `filters.py`.
- `tools.py` — I/O and visualization: `imread()`, `save()`, `display_vector_field()`, `transform_coordinates()` (always call before saving/displaying results — raw PIV output coordinates are in image/array convention, not physical/plot convention).
- `scaling.py` — coordinate scaling and transformation.
- `preprocess.py` — image preprocessing (background subtraction, masking).
- `smoothn.py` — robust spline smoothing; a Python port of `smoothn.m` (D. Garcia / Prof. Lewis) bundled for convenience — it is not covered by the OpenPIV license, see README for attribution.
- `phase_separation.py` — solid-phase / liquid-tracer separation utilities.

### Typical call chains
- Quick path: `piv.simple_piv(frame_a, frame_b, plot=False)` → `(x, y, u, v, s2n)`.
- Full pipeline: `piv.process_pair(frame_a, frame_b)` → `(x, y, u, v, mask)`.
- Manual pipeline: `pyprocess.extended_search_area_piv()` for raw `(u, v, s2n)` → `validation.sig2noise_val()`/`global_val()` to flag spurious vectors → `filters.replace_outliers()` to fill them → `tools.transform_coordinates()` → `tools.save()`.
- Batch/multi-pass: build a `PIVSettings`, then `windef.piv(settings)`.

### Tests
- `openpiv/test/` (~216 tests). `conftest.py` forces the `Agg` matplotlib backend and patches `plt.show` so tests run headless — keep that in mind if adding plotting code.

### Packaging notes
- Package name on PyPI is `OpenPIV`; import name is lowercase `openpiv`.
- Dependencies are declared in `pyproject.toml` (Poetry) with a fallback `setup.py`; keep both in sync if changing dependencies or version.
- `pyproject.toml` still uses the deprecated `[tool.poetry.dev-dependencies]` section — this warns but is harmless, not a bug to fix incidentally.
