# OpenPIV Python Developer & Security Guidelines

## Security and Dependency Auditing
- **Vulnerability Checks**: Before committing any changes that add or update dependencies or lockfiles (`Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `requirements.txt`), ALWAYS run:
  ```powershell
  python .agents/skills/security-audit/scripts/audit_deps.py
  ```
  Ensure all dependencies have 0 known vulnerabilities across OSV, RustSec, GHSA, and PyPI databases.
- **Security Advisories**: If a vulnerability is flagged (e.g. via Trivy, Dependabot, or Sourcery PR checks), immediately upgrade the affected crate/package in `Cargo.toml`/`pyproject.toml`, run `cargo update` or `uv lock`, and adapt any breaking API changes.

## File Formatting & Integrity
- **No UTF-8 BOM**: All source files (`.py`, `.rs`, `.toml`, `.yml`, `.rst`, `.md`) must be saved in standard UTF-8 without byte-order marks (`\xef\xbb\xbf`).

## Rust Acceleration & Backend Architecture
- **Dual-Backend Parity**: Any performance-critical function offloaded to `openpiv_rust` MUST provide identical numerical results to its pure Python/SciPy counterpart.
- **Explicit Backend Control**: Functions accelerated with Rust (`fft_correlate_images`, `find_subpixel_peak_position`, `correlation_to_displacement`, `sig2noise_ratio`, `local_norm_median_val`, `sliding_window_array`) must accept a `backend: str = "auto"` parameter:
  - `"auto"`: Uses `openpiv_rust` if installed, falls back cleanly to Python/SciPy.
  - `"rust"`: Uses `openpiv_rust`. Raises informative `ImportError` if not compiled/installed.
  - `"scipy"` / `"python"`: Forces the pure Python/SciPy reference path.
- **Test Suite Safety**: Rust-specific tests must use `openpiv_rust = pytest.importorskip("openpiv_rust")` at the module top level so test suites pass cleanly in environments without the Rust compiler.
