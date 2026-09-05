---
name: security-audit
description: >-
  Audits project dependencies and lockfiles for security vulnerabilities, CVEs, and security advisories (e.g. GHSA, RustSec, PyPI) in Python and Rust codebases. Use whenever adding, updating, or modifying dependencies in Cargo.toml, Cargo.lock, pyproject.toml, uv.lock, or requirements.txt, before submitting pull requests, or when security scanning is requested.
---

# Security & Dependency Vulnerability Audit

This skill guides the agent in auditing project dependencies for known security vulnerabilities (CVEs, GHSA advisories, RustSec advisories) across both Rust and Python ecosystems before code is committed or pushed.

## When to Run

Run this audit workflow whenever:
1. Dependencies are added, updated, or upgraded in `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, or `requirements.txt`.
2. A lockfile is regenerated or modified.
3. Preparing a pull request or pushing changes to remote branches.
4. Triaging security alerts reported by GitHub Dependabot, Trivy, or Sourcery.

---

## Audit Procedures

### 1. Fast Batch Audit (Rust & Python)

Run the included multi-ecosystem audit script:
```powershell
python .agents/skills/security-audit/scripts/audit_deps.py
```
This tool:
* Parses all detected `Cargo.lock` files.
* Queries the [OSV.dev](https://osv.dev) database (aggregating RustSec, GitHub Security Advisories [GHSA], CVE, and crates.io security bulletins) in batch via HTTP in ~200ms.
* Runs `uvx pip-audit` to scan Python packages against PyPI / OSV advisory databases.
* Returns exit code `0` on success, or exit code `1` with exact advisory IDs, affected packages, and remediation versions if vulnerabilities are detected.

### 2. Rust-Specific Audit (`cargo-audit`)

If `cargo-audit` is available:
```powershell
cargo audit --file crates/openpiv_rust/Cargo.lock
```
To install `cargo-audit`:
```powershell
cargo install cargo-audit --locked
```

### 3. Python-Specific Audit (`pip-audit`)

Run without installation via `uvx`:
```powershell
uvx pip-audit
```
Or within an active virtualenv:
```powershell
pip-audit
```

---

## Remediation Workflow

When a vulnerability is detected:
1. **Identify the Advisory**: Note the advisory ID (e.g., `GHSA-36hh-v3qg-5jq4` / `RUSTSEC-2026-0176`) and the minimum fixed version.
2. **Update the Manifest**:
   - For Rust: Update `Cargo.toml` with the patched version requirement (e.g., `pyo3 = "0.29"`).
   - For Python: Update `pyproject.toml` or `dependencies` with `>= <fixed-version>`.
3. **Regenerate Lockfiles**:
   - For Rust: Run `cargo update` or `cargo update -p <crate_name>`.
   - For Python: Run `uv lock --upgrade-package <package_name>`.
4. **Adapt Breaking Changes**: Check if the dependency upgrade introduces breaking API changes (e.g., PyO3 API renames such as `py.allow_threads` -> `py.detach`), compile, and run the test suite.
5. **Re-run the Audit**: Confirm that `audit_deps.py` reports `0` known vulnerabilities.
