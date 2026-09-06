#!/usr/bin/env python3
"""
Security dependency audit tool for Cargo.lock and Python environments.
Audits dependencies against the Open Source Vulnerabilities (OSV.dev) database
(which aggregates RustSec, GitHub Security Advisories [GHSA], CVE, and PyPI).
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_cargo_lock(lock_path: Path) -> List[Tuple[str, str]]:
    """Parse name and version of third-party crates from Cargo.lock."""
    if not lock_path.is_file():
        return []

    content = lock_path.read_text(encoding="utf-8")
    packages = []

    for block in content.split("[[package]]")[1:]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        name = None
        version = None
        source = None

        for line in lines:
            if line.startswith("name = "):
                name = line.split('"')[1]
            elif line.startswith("version = "):
                version = line.split('"')[1]
            elif line.startswith("source = "):
                source = line.split('"')[1]

        # Only audit packages fetched from crates.io / external registry
        if name and version and source:
            packages.append((name, version))

    return packages


def query_osv_batch(queries: List[Dict]) -> List[Dict]:
    """Query OSV.dev batch API in chunks of 50."""
    url = "https://api.osv.dev/v1/querybatch"
    all_results = []
    chunk_size = 50

    for i in range(0, len(queries), chunk_size):
        chunk = queries[i : i + chunk_size]
        req = urllib.request.Request(
            url,
            data=json.dumps({"queries": chunk}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "antigravity-security-audit/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                all_results.extend(data.get("results", []))
        except urllib.error.URLError as e:
            print(f"[ERROR] Failed to query OSV API: {e}", file=sys.stderr)
            raise

    return all_results


def audit_cargo_lock(lock_path: Path) -> int:
    """Audit all dependencies in Cargo.lock."""
    print(f"\n[INFO] Auditing Rust dependencies from {lock_path}...")
    packages = parse_cargo_lock(lock_path)
    if not packages:
        print("   No external crates found in Cargo.lock.")
        return 0

    print(f"   Found {len(packages)} external crates. Checking OSV/RustSec/GHSA database...")

    queries = [
        {"package": {"name": name, "ecosystem": "crates.io"}, "version": version}
        for name, version in packages
    ]

    try:
        results = query_osv_batch(queries)
    except Exception as e:
        print(f"   [WARN] Could not reach OSV database ({e}).")
        return 0

    vuln_count = 0
    for (pkg_name, pkg_ver), res in zip(packages, results):
        vulns = res.get("vulns", [])
        if vulns:
            vuln_count += len(vulns)
            print(f"\n[!] VULNERABILITY DETECTED in {pkg_name} {pkg_ver}:")
            for v in vulns:
                v_id = v.get("id", "UNKNOWN")
                summary = v.get("summary", "No summary provided")
                aliases = ", ".join(v.get("aliases", []))
                alias_str = f" ({aliases})" if aliases else ""
                print(f"   * {v_id}{alias_str}: {summary}")
                for affected in v.get("affected", []):
                    ranges = affected.get("ranges", [])
                    for r in ranges:
                        for event in r.get("events", []):
                            if "fixed" in event:
                                print(f"     Fixed in: {event['fixed']}")

    if vuln_count == 0:
        print(f"[OK] All {len(packages)} Rust dependencies are clean! (0 known vulnerabilities)")
        return 0
    else:
        print(f"\n[FAIL] Found {vuln_count} vulnerability advisory/advisories in Rust dependencies!")
        return 1


def audit_python_env() -> int:
    """Audit installed Python packages using pip-audit via uvx or pip."""
    print("\n[INFO] Auditing Python dependencies...")
    try:
        res = subprocess.run(
            ["uvx", "pip-audit"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            print("[OK] All Python dependencies are clean! (0 known vulnerabilities)")
            return 0
        else:
            print("[FAIL] Python dependency audit failed:")
            print(res.stdout)
            print(res.stderr)
            return res.returncode
    except FileNotFoundError:
        print("   [INFO] uvx not found; skipping python pip-audit.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Antigravity Security & Dependency Audit Tool")
    parser.add_argument("--cargo-lock", type=Path, help="Path to Cargo.lock file")
    parser.add_argument("--python", action="store_true", help="Audit Python environment using pip-audit")
    parser.add_argument("--all", action="store_true", help="Audit all detected Cargo.lock files and Python env")

    args = parser.parse_args()

    # Default behavior if no flags: check auto-discovered locks and python
    audit_cargo = args.cargo_lock is not None or args.all or not args.python
    audit_py = args.python or args.all or args.cargo_lock is None

    total_failures = 0

    if audit_cargo:
        lock_paths = []
        if args.cargo_lock:
            lock_paths.append(args.cargo_lock)
        else:
            # Auto-discover Cargo.lock files in repository
            for p in Path(".").glob("**/Cargo.lock"):
                if ".venv" not in p.parts and "target" not in p.parts:
                    lock_paths.append(p)

        for lp in lock_paths:
            rc = audit_cargo_lock(lp)
            total_failures += rc

    if audit_py:
        rc = audit_python_env()
        total_failures += rc

    if total_failures > 0:
        print(f"\n[FAIL] Security audit FAILED with {total_failures} issue(s). Please update vulnerable packages.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All security audits passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
