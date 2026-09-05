import time
import tempfile
import pathlib
import numpy as np
from imageio.v3 import imwrite
from openpiv import windef
from openpiv.settings import PIVSettings
from openpiv.test import test_process

def run_multigrid(frame_a, frame_b, backend, n_runs=3):
    settings = PIVSettings()
    settings.windowsizes = (64, 32, 16)
    settings.overlap = (32, 16, 8)
    settings.num_iterations = 3
    settings.backend = backend
    settings.sig2noise_validate = False
    settings.show_all_plots = False
    settings.show_plot = False
    
    # Warmup
    windef.multigrid_windef(frame_a, frame_b, settings)
    
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        x, y, u, v, flags = windef.multigrid_windef(frame_a, frame_b, settings)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000.0, (u, v)

def run_mp(frame_a, frame_b, backend):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        img_dir = tmp / "images"
        img_dir.mkdir()
        out_dir = tmp / "out"
        out_dir.mkdir()
        for i in range(4):
            imwrite(img_dir / f"pair_{i:03d}_a.tif", frame_a.astype(np.uint8))
            imwrite(img_dir / f"pair_{i:03d}_b.tif", frame_b.astype(np.uint8))
        settings = PIVSettings()
        settings.filepath_images = img_dir
        settings.save_path = out_dir
        settings.save_folder_suffix = f"{backend}_bench"
        settings.frame_pattern_a = "pair_*_a.tif"
        settings.frame_pattern_b = "pair_*_b.tif"
        settings.windowsizes = (64, 32, 16)
        settings.overlap = (32, 16, 8)
        settings.num_iterations = 3
        settings.backend = backend
        settings.n_cpus = 2
        settings.show_plot = False
        settings.save_plot = False
        settings.show_all_plots = False
        settings.sig2noise_validate = False
        
        t0 = time.perf_counter()
        windef.piv(settings)
        return (time.perf_counter() - t0) * 1000.0

def main():
    frame_a, frame_b = test_process.create_pair(image_size=256)
    
    print("=" * 60)
    print("  BENCHMARK: Multigrid Window Deformation (256x256, 3 passes)")
    print("=" * 60)
    time_scipy, (u_s, v_s) = run_multigrid(frame_a, frame_b, "scipy")
    time_rust, (u_r, v_r) = run_multigrid(frame_a, frame_b, "rust")
    speedup_mg = time_scipy / time_rust
    max_diff = max(np.nanmax(np.abs(u_s - u_r)), np.nanmax(np.abs(v_s - v_r)))
    print(f"Multigrid SciPy: {time_scipy:6.2f} ms | Rust: {time_rust:6.2f} ms | Speedup: {speedup_mg:4.2f}x | Max Diff: {max_diff:.2e}")

    print("\n" + "=" * 60)
    print("  BENCHMARK: Multiprocessing (4 pairs, 2 worker processes)")
    print("=" * 60)
    mp_scipy = run_mp(frame_a, frame_b, "scipy")
    mp_rust = run_mp(frame_a, frame_b, "rust")
    speedup_mp = mp_scipy / mp_rust
    print(f"Multiprocessing SciPy: {mp_scipy:6.2f} ms | Rust: {mp_rust:6.2f} ms | Speedup: {speedup_mp:4.2f}x")

if __name__ == "__main__":
    main()
