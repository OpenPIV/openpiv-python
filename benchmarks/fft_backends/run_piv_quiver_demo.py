import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib.resources import files
from openpiv import pyprocess, tools, validation, filters
import openpiv_rust

ARTIFACT_DIR = Path(r"C:\Users\alex\.gemini\antigravity-cli\brain\371a4578-c887-4c4a-bba3-7bbfb7f1997c")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def run_real_piv_demo():
    print("=== Running Real PIV Quiver Demo (exp1_001) ===")
    path = files('openpiv') / "data" / "test1"
    frame_a = tools.imread(path / "exp1_001_a.bmp").astype(np.int32)
    frame_b = tools.imread(path / "exp1_001_b.bmp").astype(np.int32)

    window_size = 32
    overlap = 16
    search_area_size = 32

    # Prepare window arrays
    aa = pyprocess.sliding_window_array(frame_a, (search_area_size, search_area_size), (overlap, overlap))
    bb = pyprocess.sliding_window_array(frame_b, (search_area_size, search_area_size), (overlap, overlap))
    n_rows, n_cols = pyprocess.get_field_shape(frame_a.shape, (search_area_size, search_area_size), (overlap, overlap))

    # 1. Scipy correlation
    t0 = time.perf_counter()
    scipy_corr = pyprocess.fft_correlate_images(aa, bb, correlation_method="circular", normalized_correlation=False)
    t_scipy_corr = time.perf_counter() - t0

    # 2. Rust correlation
    t0 = time.perf_counter()
    rust_corr = openpiv_rust.fft_correlate_circular(aa.astype(float), bb.astype(float), normalized_correlation=False)
    t_rust_corr = time.perf_counter() - t0

    speedup_corr = t_scipy_corr / max(t_rust_corr, 1e-6)
    print(f"Correlation Time -> SciPy: {t_scipy_corr*1000:.2f} ms | Rust: {t_rust_corr*1000:.2f} ms | Speedup: {speedup_corr:.1f}x")

    # Full End-to-end PIV
    t0 = time.perf_counter()
    u_scipy, v_scipy, s2n_scipy = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        correlation_method="circular",
        sig2noise_method="peak2peak",
    )
    t_scipy_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    rust_corr = openpiv_rust.fft_correlate_circular(aa.astype(float), bb.astype(float), normalized_correlation=False)
    u_rust, v_rust = pyprocess.correlation_to_displacement(rust_corr, n_rows, n_cols)
    s2n_rust = pyprocess.sig2noise_ratio(rust_corr, sig2noise_method="peak2peak", width=2)
    t_rust_full = time.perf_counter() - t0

    diff_u = np.nanmax(np.abs(u_scipy - u_rust))
    diff_v = np.nanmax(np.abs(v_scipy - v_rust))
    print(f"Full PIV Time    -> SciPy: {t_scipy_full*1000:.2f} ms | Rust: {t_rust_full*1000:.2f} ms")
    print(f"Max difference in velocity field: u={diff_u:.2e}, v={diff_v:.2e}")

    # Coordinates
    x, y = pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=search_area_size, overlap=overlap)

    # Outlier filtering
    flags = validation.sig2noise_val(s2n_rust, threshold=1.05)
    flags_2d = flags.reshape(n_rows, n_cols)
    u_clean, v_clean = filters.replace_outliers(u_rust, v_rust, flags_2d, method='localmean', max_iter=3, kernel_size=2)

    # Plot Quiver Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="white")
    
    # Panel 1: Scipy
    ax = axes[0]
    ax.imshow(frame_a, cmap="gray", alpha=0.5, origin="upper")
    speed_scipy = np.sqrt(u_scipy**2 + v_scipy**2)
    q1 = ax.quiver(x, y, u_scipy, -v_scipy, speed_scipy, cmap="plasma", scale=45, width=0.004)
    ax.set_title(f"SciPy Mode Vector Field\nCorr: {t_scipy_corr*1000:.1f} ms | Total: {t_scipy_full*1000:.1f} ms", fontsize=12, fontweight="bold")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    plt.colorbar(q1, ax=ax, label="Displacement [px]", fraction=0.046, pad=0.04)

    # Panel 2: Rust
    ax = axes[1]
    ax.imshow(frame_a, cmap="gray", alpha=0.5, origin="upper")
    speed_rust = np.sqrt(u_rust**2 + v_rust**2)
    q2 = ax.quiver(x, y, u_rust, -v_rust, speed_rust, cmap="plasma", scale=45, width=0.004)
    ax.set_title(f"Rust Mode Vector Field (Rayon 2D RealFFT)\nCorr: {t_rust_corr*1000:.1f} ms ({speedup_corr:.1f}x Faster)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    plt.colorbar(q2, ax=ax, label="Displacement [px]", fraction=0.046, pad=0.04)

    # Panel 3: Difference Map
    ax = axes[2]
    diff_speed = np.sqrt((u_scipy - u_rust)**2 + (v_scipy - v_rust)**2)
    im = ax.imshow(diff_speed, cmap="coolwarm", origin="upper", extent=[x.min(), x.max(), y.max(), y.min()])
    ax.set_title(f"Vector Discrepancy (|u_rust - u_scipy|)\nMax Absolute Error: {np.nanmax(diff_speed):.2e} px", fontsize=12, fontweight="bold")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    plt.colorbar(im, ax=ax, label="Difference [px]", fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = ARTIFACT_DIR / "quiver_real_piv.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved real PIV quiver plot to: {out_path}")

def run_synthetic_vortex_demo():
    print("\n=== Running Synthetic Vortex Quiver Demo ===")
    np.random.seed(42)
    img_h, img_w = 256, 256
    
    n_particles = 4000
    px = np.random.uniform(0, img_w, n_particles)
    py = np.random.uniform(0, img_h, n_particles)

    cx, cy = 128.0, 128.0
    r0 = 40.0
    gamma = 400.0

    rx = px - cx
    ry = py - cy
    r = np.sqrt(rx**2 + ry**2) + 1e-6
    v_theta = gamma * r / (r0**2 + r**2)
    
    dx_particles = -v_theta * (ry / r)
    dy_particles = v_theta * (rx / r)

    frame_a = np.zeros((img_h, img_w), dtype=np.float32)
    frame_b = np.zeros((img_h, img_w), dtype=np.float32)

    for i in range(n_particles):
        x0, y0 = int(round(px[i])), int(round(py[i]))
        if 1 <= x0 < img_w - 1 and 1 <= y0 < img_h - 1:
            frame_a[y0, x0] += 200.0
            frame_a[y0+1, x0] += 100.0
            frame_a[y0-1, x0] += 100.0
            frame_a[y0, x0+1] += 100.0
            frame_a[y0, x0-1] += 100.0

        x1, y1 = int(round(px[i] + dx_particles[i])), int(round(py[i] + dy_particles[i]))
        if 1 <= x1 < img_w - 1 and 1 <= y1 < img_h - 1:
            frame_b[y1, x1] += 200.0
            frame_b[y1+1, x1] += 100.0
            frame_b[y1-1, x1] += 100.0
            frame_b[y1, x1+1] += 100.0
            frame_b[y1, x1-1] += 100.0

    window_size = 32
    overlap = 16
    search_area_size = 32

    aa = pyprocess.sliding_window_array(frame_a, (search_area_size, search_area_size), (overlap, overlap))
    bb = pyprocess.sliding_window_array(frame_b, (search_area_size, search_area_size), (overlap, overlap))
    n_rows, n_cols = pyprocess.get_field_shape(frame_a.shape, (search_area_size, search_area_size), (overlap, overlap))

    # Scipy correlation
    t0 = time.perf_counter()
    scipy_corr = pyprocess.fft_correlate_images(aa, bb, correlation_method="circular", normalized_correlation=False)
    t_scipy_corr = time.perf_counter() - t0

    # Rust correlation
    t0 = time.perf_counter()
    rust_corr = openpiv_rust.fft_correlate_circular(aa.astype(float), bb.astype(float), normalized_correlation=False)
    t_rust_corr = time.perf_counter() - t0

    speedup_vortex = t_scipy_corr / max(t_rust_corr, 1e-6)
    print(f"Vortex Correlation -> SciPy: {t_scipy_corr*1000:.2f} ms | Rust: {t_rust_corr*1000:.2f} ms | Speedup: {speedup_vortex:.1f}x")

    u_rust, v_rust = pyprocess.correlation_to_displacement(rust_corr, n_rows, n_cols)
    x, y = pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=search_area_size, overlap=overlap)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    
    # Panel 1: Rust Quiver Field
    ax = axes[0]
    ax.imshow(frame_a, cmap="gray", alpha=0.5, origin="upper")
    speed = np.sqrt(u_rust**2 + v_rust**2)
    q = ax.quiver(x, y, u_rust, -v_rust, speed, cmap="inferno", scale=25, width=0.005)
    ax.set_title(f"Rust Mode Quiver - Lamb-Oseen Vortex Flow\nFFT Correlation: {t_rust_corr*1000:.2f} ms ({speedup_vortex:.1f}x Faster)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    plt.colorbar(q, ax=ax, label="Displacement Velocity [px]")

    # Panel 2: Streamlines
    ax = axes[1]
    strm = ax.streamplot(x[0, :], y[:, 0], u_rust, -v_rust, color=speed, cmap="inferno", density=1.5, linewidth=1.5)
    ax.set_title("Streamlines Recovered by Rust Engine", fontsize=12, fontweight="bold")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    plt.colorbar(strm.lines, ax=ax, label="Velocity Magnitude [px]")

    plt.tight_layout()
    out_path = ARTIFACT_DIR / "quiver_vortex_piv.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved vortex quiver plot to: {out_path}")

if __name__ == "__main__":
    run_real_piv_demo()
    run_synthetic_vortex_demo()
