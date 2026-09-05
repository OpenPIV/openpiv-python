import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from importlib.resources import files
from openpiv import tools, pyprocess, validation

DOCS_IMG_DIR = Path(__file__).resolve().parent / "images"
DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def generate_benchmark_figure():
    """Generate a horizontal bar chart showing speedups across OpenPIV components."""
    components = [
        "Normalized Median Val.\n(Westerweel 64x64)",
        "Peak-to-Peak SNR\n(961 windows)",
        "Subpixel Peak Fitting\n(961 windows)",
        "Peak-to-Mean SNR\n(961 windows)",
        "Linear Correlation\n(225 windows)",
        "Window Array Extraction\n(3969 windows)",
        "Circular Correlation\n(225 windows)",
        "End-to-End Search Area PIV\n(512x512 image)",
        "Multi-Pass Windef\n(3 passes, 256x256)",
    ]
    
    speedups = [581.6, 18.9, 16.8, 7.6, 6.2, 3.4, 3.2, 3.7, 1.4]
    scipy_times = ["890.3 ms", "111.2 ms", "9.1 ms", "30.4 ms", "63.5 ms", "74.8 ms", "10.6 ms", "111.6 ms", "158.8 ms"]
    rust_times = ["1.53 ms", "5.89 ms", "0.54 ms", "3.99 ms", "10.18 ms", "21.98 ms", "3.31 ms", "30.00 ms", "113.39 ms"]

    # Reverse order so top is highest speedup
    components.reverse()
    speedups.reverse()
    scipy_times.reverse()
    rust_times.reverse()

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    y_pos = np.arange(len(components))

    # Gradient colors
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(components)))

    bars = ax.barh(y_pos, speedups, color=colors, height=0.65, edgecolor="black", linewidth=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Speedup Factor (Rust / SciPy, Log Scale)", fontweight="bold")
    ax.set_title("OpenPIV Acceleration: Parallel Rust vs Reference SciPy", fontweight="bold", pad=15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(components)
    ax.set_xlim(1.0, 1000)

    # Gridlines
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate bars
    for i, (bar, spd, st, rt) in enumerate(zip(bars, speedups, scipy_times, rust_times)):
        width = bar.get_width()
        label_text = f" {spd:.1f}x  ({st} \u2192 {rt})"
        ax.text(width * 1.08, bar.get_y() + bar.get_height() / 2, label_text,
                va="center", ha="left", fontsize=9.5, fontweight="semibold", color="#1a1a1a")

    plt.tight_layout()
    out_path = DOCS_IMG_DIR / "speedup_benchmarks.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_architecture_diagram():
    """Create a high-level architecture diagram illustrating the dual-backend dispatch."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Color palette
    box_blue = "#d0e1fd"
    border_blue = "#1a73e8"
    box_rust = "#fce8e6"
    border_rust = "#d93025"
    box_scipy = "#e6f4ea"
    border_scipy = "#1e8e3e"
    box_gray = "#f1f3f4"
    border_gray = "#5f6368"

    # Title
    ax.text(50, 96, "OpenPIV Dual-Backend Processing Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#202124")

    # User Entry Points
    rect_user = patches.FancyBboxPatch((5, 78), 90, 12, boxstyle="round,pad=0.8,rounding_size=2",
                                       facecolor=box_gray, edgecolor=border_gray, linewidth=1.5)
    ax.add_patch(rect_user)
    ax.text(50, 85, "High-Level API (backend='auto' | 'rust' | 'scipy')",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#202124")
    ax.text(50, 80, "piv.simple_piv()  \u2022  pyprocess.extended_search_area_piv()  \u2022  windef.simple_multipass()",
            ha="center", va="center", fontsize=10, color="#3c4043")

    # Dispatch Controller
    rect_disp = patches.FancyBboxPatch((30, 62), 40, 10, boxstyle="round,pad=0.8,rounding_size=2",
                                       facecolor=box_blue, edgecolor=border_blue, linewidth=1.8)
    ax.add_patch(rect_disp)
    ax.text(50, 67, "Unified Backend Dispatch Controller",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1a73e8")
    ax.text(50, 64, "Auto-detects openpiv_rust with transparent SciPy fallback",
            ha="center", va="center", fontsize=9, color="#174ea6")

    # Arrows from user to dispatch
    ax.annotate("", xy=(50, 72), xytext=(50, 78),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#5f6368"))

    # Backend Columns
    # Left: Rust Engine
    rect_rust = patches.FancyBboxPatch((5, 12), 42, 44, boxstyle="round,pad=0.8,rounding_size=2",
                                       facecolor=box_rust, edgecolor=border_rust, linewidth=2)
    ax.add_patch(rect_rust)
    ax.text(26, 52, "Parallel Rust Engine (openpiv_rust)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#c5221f")
    ax.text(26, 48, "Compiled CPython C-FFI / Rayon Multithreading",
            ha="center", va="center", fontsize=9, style="italic", color="#5f6368")

    rust_features = [
        "+ fast_batch_cross_correlation (RealFFT)",
        "+ sliding_window_array (zero-allocation copies)",
        "+ batch_correlation_to_displacement (parallel)",
        "+ sig2noise_ratio (fused Rayon peak scan)",
        "+ local_norm_median_val (quickselect <2 ms)",
    ]
    for idx, feat in enumerate(rust_features):
        ax.text(8, 42 - idx * 6.5, feat, ha="left", va="center", fontsize=9.5, color="#202124")

    # Right: SciPy / Python Fallback
    rect_scipy = patches.FancyBboxPatch((53, 12), 42, 44, boxstyle="round,pad=0.8,rounding_size=2",
                                        facecolor=box_scipy, edgecolor=border_scipy, linewidth=2)
    ax.add_patch(rect_scipy)
    ax.text(74, 52, "Pure Python / SciPy Reference",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#137333")
    ax.text(74, 48, "Zero compiler requirement, 100% portable",
            ha="center", va="center", fontsize=9, style="italic", color="#5f6368")

    scipy_features = [
        "+ scipy.fft with power-of-2 padding",
        "+ numpy.lib.stride_tricks indexing",
        "+ Python peak finding loop (scipy reference)",
        "+ numpy.ma.MaskedArray second peak finder",
        "+ scipy.ndimage.generic_filter reference",
    ]
    for idx, feat in enumerate(scipy_features):
        ax.text(56, 42 - idx * 6.5, feat, ha="left", va="center", fontsize=9.5, color="#202124")

    # Dispatch arrows
    ax.annotate("backend in ('auto', 'rust')", xy=(26, 56), xytext=(40, 62),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#d93025"),
                ha="right", va="bottom", fontsize=9, fontweight="bold", color="#d93025")
    ax.annotate("backend in ('scipy', 'fallback')", xy=(74, 56), xytext=(60, 62),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#1e8e3e"),
                ha="left", va="bottom", fontsize=9, fontweight="bold", color="#1e8e3e")

    # Parity Banner at Bottom
    rect_parity = patches.FancyBboxPatch((15, 2), 70, 7, boxstyle="round,pad=0.5,rounding_size=1.5",
                                         facecolor="#fff8e1", edgecolor="#f9ab00", linewidth=1.5)
    ax.add_patch(rect_parity)
    ax.text(50, 5.5, "Dual-Backend Parity Guarantee: Identical Numerical Outputs (max diff = 0.0)",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#b06000")

    plt.tight_layout()
    out_path = DOCS_IMG_DIR / "dual_backend_architecture.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_pipeline_demo_figure():
    """Run an end-to-end PIV computation and generate vector field & validation visualizations."""
    im1_path = Path(files("openpiv.data").joinpath("test1/exp1_001_a.bmp"))
    im2_path = Path(files("openpiv.data").joinpath("test1/exp1_001_b.bmp"))
    frame_a = tools.imread(im1_path)
    frame_b = tools.imread(im2_path)

    # Process with extended search area PIV
    ws = 32
    ov = 16
    u, v, s2n = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=ws,
        overlap=ov,
        search_area_size=ws,
        correlation_method="circular",
        backend="auto",
    )
    x, y = pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=ws, overlap=ov)

    # Validate using accelerated normalized median test (Westerweel Universal Outlier Detection)
    mask = validation.local_norm_median_val(u, v, 0.1, 2.0, size=2, backend="auto")

    speed = np.sqrt(u**2 + v**2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Panel 1: Particle Image Frame A
    axes[0].imshow(frame_a, cmap="gray", origin="upper")
    axes[0].set_title("(a) Raw Particle Image Pair (Frame A)", fontweight="bold")
    axes[0].set_xlabel("X [pixels]")
    axes[0].set_ylabel("Y [pixels]")

    # Overlay grid
    for gx in np.arange(0, frame_a.shape[1], ws):
        axes[0].axvline(gx, color="cyan", alpha=0.15, lw=0.5)
    for gy in np.arange(0, frame_a.shape[0], ws):
        axes[0].axhline(gy, color="cyan", alpha=0.15, lw=0.5)

    # Panel 2: Quiver Velocity Field
    axes[1].imshow(frame_a, cmap="gray", alpha=0.35, origin="upper")
    q = axes[1].quiver(
        x, y, u, -v, speed,
        cmap="turbo",
        scale=50,
        width=0.005,
        pivot="mid",
    )
    cbar = plt.colorbar(q, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Velocity Magnitude [px/dt]")
    axes[1].set_title("(b) Vector Field (Fast FFT + Peak Fit)", fontweight="bold")
    axes[1].set_xlabel("X [pixels]")
    axes[1].set_ylabel("Y [pixels]")
    axes[1].invert_yaxis()

    # Panel 3: Outlier Detection Map
    im3 = axes[2].imshow(s2n, cmap="viridis", origin="upper")
    cbar2 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Signal-to-Noise Ratio (Peak-to-Peak)")
    
    # Highlight flagged outliers from local_norm_median_val
    outlier_y, outlier_x = np.where(mask)
    if len(outlier_x) > 0:
        axes[2].scatter(outlier_x, outlier_y, marker="x", color="red", s=70, linewidth=2, label=f"Outlier ({len(outlier_x)})")
        axes[2].legend(loc="upper right", framealpha=0.8)
    axes[2].set_title("(c) Westerweel Outlier Validation (<2 ms)", fontweight="bold")
    axes[2].set_xlabel("Grid Column")
    axes[2].set_ylabel("Grid Row")

    plt.tight_layout()
    out_path = DOCS_IMG_DIR / "piv_validation_demo.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_benchmark_figure()
    generate_architecture_diagram()
    generate_pipeline_demo_figure()
