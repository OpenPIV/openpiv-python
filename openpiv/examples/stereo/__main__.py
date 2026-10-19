"""Command line driver of the stereo-PIV example.

.. code-block:: bash

    python -m openpiv.examples.stereo --data-root /data/run \
        --mark-table /data/run/MarkPositionTable.xml \
        --davis-calibration /data/run/Calibration.xml all

Every step writes its results into the output directory and can be run on
its own; ``all`` simply chains them in order.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import davis, frames, pipeline
from .config import StereoConfig

__all__ = ["STEPS", "main"]

#: The pipeline steps, in execution order.
STEP_ORDER = ("calibrate", "dewarp", "disparity", "piv")


def _log(message: str) -> None:
    print(message, flush=True)


def _require_marks(config: StereoConfig) -> dict[int, davis.CameraMarks]:
    if config.mark_table is None:
        raise SystemExit(
            "no calibration marks given: pass --mark-table MarkPositionTable.xml"
        )
    return davis.parse_mark_position_table(config.mark_table)


def _load_cameras(config: StereoConfig, marks: dict[int, davis.CameraMarks]):
    cams = pipeline.fit_cameras(config.model, config.sensor_size, marks, config.cameras)
    report = pipeline.reprojection_report(cams, marks, config.cameras)
    reference: dict[int, float] = {}
    if config.davis_calibration is not None and Path(config.davis_calibration).exists():
        reference = davis.parse_calibration_rms(config.davis_calibration)
    for entry in report:
        number = entry["camera"]
        davis_rms = reference.get(number)
        suffix = f" (DaVis {davis_rms:.4f} px)" if davis_rms is not None else ""
        _log(
            f"camera {number}: RMS {entry['rms']:.4f} px{suffix} | "
            f"median {entry['median']:.3f} p95 {entry['p95']:.3f} "
            f"max {entry['max']:.3f} over {entry['marks']} marks"
        )
    return cams, report


def _read_pairs(config: StereoConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read the (pulse A, pulse B) pair of every camera of the stereo pair."""
    result = []
    for number in config.cameras:
        path = config.stream_path(number)
        info = frames.ims_info(path, nx=config.sensor_size[0], ny=config.sensor_size[1])
        _log(
            f"camera {number}: {path.name} -> {info['n_pairs']} pairs "
            f"{info['nx']}x{info['ny']} (pair {config.pair_index})"
        )
        pulse_a, pulse_b = frames.read_pair(
            path, config.pair_index, nx=config.sensor_size[0], ny=config.sensor_size[1]
        )
        result.append((pulse_a, pulse_b))
    return result


def step_info(config: StereoConfig) -> int:
    """Print the stream layout and the resolved configuration."""
    _log(f"data root : {config.data_root}")
    _log(f"cameras   : {config.cameras}")
    _log(f"sensor    : {config.sensor_size[0]}x{config.sensor_size[1]}")
    _log(f"sheet z   : {config.sheet_z} mm")
    _log(f"output    : {config.out_dir}")
    for number in config.cameras:
        path = config.stream_path(number)
        info = frames.ims_info(path, nx=config.sensor_size[0], ny=config.sensor_size[1])
        _log(
            f"camera {number}: {path.name} | {info['n_pairs']} pairs | "
            f"{info['nx']}x{info['ny']} | {info['pair_bytes']:,} bytes/pair"
        )
    return 0


def step_calibrate(config: StereoConfig) -> int:
    """Fit the stereo pair to the DaVis marks."""
    marks = _require_marks(config)
    cams, report = _load_cameras(config, marks)
    out = config.ensure_out()
    for cam in cams:
        cam.save_parameters(str(out), f"{cam.name}_{config.model}")
    (out / "reprojection.json").write_text(json.dumps(report, indent=2))
    _log(f"camera parameters -> {out}")
    return 0


def step_dewarp(config: StereoConfig) -> int:
    """Rectify the stereo pair onto a common world grid."""
    marks = _require_marks(config)
    cams, _ = _load_cameras(config, marks)
    Xw, Yw, info = pipeline.build_world_grid(cams, z=config.sheet_z, stride=config.grid_stride)
    _log(
        f"grid {Xw.shape[1]}x{Xw.shape[0]} | "
        f"X[{info['min_x']:.1f},{info['max_x']:.1f}] "
        f"Y[{info['min_y']:.1f},{info['max_y']:.1f}] | "
        f"{info['scale']:.4f} mm/px at z={info['z']}"
    )
    maps = pipeline.build_dewarp_maps(cams, Xw, Yw, z=config.sheet_z)
    out = config.ensure_out()
    scalar = {key: value for key, value in info.items() if isinstance(value, (int, float))}
    np.savez(out / "common_grid.npz", Xw=Xw, Yw=Yw, **scalar)
    np.savez(out / "maps.npz", **maps)
    _log(f"grid + dewarp maps -> {out}")
    return 0


def step_disparity(config: StereoConfig) -> int:
    """Measure the particle disparity and self-calibrate the mappings."""
    marks = _require_marks(config)
    cams, _ = _load_cameras(config, marks)
    Xw, Yw, _ = pipeline.build_world_grid(cams, z=config.sheet_z, stride=config.grid_stride)
    maps = pipeline.build_dewarp_maps(cams, Xw, Yw, z=config.sheet_z)
    raw = _read_pairs(config)
    corrected, report = pipeline.correct_disparity(
        [pair[0] for pair in raw], maps, config.disparity_passes, config.self_cal_iterations
    )
    _log(
        f"disparity: median |D| {report['residual_before']:.3f} px -> "
        f"{report['residual_after']:.3f} px | bulk shift "
        f"({report['bulk_shift'][0]:.2f}, {report['bulk_shift'][1]:.2f}) px"
    )
    out = config.ensure_out()
    np.savez(out / "maps_corrected.npz", **corrected)
    np.savez(
        out / "disparity.npz",
        x=report["x"],
        y=report["y"],
        dx=report["dx"],
        dy=report["dy"],
        s2n=report["s2n"],
    )
    _log(f"corrected maps -> {out}")
    return 0


def step_piv(config: StereoConfig) -> int:
    """Run the 2C PIV per camera and reconstruct the 3C velocity field.

    The 2C PIV runs on the *raw* camera images, so the displacements are
    pixel displacements in the camera frame -- which is what the stereo
    reconstruction needs.
    """
    from openpiv import filters, validation

    marks = _require_marks(config)
    cams, _ = _load_cameras(config, marks)
    out = config.ensure_out()
    raw = _read_pairs(config)

    results = []
    for number, (pulse_a, pulse_b) in zip(config.cameras, raw):
        x, y, u, v, s2n = pipeline.run_2c_piv(
            pulse_a.astype(np.float32), pulse_b.astype(np.float32),
            config.piv_window, config.piv_overlap,
        )
        results.append((x, y, u, v, s2n))
        _log(f"camera {number}: 2C grid {u.shape} | median s2n {np.nanmedian(s2n):.2f}")

    mask = np.zeros(results[0][2].shape, dtype=bool)
    filled = []
    for _, _, u, v, _ in results:
        flags = np.asarray(
            validation.global_val(
                np.ma.masked_invalid(u), np.ma.masked_invalid(v), (-10, 10), (-10, 10)
            )
        )
        flags = flags | ~np.isfinite(u) | ~np.isfinite(v)
        mask |= flags
        u_filled, v_filled = filters.replace_outliers(
            np.ma.array(u, mask=flags),
            np.ma.array(v, mask=flags),
            flags,
            method="localmean",
        )
        filled.append((np.asarray(u_filled), np.asarray(v_filled)))
    _log(f"flagged {100 * mask.mean():.1f}% of the vectors")

    U, V, W = pipeline.reconstruct_3c(
        cams,
        [results[0][0], results[1][0]],
        [results[0][1], results[1][1]],
        [filled[0][0], filled[1][0]],
        [filled[0][1], filled[1][1]],
        z=config.sheet_z,
    )
    U, V, W = (np.where(mask.ravel(), np.nan, value) for value in (U, V, W))
    world_x, world_y = pipeline.backproject_to_plane(cams[0], results[0][0], results[0][1], z=config.sheet_z)
    _log(
        f"3C: |U| {np.nanmedian(np.abs(U)):.3f} |V| {np.nanmedian(np.abs(V)):.3f} "
        f"|W| {np.nanmedian(np.abs(W)):.3f} mm/frame"
    )
    np.savez(
        out / "stereo_3c.npz",
        x=world_x,
        y=world_y,
        U=U,
        V=V,
        W=W,
        mask=mask.ravel(),
    )
    _log(f"3C field -> {out / 'stereo_3c.npz'}")
    return 0


STEPS = {
    "info": step_info,
    "calibrate": step_calibrate,
    "dewarp": step_dewarp,
    "disparity": step_disparity,
    "piv": step_piv,
}


def _split_step(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """Separate the step keyword from the configuration arguments."""
    step: str | None = None
    rest: list[str] = []
    for token in argv:
        if step is None and not token.startswith("-") and token in (*STEPS, "all"):
            step = token
        else:
            rest.append(token)
    return step, rest


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``python -m openpiv.examples.stereo``."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    step, rest = _split_step(raw_argv)
    if step is None:
        print("usage: python -m openpiv.examples.stereo STEP [options]", file=sys.stderr)
        print(f"steps: {', '.join(sorted(STEPS))}, all", file=sys.stderr)
        return 2
    config = StereoConfig.from_args(rest)
    selected = list(STEP_ORDER) if step == "all" else [step]
    try:
        for name in selected:
            _log(f"--- {name} ---")
            STEPS[name](config)
    except SystemExit:
        raise
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
