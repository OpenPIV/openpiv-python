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


def step_survey(config: StereoConfig) -> int:
    """Measure the particle displacement of every frame pair.

    The result tells how much signal each pair carries -- a pair whose
    displacement sits at the noise level cannot be compared with a
    reference, because the sign of a noise dominated component is
    random.
    """
    from .survey import summarize_survey, survey_dispersion, survey_stream

    crop = config.survey_crop
    records = survey_stream(
        config, camera=config.cameras[0], crop=crop, progress=True
    )
    summary = summarize_survey(records)
    dispersion = survey_dispersion(records, noise_level=config.noise_level)
    _log("")
    _log(f"pairs surveyed      : {summary['n_pairs']}")
    _log(f"median displacement : {summary['median_speed']:.3f} px")
    _log(f"max / min           : {summary['max_speed']:.3f} / {summary['min_speed']:.3f} px")
    _log(f"above noise ({dispersion['noise_level']} px) : "
         f"{100 * dispersion['fraction_above_noise']:.0f}% of the pairs, "
         f"{dispersion['sign_changes']} sign changes")
    _log("strongest pairs:")
    for record in summary["best"]:
        _log(
            f"  pair {record['pair']:4d}: median |u| {record['median_speed']:.3f} px "
            f"(u {record['median_u']:+.3f}, v {record['median_v']:+.3f})"
        )
    out = config.ensure_out()
    with (out / "survey.csv").open("w") as handle:
        handle.write("pair,camera,median_u,median_v,median_speed,n_vectors\n")
        for record in records:
            handle.write(
                f"{record['pair']},{record['camera']},{record['median_u']:.6f},"
                f"{record['median_v']:.6f},{record['median_speed']:.6f},"
                f"{record['n_vectors']}\n"
            )
    _log(f"survey -> {out / 'survey.csv'}")
    return 0


def step_compare(config: StereoConfig) -> int:
    """Compare the reconstruction with a DaVis vector field.

    The reference is a ``.vc7`` written by the DaVis StereoPIV module.
    Its components live in the *corrected image* frame, so the vertical
    component has to be flipped to get the world-frame velocity.
    """
    if config.davis_vector_field is None:
        raise SystemExit("no reference given: pass --davis-vector-field FILE.vc7")

    marks = _require_marks(config)
    cams, _ = _load_cameras(config, marks)
    raw = _read_pairs(config)
    fields = []
    for pulse_a, pulse_b in raw:
        fields.append(
            pipeline.run_2c_piv(
                pulse_a.astype(np.float32),
                pulse_b.astype(np.float32),
                config.piv_window,
                config.piv_overlap,
            )
        )
    U, V, W = pipeline.reconstruct_3c(
        cams,
        [fields[0][0], fields[1][0]],
        [fields[0][1], fields[1][1]],
        [fields[0][2], fields[1][2]],
        [fields[0][3], fields[1][3]],
        z=config.sheet_z,
    )
    # mm per frame -> m/s (the frame rate has to be given)
    to_ms = 1e-3 / config.frame_dt if config.frame_dt else 1.0
    world_x, world_y = pipeline.backproject_to_plane(
        cams[0], fields[0][0], fields[0][1], z=config.sheet_z
    )
    reference = davis.read_vector_field(config.davis_vector_field)
    _log(f"reference: {config.davis_vector_field.name} "
         f"({reference.u.shape[1]}x{reference.u.shape[0]} vectors, "
         f"{100 * reference.valid.mean():.0f}% valid)")
    for name, ours, theirs in (
        ("U", U * to_ms, reference.u_world),
        ("V", V * to_ms, reference.v_world),
        ("W", W * to_ms, reference.w),
    ):
        stats = pipeline.compare_fields(
            ours, theirs, reference.x, reference.y, world_x, world_y, reference.valid
        )
        _log(
            f"  {name}: ours {stats['median_ours']:+.4f} | "
            f"reference {stats['median_theirs']:+.4f} m/s | "
            f"bias {stats['bias']:+.4f} | RMSE {stats['rmse']:.4f} | "
            f"corr {stats['correlation']:+.3f} | n {stats['n']}"
        )
    return 0


STEPS = {
    "info": step_info,
    "calibrate": step_calibrate,
    "dewarp": step_dewarp,
    "disparity": step_disparity,
    "piv": step_piv,
    "survey": step_survey,
    "compare": step_compare,
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
