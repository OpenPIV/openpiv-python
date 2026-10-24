"""Cycle survey: how fast is the flow in every frame pair?

Before comparing a stereo-PIV result with a reference it pays to know
*where in the cycle* the measurement was taken.  Near a flow reversal the
in-plane velocity passes through zero, and a component that is dominated
by noise there carries no information -- it can even appear to
anti-correlate with a reference computed at the same instant.

The survey runs the ordinary 2C PIV on every frame pair of a stream and
reports the median particle displacement, which is a cheap proxy for the
signal strength of that pair.

The 2C PIV runs on the **raw** frames, the same as
:func:`openpiv.examples.stereo.pipeline.run_2c_piv` used by the 3C
reconstruction, so the numbers are directly comparable with it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import StereoConfig
from .frames import read_pair
from .pipeline import run_2c_piv

__all__ = ["summarize_survey", "survey_pair", "survey_stream"]


def _crop(frame: np.ndarray, crop: Sequence[int] | None) -> np.ndarray:
    """Cut a ``(x0, y0, width, height)`` box out of the frame."""
    if crop is None:
        return frame
    x0, y0, width, height = (int(value) for value in crop)
    return frame[y0:y0 + height, x0:x0 + width]


def survey_pair(
    config: StereoConfig,
    pair_index: int,
    camera: int = 1,
    crop: Sequence[int] | None = None,
    window: int = 64,
    overlap: int = 32,
) -> dict:
    """Measure the particle displacement of a single frame pair.

    Parameters
    ----------
    config:
        The stereo configuration; the stream of ``camera`` is read.
    pair_index:
        Index of the frame pair in the stream.
    camera:
        Which camera of the stereo pair to measure.
    crop:
        Optional ``(x0, y0, width, height)`` box, to concentrate the
        measurement on the region of interest (and to go faster).
    window, overlap:
        Interrogation window and overlap of the 2C PIV.

    Returns
    -------
    dict
        ``pair``, ``median_u``, ``median_v``, ``median_speed`` (pixels),
        ``n_vectors`` and the pair index.
    """
    path = config.stream_path(camera)
    pulse_a, pulse_b = read_pair(
        path, pair_index, nx=config.sensor_size[0], ny=config.sensor_size[1]
    )
    frame_a = _crop(pulse_a, crop).astype(np.float32)
    frame_b = _crop(pulse_b, crop).astype(np.float32)
    _, _, u, v, _ = run_2c_piv(frame_a, frame_b, window, overlap)
    valid = np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        return {
            "pair": pair_index,
            "camera": camera,
            "median_u": float("nan"),
            "median_v": float("nan"),
            "median_speed": float("nan"),
            "n_vectors": 0,
        }
    speed = np.hypot(u[valid], v[valid])
    return {
        "pair": pair_index,
        "camera": camera,
        "median_u": float(np.median(u[valid])),
        "median_v": float(np.median(v[valid])),
        "median_speed": float(np.median(speed)),
        "n_vectors": int(valid.sum()),
    }


def survey_stream(
    config: StereoConfig,
    camera: int = 1,
    pairs: Sequence[int] | None = None,
    crop: Sequence[int] | None = None,
    window: int = 64,
    overlap: int = 32,
    progress: bool = True,
) -> list[dict]:
    """Measure the displacement of every frame pair of a stream.

    Returns
    -------
    list of dict
        One record per pair, see :func:`survey_pair`.
    """
    if pairs is None:
        from .frames import ims_info

        info = ims_info(
            config.stream_path(camera),
            nx=config.sensor_size[0],
            ny=config.sensor_size[1],
        )
        pairs = range(int(info["n_pairs"]))
    records = []
    for index in pairs:
        record = survey_pair(config, int(index), camera, crop, window, overlap)
        records.append(record)
        if progress:
            print(
                f"pair {record['pair']:4d}: median |u| {record['median_speed']:.3f} px "
                f"(u {record['median_u']:+.3f}, v {record['median_v']:+.3f}) "
                f"n {record['n_vectors']}",
                flush=True,
            )
    return records


def summarize_survey(records: Sequence[dict]) -> dict:
    """Summarize a survey: the strongest pairs and the cycle statistics.

    Returns
    -------
    dict
        ``best`` (the records with the largest displacement, strongest
        first), ``weakest``, ``median_speed`` and the pair range.
    """
    ordered = sorted(records, key=lambda record: record["median_speed"], reverse=True)
    speeds = np.array([record["median_speed"] for record in records], dtype=float)
    finite = speeds[np.isfinite(speeds)]
    return {
        "best": ordered[:5],
        "worst": ordered[-5:],
        "median_speed": float(np.median(finite)) if finite.size else float("nan"),
        "max_speed": float(finite.max()) if finite.size else float("nan"),
        "min_speed": float(finite.min()) if finite.size else float("nan"),
        "n_pairs": len(records),
    }


def survey_dispersion(
    records: Sequence[dict], noise_level: float = 0.05
) -> dict:
    """How many pairs carry a signal above the noise level?

    The in-plane velocity of a noise-dominated pair changes sign from one
    pair to the next; a pair with a real signal keeps its sign over a
    stretch of the cycle.  The fraction of pairs whose signed
    displacement is larger than ``noise_level`` pixels is a simple
    measure of how much of the record is usable.

    Returns
    -------
    dict
        ``fraction_above_noise``, ``noise_level`` and the sign changes.
    """
    signed = np.array([record["median_u"] for record in records], dtype=float)
    finite = signed[np.isfinite(signed)]
    above = finite[np.abs(finite) > noise_level]
    signs = np.sign(above)
    changes = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    return {
        "fraction_above_noise": float(above.size / finite.size) if finite.size else 0.0,
        "noise_level": noise_level,
        "sign_changes": changes,
    }
