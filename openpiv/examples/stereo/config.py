"""Configuration for the stereo-PIV example pipeline.

The example is driven by a single :class:`StereoConfig` object.  It can be
built in three ways:

* :meth:`StereoConfig.from_env` -- read defaults from ``OPENPIV_STEREO_*``
  environment variables (handy for shell/CI usage);
* :meth:`StereoConfig.from_args` -- parse a command line;
* directly in Python -- keyword arguments of the dataclass.

No path is hard-coded: every file the pipeline touches is either given
explicitly or derived from :attr:`StereoConfig.data_root`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = ["ENV_PREFIX", "StereoConfig"]

ENV_PREFIX = "OPENPIV_STEREO_"

#: Camera model used for the stereo pair.  ``poly`` is the Soloff model that
#: reproduces the DaVis calibration of the reference data set; ``dlt`` and
#: ``pinhole`` are supported for comparison.
MODEL_CHOICES = ("poly", "dlt", "pinhole")


def _env_path(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(ENV_PREFIX + name)
    if value:
        return Path(value).expanduser()
    return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(ENV_PREFIX + name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(ENV_PREFIX + name)
    return float(value) if value else default


def _env_cameras(name: str, default: Sequence[int]) -> tuple[int, ...]:
    value = os.environ.get(ENV_PREFIX + name)
    if not value:
        return tuple(default)
    cams = tuple(int(part) for part in value.replace(",", " ").split())
    if len(cams) < 2:
        raise ValueError(f"{ENV_PREFIX}{name} needs at least two cameras, got {cams}")
    return cams


@dataclass(frozen=True)
class StereoConfig:
    """Everything the stereo-PIV example needs to run.

    Attributes
    ----------
    data_root:
        Directory holding the raw camera streams.
    cameras:
        Camera numbers forming the stereo pair, e.g. ``(1, 2)``.
    ims_files:
        Mapping ``camera number -> .ims stream path``.  When empty the
        streams are looked up as ``Camera<N>-<M>.ims`` inside
        :attr:`data_root`.
    mark_table:
        DaVis ``MarkPositionTable.xml`` export (calibration marks).
    davis_calibration:
        DaVis ``Calibration.xml`` export (used only to report the
        reference reprojection RMS).
    sensor_size:
        ``(nx, ny)`` of the camera sensor in pixels.
    sheet_z:
        Light-sheet plane in calibration world coordinates (mm).
    pair_index:
        Index of the ``.ims`` frame pair to process.
    output_dir:
        Directory that receives the intermediate arrays and figures.
    disparity_passes:
        ``(window, overlap)`` pairs for the disparity estimation, coarse
        to fine.
    self_cal_iterations:
        Number of disparity self-calibration iterations on the finest pass.
    piv_window, piv_overlap:
        Interrogation window and overlap of the 2C PIV pass.
    model:
        Camera model used for the stereo pair, see :data:`MODEL_CHOICES`.
    grid_stride:
        Pixel stride used when back-projecting the common world grid.
    """

    data_root: Path
    cameras: tuple[int, ...] = (1, 2)
    ims_files: Mapping[int, Path] = field(default_factory=dict)
    mark_table: Path | None = None
    davis_calibration: Path | None = None
    sensor_size: tuple[int, int] = (2432, 2048)
    sheet_z: float = 0.0
    pair_index: int = 0
    output_dir: Path | None = None
    disparity_passes: tuple[tuple[int, int], ...] = ((128, 64), (64, 32))
    self_cal_iterations: int = 3
    piv_window: int = 64
    piv_overlap: int = 32
    model: str = "poly"
    grid_stride: int = 4

    def __post_init__(self) -> None:
        if len(self.cameras) < 2:
            raise ValueError(f"stereo needs at least two cameras, got {self.cameras}")
        if self.model not in MODEL_CHOICES:
            raise ValueError(f"model must be one of {MODEL_CHOICES}, got {self.model!r}")
        if not self.disparity_passes:
            raise ValueError("at least one disparity pass is required")

    # ------------------------------------------------------------------
    # derived paths
    # ------------------------------------------------------------------
    @property
    def out_dir(self) -> Path:
        """Directory that receives outputs (created on demand)."""
        return self.output_dir if self.output_dir is not None else self.data_root / "stereo_out"

    def ensure_out(self, sub: str = "") -> Path:
        """Return (and create) the output directory, optionally a sub-folder."""
        target = self.out_dir / sub if sub else self.out_dir
        target.mkdir(parents=True, exist_ok=True)
        return target

    def stream_path(self, camera: int) -> Path:
        """Path of the ``.ims`` stream of ``camera``."""
        if camera in self.ims_files:
            return Path(self.ims_files[camera])
        matches = sorted(self.data_root.glob(f"Camera{camera}-*.ims"))
        if not matches:
            raise FileNotFoundError(
                f"no .ims stream for camera {camera} in {self.data_root}"
            )
        # the stream dwarfs the small index file, keep the largest
        return max(matches, key=lambda p: p.stat().st_size)

    def with_data_root(self, root: str | os.PathLike[str]) -> StereoConfig:
        """Return a copy that reads from another data root."""
        return replace(self, data_root=Path(root))

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, data_root: str | os.PathLike[str] | None = None) -> StereoConfig:
        """Build a configuration from ``OPENPIV_STEREO_*`` environment variables.

        Recognised variables: ``DATA_ROOT``, ``CAMERAS``, ``IMS_FILES`` (a
        ``1=path,2=path`` mapping), ``MARK_TABLE``, ``DAVIS_CALIBRATION``,
        ``SENSOR_SIZE`` (``"nx,ny"``), ``SHEET_Z``, ``PAIR_INDEX``,
        ``OUTPUT_DIR``, ``SELF_CAL_ITERATIONS``, ``PIV_WINDOW``,
        ``PIV_OVERLAP`` and ``MODEL``.
        """
        root = Path(
            data_root
            if data_root is not None
            else _env_path("DATA_ROOT", Path.cwd())
        ).expanduser()

        ims_files: dict[int, Path] = {}
        raw = os.environ.get(ENV_PREFIX + "IMS_FILES", "")
        for item in raw.split(","):
            if not item.strip():
                continue
            key, _, value = item.partition("=")
            ims_files[int(key.strip())] = Path(value.strip()).expanduser()

        sensor = os.environ.get(ENV_PREFIX + "SENSOR_SIZE", "")
        if sensor:
            parts = [int(p) for p in sensor.replace(",", " ").split()]
            if len(parts) != 2:
                raise ValueError(f"{ENV_PREFIX}SENSOR_SIZE must be 'nx,ny', got {sensor!r}")
            sensor_size = (parts[0], parts[1])
        else:
            sensor_size = (2432, 2048)

        return cls(
            data_root=root,
            cameras=_env_cameras("CAMERAS", (1, 2)),
            ims_files=ims_files,
            mark_table=_env_path("MARK_TABLE"),
            davis_calibration=_env_path("DAVIS_CALIBRATION"),
            sensor_size=sensor_size,
            sheet_z=_env_float("SHEET_Z", 0.0),
            pair_index=_env_int("PAIR_INDEX", 0),
            output_dir=_env_path("OUTPUT_DIR"),
            self_cal_iterations=_env_int("SELF_CAL_ITERATIONS", 3),
            piv_window=_env_int("PIV_WINDOW", 64),
            piv_overlap=_env_int("PIV_OVERLAP", 32),
            model=os.environ.get(ENV_PREFIX + "MODEL", "poly"),
        )

    @classmethod
    def from_args(
        cls, argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
    ) -> StereoConfig:
        """Build a configuration from a command line (see ``__main__``)."""
        import argparse

        parser = argparse.ArgumentParser(
            prog="openpiv.examples.stereo",
            description="Stereo-PIV example: calibrate, dewarp, disparity, 3C.",
        )
        parser.add_argument("--data-root", type=Path, default=None, help="directory with the .ims streams")
        parser.add_argument("--cameras", type=str, default=None, help="stereo pair, e.g. '1,2'")
        parser.add_argument("--ims-files", type=str, default=None, help="mapping '1=path,2=path'")
        parser.add_argument("--mark-table", type=Path, default=None, help="DaVis MarkPositionTable.xml")
        parser.add_argument("--davis-calibration", type=Path, default=None, help="DaVis Calibration.xml")
        parser.add_argument("--sensor-size", type=str, default=None, help="sensor size 'nx,ny'")
        parser.add_argument("--sheet-z", type=float, default=None, help="light sheet plane in mm")
        parser.add_argument("--pair-index", type=int, default=None, help="frame pair to process")
        parser.add_argument("--output-dir", type=Path, default=None, help="output directory")
        parser.add_argument("--model", choices=MODEL_CHOICES, default=None, help="camera model")
        parser.add_argument("--piv-window", type=int, default=None)
        parser.add_argument("--piv-overlap", type=int, default=None)
        parser.add_argument("--self-cal-iterations", type=int, default=None)
        args = parser.parse_args(list(argv) if argv is not None else None)

        if env is not None:
            for key, value in env.items():
                os.environ[ENV_PREFIX + key] = value

        config = cls.from_env(args.data_root)
        updates: dict[str, object] = {}
        if args.cameras:
            updates["cameras"] = tuple(int(c) for c in args.cameras.replace(",", " ").split())
        if args.ims_files:
            mapping: dict[int, Path] = {}
            for item in args.ims_files.split(","):
                if not item.strip():
                    continue
                key, _, value = item.partition("=")
                mapping[int(key.strip())] = Path(value.strip()).expanduser()
            updates["ims_files"] = mapping
        if args.mark_table is not None:
            updates["mark_table"] = args.mark_table
        if args.davis_calibration is not None:
            updates["davis_calibration"] = args.davis_calibration
        if args.sensor_size:
            parts = [int(p) for p in args.sensor_size.replace(",", " ").split()]
            updates["sensor_size"] = (parts[0], parts[1])
        if args.sheet_z is not None:
            updates["sheet_z"] = args.sheet_z
        if args.pair_index is not None:
            updates["pair_index"] = args.pair_index
        if args.output_dir is not None:
            updates["output_dir"] = args.output_dir
        if args.model is not None:
            updates["model"] = args.model
        if args.piv_window is not None:
            updates["piv_window"] = args.piv_window
        if args.piv_overlap is not None:
            updates["piv_overlap"] = args.piv_overlap
        if args.self_cal_iterations is not None:
            updates["self_cal_iterations"] = args.self_cal_iterations
        return replace(config, **updates) if updates else config
