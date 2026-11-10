# Stereo-PIV example

A worked example that drives `openpiv.calibration.stereo` over a DaVis
stereo data set: it calibrates the pair, rectifies both cameras onto a
common world grid, corrects the residual disparity (self-calibration),
runs the 2C PIV per camera and reconstructs the 3C velocity field.

```
openpiv.examples.stereo
├── config.py     StereoConfig  — every path and parameter in one place
├── davis.py      readers for the DaVis XML exports
├── frames.py     .ims / .im7 access (optional readim7 dependency)
└── pipeline.py   the steps, as plain-array functions
```

## Data set

The example expects a DaVis acquisition directory:

```
run/
├── Camera1-1-001.ims     raw 12-bit packed frame pairs
├── Camera2-1-002.ims
└── calibration/
    ├── MarkPositionTable.xml   calibration marks (world + raw pixel)
    └── Calibration.xml         DaVis' own fit, for reference
```

`.ims` streams are read with
[readim7](https://github.com/alexlib/readim7) (optional dependency):

```bash
pip install "readim7 @ git+https://github.com/alexlib/readim7.git"
```

The stream geometry normally comes from the small index file DaVis writes
next to the stream (`Camera<N>-0.ims`).  When that index is not a sibling
of the stream, pass the sensor size explicitly with `--sensor-size`.

## Usage

```bash
python -m openpiv.examples.stereo --data-root run \
    --mark-table run/calibration/MarkPositionTable.xml \
    --davis-calibration run/calibration/Calibration.xml \
    --cameras 1,2 --sensor-size 2432,2048 --sheet-z 0 all
```

Single steps (each writes into the output directory):

| Step | Does |
|---|---|
| `info` | print the stream layout and the resolved configuration |
| `calibrate` | fit the stereo pair, report the RMS against the DaVis fit |
| `dewarp` | build the common world grid and the dewarp mappings |
| `disparity` | measure the particle disparity, self-calibrate the mappings |
| `piv` | 2C PIV per camera + 3C reconstruction |

Every option can also be set through the environment, e.g.
`OPENPIV_STEREO_DATA_ROOT`, `OPENPIV_STEREO_CAMERAS`,
`OPENPIV_STEREO_SHEET_Z`, `OPENPIV_STEREO_MARK_TABLE`.

## From Python

```python
from openpiv.examples.stereo import (
    StereoConfig, parse_mark_position_table, fit_cameras,
    build_world_grid, build_dewarp_maps, run_2c_piv, reconstruct_3c,
)

config = StereoConfig.from_env()
marks = parse_mark_position_table(config.mark_table)
cams = fit_cameras(config.model, config.sensor_size, marks, config.cameras)
Xw, Yw, info = build_world_grid(cams, z=config.sheet_z)
maps = build_dewarp_maps(cams, Xw, Yw, z=config.sheet_z)
```

## Units

* world coordinates, calibration plate geometry and velocities are in **mm**;
* the 3C output is in **mm per frame** — multiply by `1e3 / dt[s]` for m/s;
* pixel displacements of the 2C PIV are in **pixels**;
* `sheet_z` is the light-sheet plane in calibration world coordinates; the
  dual-level plate used for the reference data has marks at `z = 0` and
  `z = -3` mm.

## Which frame the displacements are in

`reconstruct_3c()` expects **camera pixel** displacements, because that is
the frame the calibration Jacobian `d(image)/d(world)` lives in. Two
consequences:

* the **disparity** is measured on the *dewarped* images — there both
  cameras share the common world grid, so the particles match;
* the **2C PIV** for the 3C reconstruction runs on the *raw* images, and the
  partner field is sampled at the projected position of each particle
  (`backproject_to_plane` → `project_points` → `sample_velocity_field`).

Mixing the two frames (PIV on the rectified images, reconstruction in the
camera frame) biases the 3C field.

## Reference numbers

On the DaVis pitching-wing data set (upper stereo pair, cameras 1 and 2,
2432×2048, 300 pairs, dual-level LaVision 204-15 plate):

| Step | Result |
|---|---|
| Calibration, camera 1 | RMS **0.408 px** (DaVis: 0.396 px) over 213 marks |
| Calibration, camera 2 | RMS **0.369 px** (DaVis: 0.361 px) over 254 marks |
| Common world grid @ z=0 | 2432×2048 nodes, **0.0850 mm/px**, X[-104.6, 102.1], Y[-139.6, 45.1] |
| Disparity, before | median \|D\| = **25.3 px** (bulk shift −4.2, −24.0 px) |
| Disparity, after | median \|D\| = **2.7 px** |

The bulk shift is the offset between the calibration plane and the laser
sheet — exactly what the planar self-calibration corrects.

## Comparing with a DaVis vector field

`compare` takes the ``.vc7`` written by the DaVis StereoPIV module and
compares it with the reconstruction, on the grid of the reference:

```bash
python -m openpiv.examples.stereo --data-root run \
    --mark-table run/calibration/MarkPositionTable.xml \
    --pair-index 298 --frame-dt 1.45e-3 \
    --davis-vector-field run/StereoPIV/B00299.vc7 compare
```

Two conventions have to be respected:

* the DaVis components live in the **corrected image** frame, where the
  image row grows downwards while the world Y axis points upwards — the
  vertical component therefore has to be **flipped** to get the world
  velocity (`VectorField.v_world` does that);
* the velocities are in m/s, ours in mm per frame — `frame_dt` (the
  frame interval in seconds) does the conversion.

With those in place, the reference and the reconstruction agree on the
reference data set:

| Component | ours | DaVis | bias | RMSE | correlation |
|---|---|---|---|---|---|
| U | −0.0006 m/s | −0.0009 m/s | −0.008 | 0.121 | +0.35 |
| V | −0.0612 m/s | −0.0670 m/s | +0.006 | 0.082 | +0.56 |

## The cycle survey

`survey` measures the particle displacement of **every** frame pair, so
that you know how much signal a pair carries before comparing it with a
reference:

```bash
python -m openpiv.examples.stereo --data-root run --survey-crop 600,400,1200,1200 survey
```

On the reference data set (300 pairs, 1.45 ms per frame, so 0.44 s of
recording):

* the **vertical** motion is steady: median 1.3–1.6 px per frame, and its
  sign never changes along the record;
* the **horizontal** motion is concentrated in ~10 % of the field (near
  the wing): the median is 0.3–0.7 px, but the 90th percentile reaches
  1.9–2.8 px;
* the record spans 0.44 s, i.e. **22 % of one 0.5 Hz wing cycle** — there
  is no "fast phase" to be found in it, the horizontal signal varies by
  less than a factor of two.

That is why a comparison at a single pair can look meaningless: in the
quiet regions the horizontal velocity is at the noise level, so its sign
is random and it anti-correlates with any reference. Judge the
horizontal component in the active region (the 90th percentile of the
displacement), not over the whole field.


