# Implementation Plan: Stereo-PIV calibration 1-4

## Overview
Get stereoscopic calibration working end-to-end on top of PR #324 (`openpiv.calibration`): (1) stereo pair calibration, (2) dewarp to common world grid, (3) disparity/self-calibration correction, (4) 3C reconstruction. Fixes known `get_image_mapping` bugs blocking stereo.

## Architecture Decisions
- New model-agnostic submodule `openpiv/calibration/stereo.py` using duck-typing (`project_points`, `project_to_z`, `resolution`, `dtype`). Works with DLT/pinhole/poly cameras. Rationale: avoids per-model forks, reuses existing triangulation.
- Numerical Jacobian (central differences) for 3C reconstruction instead of analytic per-model derivatives. Rationale: single code path, exact enough for PIV (eps=1e-6 world units), analytic poly derivatives kept as fast path later.
- Common world grid = intersection (not union) of camera back-projections at sheet z. Rationale: stereo needs overlap; union creates extrapolated borders.
- Disparity via existing `pyprocess.extended_search_area_piv` on dewarped pair (same-t). Rationale: reuses validated correlation, no new dependency.
- Wieneke self-calibration: symmetric half-correction + iterate. Rationale: literature standard, converges in 2-3 iters to <0.1px.
- Keep `calib_utils.get_image_mapping(cam)` backward-compatible (ignore legacy func args), delegate to new code. Fixes: Y-bounds used `world_x` instead of `world_y`, scale formula assumed centered target, off-by-`Scale` linspace start.

## Task List

### Phase 1: Foundation (fix dewarp)
- [ ] Task 1: RED — failing test for `get_image_mapping` roundtrip + common grid overlap
- [ ] Task 2: Fix `get_image_mapping`, add `get_common_world_grid/get_dewarp_mapping/dewarp_image` (or in stereo.py + delegate)

### Checkpoint: Foundation
- [ ] calib tests pass, dewarped checker/synthetic dots reproject <0.5px

### Phase 2: Stereo core
- [ ] Task 3: `triangulate`, `stereo_reconstruct_3c` (numerical J, lstsq per point) + tests on synthetic poly/DLT cams
- [ ] Task 4: `estimate_disparity` + `self_calibration` iterative correction + tests (known shift recovered, residual <0.1px)

### Checkpoint: Core
- [ ] `test_calib_stereo.py` 1-4 green, full calib suite green

### Phase 3: Polish
- [ ] Task 5: Export `stereo` in docs/__init__, example snippet, run full `openpiv` suite

### Checkpoint: Complete
- [ ] All acceptance criteria met, ready for review

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| `project_to_z` per-model inconsistencies | High | Test roundtrip per model, numerical fallback |
| Disparity correlation peak-locking on sparse particles | Med | Ensemble + validation + smoothn, synthetic dense dots in tests |
| Large extrapolation outside target | Med | Intersection grid + mask, warn on out-of-bounds |

## Open Questions
- Preferred default stereo model? Proposing `poly_model` (Soloff) — confirm.
- `windef` dewarp integration deferred? Yes — stereo.py first, windef hook follow-up.
