.. _stereo_calibration:

Stereo calibration and stereo-PIV
=================================

This page explains how two (u,v) displacement fields become one (U,V,W)
velocity field, how the calibration enters, and what disparity correction
fixes. The implementation lives in ``openpiv.calibration.stereo``.

What each camera measures
-------------------------

Per camera, ordinary 2C PIV on its **dewarped** frames gives one pixel
displacement per grid point: left :math:`(u_L, v_L)`, right :math:`(u_R, v_R)`.
That is **4 numbers** per point for **3** velocity components :math:`(U,V,W)` —
the redundancy is what makes stereo possible (and what exposes bad calibration).

Calibration = the projection function
--------------------------------------

Calibration gives, per camera, a world → image map :math:`F`
(:meth:`camera.project_points <openpiv.calibration.poly_model.camera>`,
here a Soloff 3rd-order polynomial):

.. math::
   F_{cam}(X, Y, Z) = (x, y)

A world displacement :math:`(U,V,W)\,dt` appears in the images through the
**Jacobian** :math:`J` of :math:`F` at that grid point:

.. math::
   \begin{pmatrix} u_L \\ v_L \\ u_R \\ v_R \end{pmatrix}
   =
   \underbrace{
   \begin{pmatrix}
   \partial F_L/\partial X & \partial F_L/\partial Y & \partial F_L/\partial Z \\
   \partial F_R/\partial X & \partial F_R/\partial Y & \partial F_R/\partial Z
   \end{pmatrix}
   }_{J\;(4\times3)}
   \begin{pmatrix} U \\ V \\ W \end{pmatrix} dt

Per grid point we solve the tiny least-squares system :math:`J\,V = d`
(:func:`openpiv.calibration.stereo.reconstruct_3c`). In-plane :math:`(U,V)`
comes mostly from the **common-mode** motion both cameras see; :math:`W`
comes from the **difference** (left sees it shift one way, right the other) —
so :math:`W` needs two views and is the most calibration-sensitive component.

What disparity correction fixes (two effects)
--------------------------------------------

The calibration was made on the **plate** — a different plane and time than the
laser sheet — so dewarped images carry a systematic left-vs-right offset :math:`D`:

- **Measurement frame.** :math:`(u_L,v_L)` and :math:`(u_R,v_R)` must refer to
  the *same* physical point. Correcting the maps (resample each by :math:`\mp D/2`
  with :func:`openpiv.calibration.stereo.correct_maps`) re-aligns the frames
  *before* PIV.
- **Linearization point.** :math:`J` is evaluated at the grid point, and under
  oblique (Scheimpflug) angles it varies fast — :math:`\partial/\partial Z`
  reaches thousands of px/mm. Uncorrected maps evaluate :math:`J` at the wrong place.

Median :math:`|D|` after correction is the single-number quality gate for
everything downstream: if it does not drop, do not trust the 3C field.

Workflow
--------

1. Calibrate both cameras from a dual-plane target
   (``poly_model.camera(...).minimize_params(obj_xyz, img_xy)``).
2. Build the common world grid at the sheet plane and dewarp
   (``get_common_world_grid`` → ``get_dewarp_mapping`` → ``dewarp_image``).
3. Per frame pair: estimate disparity :math:`D = R - L` on the dewarped
   partner images — separately for pulse A and pulse B, averaged —
   and correct the maps (``estimate_disparity`` → ``correct_maps``).
   The laser sheet drifts, so **every pair gets its own correction**.
4. 2C PIV per dewarped camera (``pyprocess.extended_search_area_piv``),
   validate, then ``reconstruct_3c`` → :math:`(U,V,W)`.
5. Batch: repeat 3–4 over the pair range (maps are ~150 MB each — save
   vectors and disparity statistics, not maps).

API
---

The ``openpiv.calibration.stereo`` module
-----------------------------------------

.. automodule:: openpiv.calibration.stereo
    :members:
