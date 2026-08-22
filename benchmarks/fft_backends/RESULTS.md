# FFT backend exploration (branch `explore/fft-backends`)

Context: v0.25.5 switched `openpiv.pyprocess.fft_correlate_images` from
`numpy.fft` to `scipy.fft`, a ~2-3x win with no new dependency (see
CHANGES.txt). This branch checks whether `pyFFTW`, `rocket-fft` (numba), or
a hand-rolled Cython FFT wrapper could beat `scipy.fft` by enough to justify
a new (and, for pyFFTW/Cython+FFTW, compiled) dependency.

## Setup

Windows 11, 8 logical cores. Benchmarked on the same batched-window shapes
`fft_correlate_images` actually produces (many small windows stacked along
axis 0, FFT over the last two axes) — not a generic single large-array FFT
benchmark, since that's not our access pattern.

## `bench_single_call.py` — cold-ish, one-shot calls

| shape | numpy.fft | scipy.fft | pyfftw (ESTIMATE, cached) | rocket-fft (numba) |
|---|---|---|---|---|
| (2145,63,63) finest pass | 813ms | 421ms | 399ms (1.05x) | 1088ms (0.39x) |
| (660,31,31) windowsize=32 | 170ms | 47ms | 48ms (0.97x) | 107ms (0.44x) |
| (143,127,127) windowsize=64 padded | 661ms | 353ms | 399ms (0.88x) | 756ms (0.47x) |

All outputs numerically match (`np.allclose`, atol=1e-6).

- **pyFFTW with default (ESTIMATE) planning is a wash against `scipy.fft`** —
  sometimes marginally faster, sometimes slower, never a clear win.
- **rocket-fft (numba-jitted `numpy.fft`) is consistently slower**, 0.4-0.5x.
  Numba's FFT implementation doesn't compete with scipy's batched C++
  (`duccfft`) backend for this shape of workload. Not worth pursuing.

## `bench_repeated_calls.py` — many calls at a fixed shape (batch job)

Simulates the realistic case: a batch run processes many image pairs, so the
same window shape recurs many times per pass. This is the scenario where
FFTW's expensive `MEASURE`/wisdom planning is supposed to pay for itself.

At (660,31,31), 40 repeated calls:

- `scipy.fft`: 55.5 ms/call
- `pyfftw` (ESTIMATE, cached): 54.4 ms/call (1.02x)
- `pyfftw` (`FFTW_MEASURE`, plan reused): **921 ms one-time plan cost**, then
  51.8 ms/call steady-state (1.07x vs scipy.fft)
- `pyfftw` (`FFTW_MEASURE`) amortized over just 40 calls: **0.74x** — net
  *slower* than scipy.fft, because the one-time plan-build cost dominates.

Even in the best case (steady-state after the plan is built), pyFFTW is only
~7% faster than scipy.fft. Breaking even on the 921ms `MEASURE` planning
cost alone takes roughly 250+ image pairs at this window size — and a real
multipass run uses several *different* window shapes per image pair, so
each shape would need its own amortized plan.

## Conclusion

`scipy.fft` (already merged, v0.25.5) is the best cost/benefit choice:

- Beats `numpy.fft` by 2-3x, `rocket-fft` by ~2x, with **zero new
  dependencies** (scipy is already required).
- `pyFFTW` offers at best a ~7% steady-state edge, erased by its own
  planning overhead unless a batch run reuses one window shape hundreds of
  times — not representative of typical multipass PIV runs (several
  different window sizes per pass).
- A hand-rolled Cython FFT wrapper would, at best, match pyFFTW (same
  underlying FFTW library, same planning cost) — i.e. still short of
  scipy.fft's real-world advantage, while adding a C build dependency this
  project deliberately removed (see `CLAUDE.md`: "no Cython extensions").
  **Not built** — the benchmark numbers above rule it out before writing any
  Cython.

**Recommendation: do not merge this branch's dependencies into `pyproject.toml`.**
Keep `scipy.fft` as shipped in v0.25.5. `pyfftw`/`rocket-fft`/`numba` were
installed only in this branch's throwaway `uv pip install` for benchmarking
and are not added to `pyproject.toml`.
