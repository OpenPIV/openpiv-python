use num_complex::Complex;
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, ToPyArray,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};
use std::borrow::Cow;
use std::sync::Arc;

/// Safely extracts or linearizes an ndarray view into a C-contiguous slice [N, H, W].
fn to_c_contiguous<'a>(arr: &'a numpy::ndarray::ArrayView3<'a, f64>) -> Cow<'a, [f64]> {
    if let Some(s) = arr.as_slice() {
        Cow::Borrowed(s)
    } else {
        let n = arr.shape()[0];
        let h = arr.shape()[1];
        let w = arr.shape()[2];
        let mut vec = Vec::with_capacity(n * h * w);
        for i in 0..n {
            for r in 0..h {
                for c in 0..w {
                    vec.push(arr[[i, r, c]]);
                }
            }
        }
        Cow::Owned(vec)
    }
}

/// Safely extracts or linearizes an ndarray 2D view into a C-contiguous slice [H, W].
fn to_c_contiguous_2d<'a>(arr: &'a numpy::ndarray::ArrayView2<'a, f64>) -> Cow<'a, [f64]> {
    if let Some(s) = arr.as_slice() {
        Cow::Borrowed(s)
    } else {
        let h = arr.shape()[0];
        let w = arr.shape()[1];
        let mut vec = Vec::with_capacity(h * w);
        for r in 0..h {
            for c in 0..w {
                vec.push(arr[[r, c]]);
            }
        }
        Cow::Owned(vec)
    }
}

/// Engine for Circular 2D cross-correlation (OpenPIV standard mode)
struct CircularEngine2D {
    h: usize,
    w: usize,
    w_freq: usize,
    r2c_row: Arc<dyn RealToComplex<f64>>,
    c2r_row: Arc<dyn ComplexToReal<f64>>,
    c2c_col_fwd: Arc<dyn Fft<f64>>,
    c2c_col_inv: Arc<dyn Fft<f64>>,
}

struct CircularScratch {
    freq_a: Vec<Complex<f64>>,
    freq_b: Vec<Complex<f64>>,
    row_scratch_fwd: Vec<Complex<f64>>,
    col_scratch_fwd: Vec<Complex<f64>>,
    row_scratch_inv: Vec<Complex<f64>>,
    col_scratch_inv: Vec<Complex<f64>>,
    col_buf: Vec<Complex<f64>>,
    in_row_copy: Vec<f64>,
    temp_real: Vec<f64>,
}

impl CircularEngine2D {
    fn new(h: usize, w: usize) -> Self {
        let mut real_planner = RealFftPlanner::<f64>::new();
        let r2c_row = real_planner.plan_fft_forward(w);
        let c2r_row = real_planner.plan_fft_inverse(w);

        let mut c2c_planner = FftPlanner::<f64>::new();
        let c2c_col_fwd = c2c_planner.plan_fft_forward(h);
        let c2c_col_inv = c2c_planner.plan_fft_inverse(h);

        let w_freq = w / 2 + 1;

        Self {
            h,
            w,
            w_freq,
            r2c_row,
            c2r_row,
            c2c_col_fwd,
            c2c_col_inv,
        }
    }

    fn create_scratch(&self) -> CircularScratch {
        CircularScratch {
            freq_a: vec![Complex::new(0.0, 0.0); self.h * self.w_freq],
            freq_b: vec![Complex::new(0.0, 0.0); self.h * self.w_freq],
            row_scratch_fwd: self.r2c_row.make_scratch_vec(),
            col_scratch_fwd: vec![Complex::new(0.0, 0.0); self.c2c_col_fwd.get_inplace_scratch_len()],
            row_scratch_inv: self.c2r_row.make_scratch_vec(),
            col_scratch_inv: vec![Complex::new(0.0, 0.0); self.c2c_col_inv.get_inplace_scratch_len()],
            col_buf: vec![Complex::new(0.0, 0.0); self.h],
            in_row_copy: vec![0.0; self.w],
            temp_real: vec![0.0; self.h * self.w],
        }
    }

    fn correlate_pair(
        &self,
        window_a: &[f64],
        window_b: &[f64],
        out_slice: &mut [f64],
        normalized_correlation: bool,
        scratch: &mut CircularScratch,
    ) {
        let h = self.h;
        let w = self.w;
        let w_freq = self.w_freq;

        // 1. Forward 2D Real FFT for window A
        for r in 0..h {
            scratch.in_row_copy.copy_from_slice(&window_a[r * w..(r + 1) * w]);
            let out_row = &mut scratch.freq_a[r * w_freq..(r + 1) * w_freq];
            let _ = self.r2c_row.process_with_scratch(&mut scratch.in_row_copy, out_row, &mut scratch.row_scratch_fwd);
        }
        for c in 0..w_freq {
            for r in 0..h {
                scratch.col_buf[r] = scratch.freq_a[r * w_freq + c];
            }
            self.c2c_col_fwd
                .process_with_scratch(&mut scratch.col_buf, &mut scratch.col_scratch_fwd);
            for r in 0..h {
                scratch.freq_a[r * w_freq + c] = scratch.col_buf[r];
            }
        }

        // 2. Forward 2D Real FFT for window B
        for r in 0..h {
            scratch.in_row_copy.copy_from_slice(&window_b[r * w..(r + 1) * w]);
            let out_row = &mut scratch.freq_b[r * w_freq..(r + 1) * w_freq];
            let _ = self.r2c_row.process_with_scratch(&mut scratch.in_row_copy, out_row, &mut scratch.row_scratch_fwd);
        }
        for c in 0..w_freq {
            for r in 0..h {
                scratch.col_buf[r] = scratch.freq_b[r * w_freq + c];
            }
            self.c2c_col_fwd
                .process_with_scratch(&mut scratch.col_buf, &mut scratch.col_scratch_fwd);
            for r in 0..h {
                scratch.freq_b[r * w_freq + c] = scratch.col_buf[r];
            }
        }

        // 3. Frequency domain cross-correlation: Fa.conj() * Fb
        for i in 0..(h * w_freq) {
            scratch.freq_a[i] = scratch.freq_a[i].conj() * scratch.freq_b[i];
        }

        // 4. Inverse 2D Real FFT on freq_a
        for c in 0..w_freq {
            for r in 0..h {
                scratch.col_buf[r] = scratch.freq_a[r * w_freq + c];
            }
            self.c2c_col_inv
                .process_with_scratch(&mut scratch.col_buf, &mut scratch.col_scratch_inv);
            for r in 0..h {
                scratch.freq_a[r * w_freq + c] = scratch.col_buf[r];
            }
        }

        for r in 0..h {
            let in_freq_row = &mut scratch.freq_a[r * w_freq..(r + 1) * w_freq];
            // Enforce strictly 0 imaginary residuals at DC and Nyquist frequencies
            in_freq_row[0].im = 0.0;
            if w % 2 == 0 {
                in_freq_row[w_freq - 1].im = 0.0;
            }
            let out_real_row = &mut scratch.temp_real[r * w..(r + 1) * w];
            let _ = self.c2r_row.process_with_scratch(in_freq_row, out_real_row, &mut scratch.row_scratch_inv);
        }

        // 5. Normalization scale factor + 2D fftshift
        let fft_norm = 1.0 / ((h * w) as f64);
        let scale = if normalized_correlation {
            fft_norm * fft_norm
        } else {
            fft_norm
        };

        let shift_r = h / 2;
        let shift_c = w / 2;

        for r in 0..h {
            let target_r = (r + shift_r) % h;
            for c in 0..w {
                let target_c = (c + shift_c) % w;
                out_slice[target_r * w + target_c] = scratch.temp_real[r * w + c] * scale;
            }
        }
    }
}

struct LinearScratch {
    freq_a: Vec<Complex<f64>>,
    freq_b: Vec<Complex<f64>>,
    row_scratch_fwd: Vec<Complex<f64>>,
    col_scratch_fwd: Vec<Complex<f64>>,
    row_scratch_inv: Vec<Complex<f64>>,
    col_scratch_inv: Vec<Complex<f64>>,
    col_buf: Vec<Complex<f64>>,
    in_row_pad: Vec<f64>,
    temp_real: Vec<f64>,
}

/// Engine for Full Linear 2D cross-correlation (matches scipy.signal.correlate mode='full')
struct FullEngine2D {
    win_h: usize,
    win_w: usize,
    out_h: usize,
    out_w: usize,
    fft_h: usize,
    fft_w: usize,
    w_freq: usize,
    r2c_row: Arc<dyn RealToComplex<f64>>,
    c2r_row: Arc<dyn ComplexToReal<f64>>,
    c2c_col_fwd: Arc<dyn Fft<f64>>,
    c2c_col_inv: Arc<dyn Fft<f64>>,
}

impl FullEngine2D {
    fn new(win_h: usize, win_w: usize) -> Self {
        let out_h = 2 * win_h - 1;
        let out_w = 2 * win_w - 1;

        let fft_h = out_h.next_power_of_two();
        let fft_w = out_w.next_power_of_two();

        let mut real_planner = RealFftPlanner::<f64>::new();
        let r2c_row = real_planner.plan_fft_forward(fft_w);
        let c2r_row = real_planner.plan_fft_inverse(fft_w);

        let mut c2c_planner = FftPlanner::<f64>::new();
        let c2c_col_fwd = c2c_planner.plan_fft_forward(fft_h);
        let c2c_col_inv = c2c_planner.plan_fft_inverse(fft_h);

        let w_freq = fft_w / 2 + 1;

        Self {
            win_h,
            win_w,
            out_h,
            out_w,
            fft_h,
            fft_w,
            w_freq,
            r2c_row,
            c2r_row,
            c2c_col_fwd,
            c2c_col_inv,
        }
    }

    fn create_scratch(&self) -> LinearScratch {
        LinearScratch {
            freq_a: vec![Complex::new(0.0, 0.0); self.fft_h * self.w_freq],
            freq_b: vec![Complex::new(0.0, 0.0); self.fft_h * self.w_freq],
            row_scratch_fwd: self.r2c_row.make_scratch_vec(),
            col_scratch_fwd: vec![Complex::new(0.0, 0.0); self.c2c_col_fwd.get_inplace_scratch_len()],
            row_scratch_inv: self.c2r_row.make_scratch_vec(),
            col_scratch_inv: vec![Complex::new(0.0, 0.0); self.c2c_col_inv.get_inplace_scratch_len()],
            col_buf: vec![Complex::new(0.0, 0.0); self.fft_h],
            in_row_pad: vec![0.0; self.fft_w],
            temp_real: vec![0.0; self.fft_h * self.fft_w],
        }
    }

    fn correlate_pair(&self, window_a: &[f64], window_b: &[f64], out_slice: &mut [f64], scratch: &mut LinearScratch) {
        let win_h = self.win_h;
        let win_w = self.win_w;
        let out_h = self.out_h;
        let out_w = self.out_w;
        let fft_h = self.fft_h;
        let fft_w = self.fft_w;
        let w_freq = self.w_freq;

        // 1. Forward 2D Real FFT for padded window A
        for r in 0..fft_h {
            scratch.in_row_pad.fill(0.0);
            if r < win_h {
                scratch.in_row_pad[..win_w].copy_from_slice(&window_a[r * win_w..(r + 1) * win_w]);
            }
            let out_row = &mut scratch.freq_a[r * w_freq..(r + 1) * w_freq];
            let _ = self.r2c_row.process_with_scratch(&mut scratch.in_row_pad, out_row, &mut scratch.row_scratch_fwd);
        }
        for c in 0..w_freq {
            for r in 0..fft_h {
                scratch.col_buf[r] = scratch.freq_a[r * w_freq + c];
            }
            self.c2c_col_fwd
                .process_with_scratch(&mut scratch.col_buf, &mut scratch.col_scratch_fwd);
            for r in 0..fft_h {
                scratch.freq_a[r * w_freq + c] = scratch.col_buf[r];
            }
        }

        // 2. Forward 2D Real FFT for padded window B
        for r in 0..fft_h {
            scratch.in_row_pad.fill(0.0);
            if r < win_h {
                scratch.in_row_pad[..win_w].copy_from_slice(&window_b[r * win_w..(r + 1) * win_w]);
            }
            let out_row = &mut scratch.freq_b[r * w_freq..(r + 1) * w_freq];
            let _ = self.r2c_row.process_with_scratch(&mut scratch.in_row_pad, out_row, &mut scratch.row_scratch_fwd);
        }
        for c in 0..w_freq {
            for r in 0..fft_h {
                scratch.col_buf[r] = scratch.freq_b[r * w_freq + c];
            }
            self.c2c_col_fwd
                .process_with_scratch(&mut scratch.col_buf, &mut scratch.col_scratch_fwd);
            for r in 0..fft_h {
                scratch.freq_b[r * w_freq + c] = scratch.col_buf[r];
            }
        }

        // 3. Frequency domain cross-correlation: Fa * Fb.conj()
        // Matches scipy.signal.correlate(a, b, mode='full')
        for i in 0..(fft_h * w_freq) {
            scratch.freq_a[i] = scratch.freq_a[i] * scratch.freq_b[i].conj();
        }

        // 4. Inverse 2D Real FFT
        for c in 0..w_freq {
            for r in 0..fft_h {
                scratch.col_buf[r] = scratch.freq_a[r * w_freq + c];
            }
            self.c2c_col_inv
                .process_with_scratch(&mut scratch.col_buf, &mut scratch.col_scratch_inv);
            for r in 0..fft_h {
                scratch.freq_a[r * w_freq + c] = scratch.col_buf[r];
            }
        }

        for r in 0..fft_h {
            let in_freq_row = &mut scratch.freq_a[r * w_freq..(r + 1) * w_freq];
            in_freq_row[0].im = 0.0;
            if fft_w % 2 == 0 {
                in_freq_row[w_freq - 1].im = 0.0;
            }
            let out_real_row = &mut scratch.temp_real[r * fft_w..(r + 1) * fft_w];
            let _ = self.c2r_row.process_with_scratch(in_freq_row, out_real_row, &mut scratch.row_scratch_inv);
        }

        let scale = 1.0 / ((fft_h * fft_w) as f64);

        // 5. Crop and unshift into out_slice
        for r in 0..out_h {
            let r_idx = (r as isize - (win_h as isize - 1)).rem_euclid(fft_h as isize) as usize;
            for c in 0..out_w {
                let c_idx = (c as isize - (win_w as isize - 1)).rem_euclid(fft_w as isize) as usize;
                out_slice[r * out_w + c] = scratch.temp_real[r_idx * fft_w + c_idx] * scale;
            }
        }
    }
}

/// Circular batched cross-correlation (OpenPIV standard mode, returns (N, H, W))
#[pyfunction]
#[pyo3(signature = (windows_a, windows_b, normalized_correlation=true))]
fn fft_correlate_circular<'py>(
    py: Python<'py>,
    windows_a: PyReadonlyArray3<f64>,
    windows_b: PyReadonlyArray3<f64>,
    normalized_correlation: bool,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let a = windows_a.as_array();
    let b = windows_b.as_array();

    // Safe error handling: validate shapes
    if a.shape() != b.shape() {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: windows_a has shape {:?}, but windows_b has shape {:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.shape()[0] == 0 || a.shape()[1] == 0 || a.shape()[2] == 0 {
        return Err(PyValueError::new_err(
            "Window arrays must have non-zero dimensions [N > 0, H > 0, W > 0]"
        ));
    }

    let num_wins = a.shape()[0];
    let win_h = a.shape()[1];
    let win_w = a.shape()[2];
    let win_size = win_h * win_w;

    let engine = CircularEngine2D::new(win_h, win_w);

    // Safe C-contiguous data access (handles any strides, transpositions, slices)
    let a_cow = to_c_contiguous(&a);
    let b_cow = to_c_contiguous(&b);

    let mut final_results = vec![0.0; num_wins * win_size];

    // Release GIL and compute across threadpool using Rayon with zero-allocation thread-local scratch
    py.allow_threads(|| {
        final_results
            .par_chunks_exact_mut(win_size)
            .enumerate()
            .for_each_init(
                || engine.create_scratch(),
                |scratch, (i, out_slice)| {
                    let w_a = &a_cow[i * win_size..(i + 1) * win_size];
                    let w_b = &b_cow[i * win_size..(i + 1) * win_size];
                    engine.correlate_pair(w_a, w_b, out_slice, normalized_correlation, scratch);
                },
            );
    });

    Ok(final_results
        .to_pyarray(py)
        .reshape([num_wins, win_h, win_w])?)
}

/// Full linear batched cross-correlation (matches scipy.signal.correlate(..., mode='full'), returns (N, 2H-1, 2W-1))
#[pyfunction]
#[pyo3(signature = (windows_a, windows_b))]
fn fast_batch_cross_correlation<'py>(
    py: Python<'py>,
    windows_a: PyReadonlyArray3<f64>,
    windows_b: PyReadonlyArray3<f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let a = windows_a.as_array();
    let b = windows_b.as_array();

    // Safe error handling: validate shapes
    if a.shape() != b.shape() {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: windows_a has shape {:?}, but windows_b has shape {:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.shape()[0] == 0 || a.shape()[1] == 0 || a.shape()[2] == 0 {
        return Err(PyValueError::new_err(
            "Window arrays must have non-zero dimensions [N > 0, H > 0, W > 0]"
        ));
    }

    let num_wins = a.shape()[0];
    let win_h = a.shape()[1];
    let win_w = a.shape()[2];

    let out_h = 2 * win_h - 1;
    let out_w = 2 * win_w - 1;
    let out_win_size = out_h * out_w;
    let in_win_size = win_h * win_w;

    let engine = FullEngine2D::new(win_h, win_w);

    // Safe C-contiguous data access (handles any strides, transpositions, slices)
    let a_cow = to_c_contiguous(&a);
    let b_cow = to_c_contiguous(&b);

    let mut final_results = vec![0.0; num_wins * out_win_size];

    py.allow_threads(|| {
        final_results
            .par_chunks_exact_mut(out_win_size)
            .enumerate()
            .for_each_init(
                || engine.create_scratch(),
                |scratch, (i, out_slice)| {
                    let w_a = &a_cow[i * in_win_size..(i + 1) * in_win_size];
                    let w_b = &b_cow[i * in_win_size..(i + 1) * in_win_size];
                    engine.correlate_pair(w_a, w_b, out_slice, scratch);
                },
            );
    });

    Ok(final_results
        .to_pyarray(py)
        .reshape([num_wins, out_h, out_w])?)
}

/// Linear batched cross-correlation alias (same as fast_batch_cross_correlation)
#[pyfunction]
#[pyo3(signature = (windows_a, windows_b))]
fn fft_correlate_linear<'py>(
    py: Python<'py>,
    windows_a: PyReadonlyArray3<f64>,
    windows_b: PyReadonlyArray3<f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    fast_batch_cross_correlation(py, windows_a, windows_b)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SubpixelMethod {
    Gaussian,
    Centroid,
    Parabolic,
}

impl SubpixelMethod {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "gaussian" => Ok(SubpixelMethod::Gaussian),
            "centroid" => Ok(SubpixelMethod::Centroid),
            "parabolic" => Ok(SubpixelMethod::Parabolic),
            other => Err(format!("Method not implemented {other}")),
        }
    }
}

pub fn subpixel_peak_position_2d(
    corr: &[f64],
    h: usize,
    w: usize,
    method: SubpixelMethod,
) -> (f64, f64) {
    if h < 3 || w < 3 {
        return (f64::NAN, f64::NAN);
    }

    // 1. Find argmax (peak1_i, peak1_j)
    let mut best_idx = 0;
    let mut best_val = corr[0];
    for (idx, &val) in corr.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }

    let peak1_i = best_idx / w;
    let peak1_j = best_idx % w;

    // 2. Check border condition
    if peak1_i == 0 || peak1_i == h - 1 || peak1_j == 0 || peak1_j == w - 1 {
        return (f64::NAN, f64::NAN);
    }

    // 3. Extract 5-point cross with eps = 1e-7
    const EPS: f64 = 1e-7;
    let c = corr[peak1_i * w + peak1_j] + EPS;
    let cl = corr[(peak1_i - 1) * w + peak1_j] + EPS;
    let cr = corr[(peak1_i + 1) * w + peak1_j] + EPS;
    let cd = corr[peak1_i * w + (peak1_j - 1)] + EPS;
    let cu = corr[peak1_i * w + (peak1_j + 1)] + EPS;

    // 4. Fallback if any point < 0
    let effective_method = if method == SubpixelMethod::Gaussian
        && (c < 0.0 || cl < 0.0 || cr < 0.0 || cd < 0.0 || cu < 0.0)
    {
        SubpixelMethod::Parabolic
    } else {
        method
    };

    match effective_method {
        SubpixelMethod::Centroid => {
            let sum_row = cl + c + cr;
            let sum_col = cd + c + cu;
            let pi = peak1_i as f64;
            let pj = peak1_j as f64;
            let sub_i = if sum_row != 0.0 {
                ((pi - 1.0) * cl + pi * c + (pi + 1.0) * cr) / sum_row
            } else {
                f64::NAN
            };
            let sub_j = if sum_col != 0.0 {
                ((pj - 1.0) * cd + pj * c + (pj + 1.0) * cu) / sum_col
            } else {
                f64::NAN
            };
            (sub_i, sub_j)
        }
        SubpixelMethod::Gaussian => {
            let nom1 = cl.ln() - cr.ln();
            let den1 = 2.0 * cl.ln() - 4.0 * c.ln() + 2.0 * cr.ln();
            let nom2 = cd.ln() - cu.ln();
            let den2 = 2.0 * cd.ln() - 4.0 * c.ln() + 2.0 * cu.ln();

            let offset_i = if den1 != 0.0 { nom1 / den1 } else { 0.0 };
            let offset_j = if den2 != 0.0 { nom2 / den2 } else { 0.0 };

            (peak1_i as f64 + offset_i, peak1_j as f64 + offset_j)
        }
        SubpixelMethod::Parabolic => {
            let den1 = 2.0 * cl - 4.0 * c + 2.0 * cr;
            let den2 = 2.0 * cd - 4.0 * c + 2.0 * cu;

            let offset_i = if den1 != 0.0 { (cl - cr) / den1 } else { 0.0 };
            let offset_j = if den2 != 0.0 { (cd - cu) / den2 } else { 0.0 };

            (peak1_i as f64 + offset_i, peak1_j as f64 + offset_j)
        }
    }
}

/// Find subpixel approximation of the correlation peak for a single 2D correlation map.
#[pyfunction]
#[pyo3(signature = (corr, subpixel_method=None))]
fn find_subpixel_peak_position(
    corr: PyReadonlyArray2<f64>,
    subpixel_method: Option<&str>,
) -> PyResult<(f64, f64)> {
    let method_str = subpixel_method.unwrap_or("gaussian");
    let method = SubpixelMethod::parse(method_str)
        .map_err(|e| PyValueError::new_err(e))?;

    let arr = corr.as_array();
    let h = arr.shape()[0];
    let w = arr.shape()[1];

    let corr_cow = to_c_contiguous_2d(&arr);
    Ok(subpixel_peak_position_2d(&corr_cow, h, w, method))
}

/// Batched subpixel peak positions for a 3D array of correlation maps (N, H, W).
/// Returns (peaks_i, peaks_j) as 1D arrays of length N.
#[pyfunction]
#[pyo3(signature = (corr, subpixel_method=None))]
fn batch_find_subpixel_peak_position<'py>(
    py: Python<'py>,
    corr: PyReadonlyArray3<f64>,
    subpixel_method: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let method_str = subpixel_method.unwrap_or("gaussian");
    let method = SubpixelMethod::parse(method_str)
        .map_err(|e| PyValueError::new_err(e))?;

    let arr = corr.as_array();
    let num_wins = arr.shape()[0];
    let h = arr.shape()[1];
    let w = arr.shape()[2];
    let win_size = h * w;

    let corr_cow = to_c_contiguous(&arr);

    let mut peaks_i = vec![0.0; num_wins];
    let mut peaks_j = vec![0.0; num_wins];

    py.allow_threads(|| {
        peaks_i
            .par_iter_mut()
            .zip(peaks_j.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (pi, pj))| {
                let win_slice = &corr_cow[idx * win_size..(idx + 1) * win_size];
                let (r, c) = subpixel_peak_position_2d(win_slice, h, w, method);
                *pi = r;
                *pj = c;
            });
    });

    Ok((peaks_i.to_pyarray(py), peaks_j.to_pyarray(py)))
}

/// Batched conversion from correlation maps (N, H, W) to (u, v) displacement grids of shape (n_rows, n_cols).
#[pyfunction]
#[pyo3(signature = (corr, n_rows, n_cols, subpixel_method=None))]
fn batch_correlation_to_displacement<'py>(
    py: Python<'py>,
    corr: PyReadonlyArray3<f64>,
    n_rows: usize,
    n_cols: usize,
    subpixel_method: Option<&str>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let method_str = subpixel_method.unwrap_or("gaussian");
    let method = SubpixelMethod::parse(method_str)
        .map_err(|e| PyValueError::new_err(e))?;

    let arr = corr.as_array();
    let num_wins = arr.shape()[0];
    let h = arr.shape()[1];
    let w = arr.shape()[2];

    if num_wins != n_rows * n_cols {
        return Err(PyValueError::new_err(format!(
            "Number of correlation windows ({num_wins}) does not match n_rows * n_cols ({n_rows} * {n_cols} = {})",
            n_rows * n_cols
        )));
    }

    let win_size = h * w;
    let corr_cow = to_c_contiguous(&arr);

    let default_peak_i = (h / 2) as f64;
    let default_peak_j = (w / 2) as f64;

    let mut u_vec = vec![0.0; num_wins];
    let mut v_vec = vec![0.0; num_wins];

    py.allow_threads(|| {
        u_vec
            .par_iter_mut()
            .zip(v_vec.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (u_val, v_val))| {
                let win_slice = &corr_cow[idx * win_size..(idx + 1) * win_size];
                let (peak_i, peak_j) = subpixel_peak_position_2d(win_slice, h, w, method);
                *u_val = peak_j - default_peak_j;
                *v_val = peak_i - default_peak_i;
            });
    });

    let u_arr = u_vec.to_pyarray(py).reshape([n_rows, n_cols])?;
    let v_arr = v_vec.to_pyarray(py).reshape([n_rows, n_cols])?;

    Ok((u_arr, v_arr))
}

#[pymodule]
fn openpiv_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft_correlate_circular, m)?)?;
    m.add_function(wrap_pyfunction!(fast_batch_cross_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(fft_correlate_linear, m)?)?;
    m.add_function(wrap_pyfunction!(find_subpixel_peak_position, m)?)?;
    m.add_function(wrap_pyfunction!(batch_find_subpixel_peak_position, m)?)?;
    m.add_function(wrap_pyfunction!(batch_correlation_to_displacement, m)?)?;
    Ok(())
}
