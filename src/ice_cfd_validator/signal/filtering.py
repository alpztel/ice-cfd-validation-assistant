"""Zero-phase signal filtering and 4th-order vectorized numerical differentiation.

Prevents crank-angle phase distortion (CA50 shift) when computing in-cylinder
pressure derivatives $dP/d\\theta$ from noisy experimental or high-frequency
3D CFD traces (Savitzky & Golay, 1964; Fornberg, 1988; Heywood, 2018).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, savgol_filter, sosfiltfilt

FloatArray = NDArray[np.float64]
FilterType = Literal["low", "high", "bandpass"]
DerivativeMethod = Literal["auto", "central4", "spline"]


def savgol_zero_phase(
    signal: FloatArray,
    window_length: int = 21,
    polyorder: int = 3,
) -> FloatArray:
    """Apply symmetric zero-phase Savitzky-Golay polynomial smoothing.

    Uses local least-squares polynomial fitting of degree ``polyorder`` over a
    symmetric stencil of ``window_length`` samples (Savitzky & Golay, 1964),
    preserving zero group delay at interior points so combustion phasing (CA50)
    remains unshifted.

    Parameters
    ----------
    signal : NDArray[np.float64]
        Input 1D signal array (e.g., cylinder pressure $P$ [Pa]).
    window_length : int, default=21
        Odd stencil width in samples ($\\ge \\text{polyorder} + 2$).
    polyorder : int, default=3
        Local polynomial degree ($\\ge 1$).

    Returns
    -------
    NDArray[np.float64]
        Zero-phase smoothed signal in identical SI units.
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Input signal must be a 1D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input signal contains NaN or infinite values.")
    if polyorder < 1:
        raise ValueError("polyorder must be >= 1.")
    if window_length % 2 == 0 or window_length < polyorder + 2:
        raise ValueError("window_length must be odd and >= polyorder + 2.")
    if window_length > arr.size:
        raise ValueError(f"window_length ({window_length}) exceeds signal size ({arr.size}).")
    return np.asarray(
        savgol_filter(arr, window_length=window_length, polyorder=polyorder, mode="interp"),
        dtype=np.float64,
    )


def butterworth_zero_phase(
    signal: FloatArray,
    fs: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: FilterType = "low",
) -> FloatArray:
    """Apply forward-backward zero-phase Butterworth IIR filtering via ``filtfilt``.

    Executes causal and anti-causal passes via ``scipy.signal.filtfilt`` so the
    effective transfer function $|H(e^{j\\omega})|^2$ has strictly zero phase
    across all frequencies (Oppenheim & Schafer, 2009), eliminating CA50 lag.

    Parameters
    ----------
    signal : NDArray[np.float64]
        Input 1D signal array (e.g., cylinder pressure $P$ [Pa]).
    fs : float
        Sampling frequency (e.g., samples/deg for crank-angle filtering or Hz
        for acoustic knock isolation).
    cutoff : float | tuple[float, float]
        Cutoff frequency (scalar for ``"low"``/``"high"``, 2-tuple for ``"bandpass"``)
        in the same units as ``fs``, strictly below the Nyquist limit $f_s / 2$.
    order : int, default=4
        Single-pass Butterworth order (effective magnitude roll-off is $2 \\times \\text{order}$).
    btype : {"low", "high", "bandpass"}, default="low"
        Filter passband topology.

    Returns
    -------
    NDArray[np.float64]
        Zero-phase filtered signal in identical SI units.
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Input signal must be a 1D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input signal contains NaN or infinite values.")
    if fs <= 0.0:
        raise ValueError("Sampling frequency fs must be > 0.")
    if order < 1:
        raise ValueError("Filter order must be >= 1.")

    nyquist = 0.5 * fs
    if btype == "bandpass":
        if not isinstance(cutoff, tuple) or len(cutoff) != 2:
            raise ValueError("cutoff must be a (low, high) 2-tuple when btype='bandpass'.")
        f_low, f_high = cutoff
        if not (0.0 < f_low < f_high < nyquist):
            raise ValueError(
                f"Bandpass cutoffs ({f_low}, {f_high}) must satisfy 0 < low < high < fs/2 ({nyquist})."
            )
    else:
        if isinstance(cutoff, tuple):
            raise ValueError(f"cutoff must be a scalar float when btype='{btype}'.")
        if not (0.0 < cutoff < nyquist):
            raise ValueError(
                f"Cutoff frequency ({cutoff}) must satisfy 0 < cutoff < fs/2 ({nyquist})."
            )

    if order <= 8:
        ba_coeffs = butter(order, cutoff, btype=btype, fs=fs, output="ba")
        assert isinstance(ba_coeffs, tuple) and len(ba_coeffs) == 2
        b_arr = np.asarray(ba_coeffs[0], dtype=np.float64)
        a_arr = np.asarray(ba_coeffs[1], dtype=np.float64)
        padlen = 3 * max(a_arr.size, b_arr.size)
        if arr.size <= padlen:
            raise ValueError(
                f"Signal length ({arr.size}) must exceed filtfilt padlen ({padlen})."
            )
        if np.all(np.isfinite(a_arr)) and np.all(np.abs(np.roots(a_arr)) < 1.0 - 1.0e-7):
            return np.asarray(filtfilt(b_arr, a_arr, arr), dtype=np.float64)

    sos = butter(order, cutoff, btype=btype, fs=fs, output="sos")
    return np.asarray(sosfiltfilt(sos, arr), dtype=np.float64)


def _lagrange5_deriv_at_first_col(x5: FloatArray, y5: FloatArray) -> FloatArray:
    """Evaluate exact 5-point O(h^4) Lagrange derivative at column 0 without loops."""
    d1 = x5[:, 1] - x5[:, 0]
    d2 = x5[:, 2] - x5[:, 0]
    d3 = x5[:, 3] - x5[:, 0]
    d4 = x5[:, 4] - x5[:, 0]

    w1 = (-d2 * d3 * d4) / (d1 * (d1 - d2) * (d1 - d3) * (d1 - d4))
    w2 = (-d1 * d3 * d4) / (d2 * (d2 - d1) * (d2 - d3) * (d2 - d4))
    w3 = (-d1 * d2 * d4) / (d3 * (d3 - d1) * (d3 - d2) * (d3 - d4))
    w4 = (-d1 * d2 * d3) / (d4 * (d4 - d1) * (d4 - d2) * (d4 - d3))
    w0 = -(w1 + w2 + w3 + w4)

    return np.asarray(
        w0 * y5[:, 0] + w1 * y5[:, 1] + w2 * y5[:, 2] + w3 * y5[:, 3] + w4 * y5[:, 4],
        dtype=np.float64,
    )


def _nonuniform_central4(x_arr: FloatArray, y_arr: FloatArray) -> FloatArray:
    """Vectorized 5-point O(h^4) finite-difference derivative on non-uniform grids."""
    dydx = np.empty_like(y_arr)

    # Interior centered 5-point stencils: target x_i moved to column 0
    win_x = sliding_window_view(x_arr, 5)
    win_y = sliding_window_view(y_arr, 5)
    perm_int = (2, 0, 1, 3, 4)
    dydx[2:-2] = _lagrange5_deriv_at_first_col(win_x[:, perm_int], win_y[:, perm_int])

    # Boundary 5-point one-sided stencils (O(h^4) at i = 0, 1, N-2, N-1)
    b_x = np.vstack((
        x_arr[[0, 1, 2, 3, 4]],
        x_arr[[1, 0, 2, 3, 4]],
        x_arr[[-2, -5, -4, -3, -1]],
        x_arr[[-1, -5, -4, -3, -2]],
    ))
    b_y = np.vstack((
        y_arr[[0, 1, 2, 3, 4]],
        y_arr[[1, 0, 2, 3, 4]],
        y_arr[[-2, -5, -4, -3, -1]],
        y_arr[[-1, -5, -4, -3, -2]],
    ))
    b_vals = _lagrange5_deriv_at_first_col(b_x, b_y)
    dydx[0], dydx[1], dydx[-2], dydx[-1] = b_vals[0], b_vals[1], b_vals[2], b_vals[3]
    return dydx


def compute_derivative(
    x: FloatArray,
    y: FloatArray,
    method: DerivativeMethod = "auto",
) -> FloatArray:
    """Compute $\\mathcal{O}(h^4)$ accurate first derivative $dy/dx$ without loops.

    Supports both uniform and non-uniform crank-angle steps:

    - ``"central4"``: Evaluates 5-point 4th-order central finite differences in the
      interior and exact 5-point $\\mathcal{O}(h^4)$ one-sided stencils at boundaries
      (Fornberg, 1988). Uses fast constant weights on uniform grids:

      $$y'_i = \\frac{-y_{i+2} + 8 y_{i+1} - 8 y_{i-1} + y_{i-2}}{12 h}$$

      and exact vectorized 5-point Lagrange finite-difference weights on non-uniform
      grids (e.g., adaptive $\\Delta t$ CFD outputs).
    - ``"spline"``: Evaluates analytical first derivatives of a ``not-a-knot`` $C^2$
      cubic spline (`scipy.interpolate.CubicSpline`).
    - ``"auto"``: Selects ``"central4"`` on uniform grids ($N \\ge 5$) and ``"spline"``
      on non-uniform grids.

    Parameters
    ----------
    x : NDArray[np.float64]
        Strictly increasing independent coordinate (e.g., $\\theta$ [deg] or $t$ [s]).
    y : NDArray[np.float64]
        Dependent physical variable (e.g., $P$ [Pa] or $V$ [$\\text{m}^3$]).
    method : {"auto", "central4", "spline"}, default="auto"
        Differentiation scheme.

    Returns
    -------
    NDArray[np.float64]
        First derivative $dy/dx$ in SI units of ``y`` per unit of ``x``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
        raise ValueError("x and y must be 1D arrays of identical length.")
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("x and y must contain finite numeric values.")
    if x_arr.size < 4:
        raise ValueError("At least 4 grid points are required for high-order differentiation.")

    dx = np.diff(x_arr)
    if np.any(dx <= 0.0):
        raise ValueError("Independent coordinate x must be strictly increasing.")

    is_uniform = x_arr.size >= 5 and np.allclose(dx, dx[0], rtol=1e-6, atol=1e-12)

    if method == "central4":
        if x_arr.size < 5:
            raise ValueError("method='central4' requires at least 5 grid points.")
        if not is_uniform:
            return _nonuniform_central4(x_arr, y_arr)
        use_uniform_central = True
    elif method == "spline":
        use_uniform_central = False
    elif method == "auto":
        use_uniform_central = is_uniform
    else:
        raise ValueError(f"Unknown derivative method '{method}'. Expected 'auto', 'central4', or 'spline'.")

    if not use_uniform_central:
        spline = CubicSpline(x_arr, y_arr, bc_type="not-a-knot")
        return np.asarray(spline(x_arr, 1), dtype=np.float64)

    h = float(np.mean(dx))
    inv_12h = 1.0 / (12.0 * h)
    dydx = np.empty_like(y_arr)

    # Interior 4th-order central stencil: i in [2, N-3]
    dydx[2:-2] = (-y_arr[4:] + 8.0 * y_arr[3:-1] - 8.0 * y_arr[1:-3] + y_arr[:-4]) * inv_12h

    # Boundary 4th-order one-sided stencils (Fornberg, 1988, Table 1)
    dydx[0] = (
        -25.0 * y_arr[0] + 48.0 * y_arr[1] - 36.0 * y_arr[2] + 16.0 * y_arr[3] - 3.0 * y_arr[4]
    ) * inv_12h
    dydx[1] = (
        -3.0 * y_arr[0] - 10.0 * y_arr[1] + 18.0 * y_arr[2] - 6.0 * y_arr[3] + y_arr[4]
    ) * inv_12h
    dydx[-2] = (
        -y_arr[-5] + 6.0 * y_arr[-4] - 18.0 * y_arr[-3] + 10.0 * y_arr[-2] + 3.0 * y_arr[-1]
    ) * inv_12h
    dydx[-1] = (
        3.0 * y_arr[-5] - 16.0 * y_arr[-4] + 36.0 * y_arr[-3] - 48.0 * y_arr[-2] + 25.0 * y_arr[-1]
    ) * inv_12h

    return dydx
