# Contributing to `ice-cfd-validation-assistant`

Thank you for contributing to **`ice-cfd-validation-assistant`**. Because this toolkit is used for quantitative thermodynamic diagnostics and **ASME V&V 20-2009** 3D CFD verification, all contributions must satisfy strict physical, dimensional, and numerical engineering invariants.

---

## 1. Non-Negotiable Engineering Rules

### 1.1 Zero Proprietary Data (Strict NDA Hygiene)
- **Never** commit confidential engine CAD/mesh geometries, proprietary OEM calibration maps, restricted test-cell datasets, or ANSYS Forte binary restart files (`*.ftres`, `*.ftsim`, `*.cas.h5`, `*.cgns`).
- All tests, examples, and benchmarks must rely exclusively on:
  1. Closed-form analytical solutions (e.g., adiabatic/polytropic cycles, forward-ODE Wiebe synthetic traces, exact power-law grid convergence polynomials), or
  2. Publicly licensed open-science datasets such as the **Sandia Engine Combustion Network (ECN)** Spray A / Spray D archives.

### 1.2 Strict SI Internal Architecture
- Every internal physics kernel (`signal/`, `thermo/`, `spray/`, `vv/`, `ignition/`, `exergy/`, `constants.py`) must accept and return **strict SI base/derived units** (`Pa`, `m`, `m^2`, `m^3`, `K`, `kg`, `kg/m^3`, `s`, `J`, `W/m^2`) with explicit crank-angle conventions (`deg` vs. `rad` documented in parameter names such as `crank_angle_deg`, `dv_dtheta_m3_per_deg`).
- Unit conversions (`MPa -> Pa`, `bar -> Pa`, `dyne/cm^2 -> Pa`, `cm^3 -> m^3`, `mm -> m`, `ms -> s`, `degC -> K`, `erg/CA -> J/deg`) happen **exclusively at ingestion (`io/`) or display (`plotting/`) boundaries**.

### 1.3 Vectorization (Zero Element-Wise Loops)
- Numerical modules (`signal/`, `thermo/`, `spray/`, `vv/`, `ignition/`, `exergy/`, `constants.py`) are audited via Python's `ast` module across `tests/test_01_io_and_constants.py` through `tests/test_07_exergy_second_law.py`.
- Element-wise Python `for` or `while` loops over crank-angle, time, temperature, or mesh arrays are **strictly forbidden** — use vectorized `numpy` broadcasting, `np.where`, `scipy.interpolate.CubicSpline`, and `scipy.signal.sosfiltfilt`.

### 1.4 Static Type Safety & Documentation
- All public functions, methods, and `@dataclass(frozen=True, slots=True)` containers require complete Python 3.11+ type annotations (including return types) and NumPy-style docstrings stating exact SI units in brackets (`[Pa]`, `[m^3/deg]`, `[W/(m^2*K)]`).
- Static type verification with `pyrefly` must pass with **0 errors**.

---

## 2. Local Development Setup

```bash
git clone https://github.com/alpztel/ice-cfd-validation-assistant.git
cd ice-cfd-validation-assistant

python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 3. Pre-Submission Verification Checklist

Before opening a Pull Request, run all three checks from the repository root:

```bash
# 1. Static type checking (must report 0 errors)
pyrefly check

# 2. Analytical & strict physical audit suite (all 65 tests must pass)
pytest -v

# 3. Headless end-to-end showcase & 300-DPI figure generation smoke test
python examples/altitude_sweep_demo.py
```

---

## 4. Adding New Correlations or V&V Models

1. Cite the primary peer-reviewed source (e.g., SAE Technical Paper number, ASME standard section, or NASA Technical Report) in both the module docstring and `CITATION.cff` / `README.md`.
2. Add domain bounds validation raising `PhysicalBoundsError` from `ice_cfd_validator.exceptions` for non-physical inputs (e.g., $P \le 0$, $T \le 0$, $r_c \le 1$, $h_1 \ge h_2$).
3. Include an analytical or asymptotic verification test in `tests/` proving machine-precision or theoretical order-of-accuracy convergence.

