# Security & Data Hygiene Policy

## Supported Versions

| Version | Supported |
| :--- | :---: |
| `0.2.x` | ✅ (Active / Recommended) |
| `0.1.x` | ✅ |

## Reporting a Vulnerability

If you discover a security issue (for example, unsafe file parsing, integer/floating-point division traps, or denial of service via malformed input CSV files) in `ice-cfd-validation-assistant`:

1. **Do not** open a public issue containing exploit payloads or crash files.
2. Report the vulnerability privately via [GitHub Security Advisories](https://github.com/alpztel/ice-cfd-validation-assistant/security/advisories/new) with a minimal, synthetic reproduction snippet.
3. You may expect an initial response within 48 hours and a patched release within 7 days.

## Numerical Safety & Boundary Sanitization

`ice-cfd-validation-assistant` incorporates defensive programming across all physics routines:
- **Strict Positivity Guards:** Spatially averaged cylinder pressure ($P > 0$), cylinder volume ($V > 0$), temperature ($T > 0$), trapped mass ($m > 0$), and AMR cell counts ($N_i > 0$) are strictly validated at ingestion to prevent non-physical floating-point overflows or square-root of negative numbers.
- **Singularity Protection:** Hardenberg-Hase delay features an explicit guard for $`P_{\mathrm{TDC}} \le 1.24\ \mathrm{MPa}`$ ($12.4\ \mathrm{bar}$) to prevent logarithmic and fractional exponent singularities; ASME GCI incorporates Celik et al. (2008) sign factor $`s = \mathrm{sgn}(\epsilon_{32}/\epsilon_{21})`$ to safeguard against `NaN` crashes on oscillatory AMR mesh triplets; Livengood-Wu solver explicitly detects high-altitude subcritical misfire ($I(\theta) < 1.0$) returning `NaN` indicators rather than erroneous roots.
- **Zero Memory Leaks:** Plotting routines run strictly headless via `FigureCanvasAgg` without contaminating the global `matplotlib.pyplot` state.

## Strict NDA & Intellectual Property Hygiene

`ice-cfd-validation-assistant` is an open-source academic and engineering verification library. Contributors and users opening issues or pull requests must **never** attach proprietary engine CAD geometries, OEM calibration tables, or confidential test-cell logs. Always reproduce bugs using synthetic analytical curves or open [Engine Combustion Network (ECN)](https://ecn.sandia.gov/) datasets.
