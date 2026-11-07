# Verification & Validation Benchmarks

This directory serves as the workspace for open-literature **Engine Combustion Network (ECN)** experimental datasets (`Spray A` $n$-dodecane $90\ \mu\text{m}$ and `Spray D` $186\ \mu\text{m}$ nozzles) and analytical verification reference tables.

## Strict Open-Science Policy
- Zero proprietary engine geometries or OEM test-cell measurements are stored in this repository.
- Official public Sandia ECN experimental penetration files (`.csv`, `.txt`, `.dat`) downloaded from [https://ecn.sandia.gov/](https://ecn.sandia.gov/) can be placed here locally and ingested directly into strict SI units (`s`, `m`) via `ice_cfd_validator.load_ecn_spray_data`:

```python
from ice_cfd_validator import compute_spray_validation_metrics, load_ecn_spray_data, parse_spray_csv

ecn = load_ecn_spray_data("benchmarks/ECN_SprayA_900K_22.8kgm3.txt", condition_label="ECN Spray A")
cfd = parse_spray_csv("path/to/forte_spray.csv")

metrics = compute_spray_validation_metrics(
    cfd.vapor_penetration_m,
    ecn.vapor_penetration_m,
    cfd_time_s=cfd.time_s,
    ref_time_s=ecn.time_s,
)
print(f"RMSE = {metrics.rmse_m * 1e3:.2f} mm | MAPE = {metrics.mape_pct:.2f}% | L2 = {metrics.l2_relative_error:.4f}")
```
