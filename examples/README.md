# Examples & End-to-End Engineering Showcases

This directory contains executable, strictly non-proprietary post-processing and verification workflows demonstrating the full capabilities of **`ice-cfd-validation-assistant`**.

## Available Workflows

### [`altitude_sweep_demo.py`](altitude_sweep_demo.py) — Turbocharged Aero-Diesel Flight-Envelope Sweep (0 ft → 15,000 ft → 30,000 ft)
Executes a complete 7-stage multi-physics thermodynamic, auto-ignition kinetics, Second-Law exergy, spray-physics, and ASME V&V 20-2009 mesh-convergence workflow on an open-literature 4-stroke Aero-Diesel single cylinder ($B = 85\ \text{mm}$, $S = 90\ \text{mm}$, $l = 155\ \text{mm}$, $r_c = 16.5$, $N = 2400\ \text{rpm}$, $\Delta P_{\text{inj}} = 100\ \text{MPa}$, $d_o = 90\ \mu\text{m}$, piston-bowl clearance $L_{\text{wall}} = 38.0\ \text{mm}$):

1. **ISO 2533 Standard Atmosphere (`compute_isa_conditions`)**: Evaluates ambient $T_{\text{amb}}(z)$, $P_{\text{amb}}(z)$, $\rho_{\text{amb}}(z)$, and $a(z)$ at `0 ft` (`0 m`), `15,000 ft` (`4,572 m`), and `30,000 ft` (`9,144 m`) with realistic single-stage turbocharger boost pressure-ratio clipping at high altitude.
2. **Closed-Cycle First-Law ODE Synthesis & Inverse AHRR (`thermo/`, `signal/`)**: Integrates a synthetic noisy in-cylinder $P(\theta)$ trace with NASA RP-1311 14-coefficient (2 × 7-term) $\gamma(T)$ and Woschni (1967) wall heat loss, applies zero-phase Savitzky–Golay smoothing, and recovers Net/Gross AHRR, `CA10`/`CA50`/`CA90`, and `IMEP`.
3. **High-Altitude Spray Penetration & Wall-Impingement Audit (`spray/`)**: Evaluates Hiroyasu & Arai (1990) and Naber & Siebers (1996) / Siebers (1999) penetration curves alongside synthetic ANSYS Forte `spray.csv` outputs, detecting piston-bowl wall impingement onset ($t_{\text{imp}} = 0.64\ \text{ms}$) at `30,000 ft` where $\rho_{g,\text{SOI}}$ drops from `36.73` to `11.87 kg/m^3`.
4. **ASME V&V 20-2009 Three-Mesh GCI (`vv/gci.py`)**: Quantifies apparent order $p = 1.920$, continuum peak pressure $\phi_{\text{ext}}^{21}$, and fine-grid discretization uncertainty $GCI_{\mathrm{fine}}^{21} = 1.072\%$ across `235k`, `680k`, and `1.92M` 3D cells.
5. **Dynamic Livengood-Wu (1955) Auto-Ignition Delay & SOC (`ignition/`)**: Evaluates Jet-A1 ($CN = 42.0$) Arrhenius ignition kinetics, tracks precursor accumulation $I(\theta) = 1.0$ from $\theta_{\text{SOI}} = -8^\circ$ ATDC, and predicts start of combustion $\theta_{\text{SOC}} = -3.47^\circ$ at sea level retarding to $+3.73^\circ$ ATDC at $30{,}000\ \text{ft}$ ($\tau_{\text{id}}$ lengthening from $0.31\ \text{ms}$ to $0.81\ \text{ms}$).
6. **Altitude Dead-State Second-Law Exergy Budget (`exergy/`)**: Evaluates ambient dead state $(P_0(z), T_0(z))$ via `DeadStateConditions.from_altitude(z)`, closed-cycle thermomechanical exergy, work and heat transfer availability rates, and Second-Law efficiency $\eta_{\mathrm{II}}$ ($47.94\% \to 44.50\%$) with strict $100.0000\%$ balance closure.
7. **300-DPI SAE/IEEE Publication Export (`plotting/`)**: Writes `thermo_diagnostics.png`, `pv_log_diagram.png`, `altitude_spray_comparison.png`, and `gci_convergence.png` directly to `../assets/`.

```bash
python examples/altitude_sweep_demo.py
```
