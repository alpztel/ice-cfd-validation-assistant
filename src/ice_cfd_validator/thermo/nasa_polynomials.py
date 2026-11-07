"""7-coefficient NASA Glenn thermodynamic polynomials for variable specific heats.

Evaluates temperature-dependent $c_p(T)$, $c_v(T)$, $\\gamma(T) = c_p(T)/c_v(T)$,
specific enthalpy $h(T)$, and internal energy $u(T)$ across low ($200\\text{--}1000\\,\\text{K}$)
and high ($1000\\text{--}6000\\,\\text{K}$) intervals without constant-$\\gamma$ approximations
(Gordon & McBride, 1994, NASA RP-1311; McBride, Zehe & Gordon, 2002; GRI-Mech 3.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ice_cfd_validator.constants import R_UNIVERSAL
from ice_cfd_validator.exceptions import PhysicalBoundsError

FloatArray = NDArray[np.float64]
Coeff7 = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class NASAGasMixture:
    """14-parameter (2 x 7-coefficient) NASA ideal-gas species or mixture model.

    Evaluates dimensionless molar heat capacity and enthalpy via NASA RP-1311:

    $$\\frac{c_p^\\circ(T)}{R_u} = a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4$$

    $$\\frac{h^\\circ(T)}{R_u T} = a_1 + \\frac{a_2}{2} T + \\frac{a_3}{3} T^2
      + \\frac{a_4}{4} T^3 + \\frac{a_5}{5} T^4 + \\frac{a_6}{T}$$

    Parameters
    ----------
    name : str
        Identifier of species or mixture.
    molar_mass_kg_per_mol : float
        Mixture mean molar mass $W$ [$\\text{kg/mol}$].
    low_coeffs : tuple[float, ...]
        7 NASA coefficients $(a_1, \\dots, a_7)$ for $T \\in [200, 1000]\\,\\text{K}$.
    high_coeffs : tuple[float, ...]
        7 NASA coefficients $(a_1, \\dots, a_7)$ for $T \\in [1000, 6000]\\,\\text{K}$.
    t_mid_k : float, default=1000.0
        Piecewise handoff temperature $T_{\\text{mid}}$ [K].
    """

    name: str
    molar_mass_kg_per_mol: float
    low_coeffs: Coeff7
    high_coeffs: Coeff7
    t_mid_k: float = 1000.0

    def __post_init__(self) -> None:
        if self.molar_mass_kg_per_mol <= 0.0:
            raise PhysicalBoundsError("Molar mass W must be strictly positive (> 0 kg/mol).")
        if len(self.low_coeffs) != 7 or len(self.high_coeffs) != 7:
            raise ValueError("NASA polynomial coefficient tuples must each contain 7 terms.")

    @property
    def specific_gas_constant(self) -> float:
        """Specific gas constant $R_{\\text{spec}} = R_u / W$ [$\\text{J}/(\\text{kg}\\cdot\\text{K})$]."""
        return R_UNIVERSAL / self.molar_mass_kg_per_mol

    def _validate_temperature(self, temperature_k: ArrayLike) -> FloatArray:
        t = np.asarray(temperature_k, dtype=np.float64)
        if not np.all((t >= 200.0) & (t <= 6000.0)):
            raise PhysicalBoundsError(
                "Temperature outside NASA 7-term validity range [200 K, 6000 K] or non-finite."
            )
        return t

    def cp(self, temperature_k: ArrayLike) -> FloatArray:
        """Evaluate specific isobaric heat capacity $c_p(T)$ [$\\text{J}/(\\text{kg}\\cdot\\text{K})$]."""
        t = self._validate_temperature(temperature_k)
        lo = self.low_coeffs
        hi = self.high_coeffs
        cp_over_r_lo = lo[0] + t * (lo[1] + t * (lo[2] + t * (lo[3] + t * lo[4])))
        cp_over_r_hi = hi[0] + t * (hi[1] + t * (hi[2] + t * (hi[3] + t * hi[4])))
        cp_over_r = np.where(t < self.t_mid_k, cp_over_r_lo, cp_over_r_hi)
        return np.asarray(cp_over_r * self.specific_gas_constant, dtype=np.float64)

    def cv(self, temperature_k: ArrayLike) -> FloatArray:
        """Evaluate specific isochoric heat capacity $c_v(T) = c_p(T) - R_{\\text{spec}}$ [$\\text{J}/(\\text{kg}\\cdot\\text{K})$]."""
        return self.cp(temperature_k) - self.specific_gas_constant

    def gamma(self, temperature_k: ArrayLike) -> FloatArray:
        """Evaluate specific heat ratio $\\gamma(T) = c_p(T) / c_v(T)$ [-]."""
        cp_val = self.cp(temperature_k)
        return cp_val / (cp_val - self.specific_gas_constant)

    def enthalpy(self, temperature_k: ArrayLike) -> FloatArray:
        """Evaluate specific enthalpy $h(T)$ [$\\text{J/kg}$] including $a_6$ formation constant."""
        t = self._validate_temperature(temperature_k)
        lo = self.low_coeffs
        hi = self.high_coeffs
        h_lo = t * (lo[0] + t * (0.5 * lo[1] + t * ((lo[2] / 3.0) + t * (0.25 * lo[3] + 0.2 * lo[4] * t)))) + lo[5]
        h_hi = t * (hi[0] + t * (0.5 * hi[1] + t * ((hi[2] / 3.0) + t * (0.25 * hi[3] + 0.2 * hi[4] * t)))) + hi[5]
        h_over_r = np.where(t < self.t_mid_k, h_lo, h_hi)
        return np.asarray(h_over_r * self.specific_gas_constant, dtype=np.float64)

    def internal_energy(self, temperature_k: ArrayLike) -> FloatArray:
        """Evaluate specific internal energy $u(T) = h(T) - R_{\\text{spec}} T$ [$\\text{J/kg}$]."""
        t = self._validate_temperature(temperature_k)
        return self.enthalpy(t) - self.specific_gas_constant * t


# Authoritative NASA RP-1311 / GRI-Mech 3.0 pure-species records
SPECIES_DB: Final[dict[str, NASAGasMixture]] = {
    "N2": NASAGasMixture(
        name="N2",
        molar_mass_kg_per_mol=0.02801340,
        low_coeffs=(
            3.29867700e00, 1.40824040e-03, -3.96322200e-06,
            5.64151500e-09, -2.44485400e-12, -1.02089990e03, 3.95037200e00,
        ),
        high_coeffs=(
            2.92664000e00, 1.48797680e-03, -5.68476000e-07,
            1.00970380e-10, -6.75335100e-15, -9.22797700e02, 5.98052800e00,
        ),
    ),
    "O2": NASAGasMixture(
        name="O2",
        molar_mass_kg_per_mol=0.03199880,
        low_coeffs=(
            3.78245636e00, -2.99673416e-03, 9.84730201e-06,
            -9.68129509e-09, 3.24372837e-12, -1.06394356e03, 3.65767573e00,
        ),
        high_coeffs=(
            3.28253784e00, 1.48308754e-03, -7.57966669e-07,
            2.09470555e-10, -2.16717794e-14, -1.08845772e03, 5.45323129e00,
        ),
    ),
    "AR": NASAGasMixture(
        name="AR",
        molar_mass_kg_per_mol=0.03994800,
        low_coeffs=(2.5e00, 0.0, 0.0, 0.0, 0.0, -7.45375000e02, 4.37967491e00),
        high_coeffs=(2.5e00, 0.0, 0.0, 0.0, 0.0, -7.45375000e02, 4.37967491e00),
    ),
    "CO2": NASAGasMixture(
        name="CO2",
        molar_mass_kg_per_mol=0.04400950,
        low_coeffs=(
            2.35677352e00, 8.98459677e-03, -7.12356269e-06,
            2.45919022e-09, -1.43699548e-13, -4.83719697e04, 9.90105222e00,
        ),
        high_coeffs=(
            3.85746029e00, 4.41437026e-03, -2.21481404e-06,
            5.23490188e-10, -4.72084164e-14, -4.87591660e04, 2.27163806e00,
        ),
    ),
    "H2O": NASAGasMixture(
        name="H2O",
        molar_mass_kg_per_mol=0.01801528,
        low_coeffs=(
            4.19864056e00, -2.03643410e-03, 6.52040211e-06,
            -5.48797062e-09, 1.77197817e-12, -3.02937267e04, -8.49032208e-01,
        ),
        high_coeffs=(
            3.03399249e00, 2.17691804e-03, -1.64072518e-07,
            -9.70419870e-11, 1.68200992e-14, -3.00042971e04, 4.96677010e00,
        ),
    ),
}


def mix_nasa_species(name: str, mole_fractions: dict[str, float]) -> NASAGasMixture:
    """Combine pure-species 7-term NASA polynomials by mole fraction $y_k$."""
    total_y = sum(mole_fractions.values())
    if total_y <= 0.0 or any(v < 0.0 for v in mole_fractions.values()):
        raise PhysicalBoundsError("Species mole fractions must be non-negative with positive sum.")

    low = np.zeros(7, dtype=np.float64)
    high = np.zeros(7, dtype=np.float64)
    molar_mass = 0.0
    for sp_name, raw_y in mole_fractions.items():
        if sp_name not in SPECIES_DB:
            raise KeyError(f"Species '{sp_name}' not found in SPECIES_DB ({sorted(SPECIES_DB)}).")
        yk = raw_y / total_y
        sp = SPECIES_DB[sp_name]
        molar_mass += yk * sp.molar_mass_kg_per_mol
        low += yk * np.asarray(sp.low_coeffs, dtype=np.float64)
        high += yk * np.asarray(sp.high_coeffs, dtype=np.float64)

    lo_t: Coeff7 = (
        float(low[0]), float(low[1]), float(low[2]), float(low[3]),
        float(low[4]), float(low[5]), float(low[6]),
    )
    hi_t: Coeff7 = (
        float(high[0]), float(high[1]), float(high[2]), float(high[3]),
        float(high[4]), float(high[5]), float(high[6]),
    )
    return NASAGasMixture(
        name=name, molar_mass_kg_per_mol=molar_mass, low_coeffs=lo_t, high_coeffs=hi_t
    )


def make_combustion_mixture(
    equivalence_ratio: float = 0.7,
    h_c_ratio: float = 26.0 / 12.0,
) -> NASAGasMixture:
    """Build lean/stoichiometric hydrocarbon-air burned-gas NASA mixture ($0 \\le \\phi \\le 1$).

    Defaults to Aero-Diesel surrogate $n$-dodecane ($\\text{C}_{12}\\text{H}_{26}$,
    $H/C = 2.167$) complete combustion products at $\\phi = 0.7$ (Heywood, 2018, Ch. 3).
    """
    if not (0.0 <= equivalence_ratio <= 1.0):
        raise PhysicalBoundsError("Lean/stoichiometric mixture requires 0.0 <= equivalence_ratio <= 1.0.")
    if h_c_ratio < 0.0:
        raise PhysicalBoundsError("Hydrogen-to-carbon ratio h_c_ratio must be >= 0.")
    nu_s = 1.0 + 0.25 * h_c_ratio
    n_c = equivalence_ratio * 0.20946 / nu_s
    moles = {
        "N2": 0.78084,
        "O2": (1.0 - equivalence_ratio) * 0.20946,
        "AR": 0.00934,
        "CO2": 0.00036 + n_c,
        "H2O": 0.5 * h_c_ratio * n_c,
    }
    return mix_nasa_species(f"burned_gas_phi_{equivalence_ratio:.2f}", moles)


DRY_AIR: Final[NASAGasMixture] = make_combustion_mixture(equivalence_ratio=0.0)
DIESEL_BURNED_PRODUCTS: Final[NASAGasMixture] = make_combustion_mixture(equivalence_ratio=0.7)
STOICHIOMETRIC_PRODUCTS: Final[NASAGasMixture] = make_combustion_mixture(equivalence_ratio=1.0)


def compute_cp(temperature_k: ArrayLike, mixture: NASAGasMixture = DRY_AIR) -> FloatArray:
    """Vectorized specific heat $c_p(T)$ [$\\text{J}/(\\text{kg}\\cdot\\text{K})$] for ``mixture``."""
    return mixture.cp(temperature_k)


def compute_cv(temperature_k: ArrayLike, mixture: NASAGasMixture = DRY_AIR) -> FloatArray:
    """Vectorized specific heat $c_v(T)$ [$\\text{J}/(\\text{kg}\\cdot\\text{K})$] for ``mixture``."""
    return mixture.cv(temperature_k)


def compute_gamma(temperature_k: ArrayLike, mixture: NASAGasMixture = DRY_AIR) -> FloatArray:
    """Vectorized specific heat ratio $\\gamma(T) = c_p(T)/c_v(T)$ [-] for ``mixture``."""
    return mixture.gamma(temperature_k)
