"""Analytical Type-III compensator design used by the PID autotuner."""

from __future__ import annotations

import math

from .models import PidParameters, PlantParams


class CompensatorDesign:
    """Compute PID gains from crossover frequency and phase margin.

    This follows the Simulation-Autotuner approach: the search space is two
    dimensional (`wc`, `phi_m`), and every candidate is converted into the four
    parallel PID parameters used by the controller model.
    """

    def __init__(self, plant: PlantParams):
        self.plant = plant

    def compute(self, wc_rad_s: float, phi_margin_deg: float) -> PidParameters:
        if wc_rad_s <= 0:
            raise ValueError("Crossover frequency must be positive.")

        wl = wc_rad_s / 10.0
        plant_value = self._plant_at(wc_rad_s)
        gain_plant = abs(plant_value)
        if gain_plant <= 0:
            raise ValueError("Plant gain is zero at crossover.")

        phi_plant = math.atan2(plant_value.imag, plant_value.real)
        phi_margin_rad = math.radians(phi_margin_deg)
        phi_boost = (-math.pi + phi_margin_rad) - phi_plant
        phi_boost = max(math.radians(-89.4), min(math.radians(89.4), phi_boost))

        sin_phi = math.sin(phi_boost)
        wz = wc_rad_s * math.sqrt((1.0 - sin_phi) / (1.0 + sin_phi))
        wp = wc_rad_s * math.sqrt((1.0 + sin_phi) / (1.0 - sin_phi))
        gpid0 = (1.0 / gain_plant) * math.sqrt((1.0 + (wc_rad_s / wp) ** 2) / (1.0 + (wc_rad_s / wz) ** 2))

        ki = gpid0 * wl
        kf = wp
        kp = gpid0 * (1.0 + wl / wz) - ki / kf
        kd = max(0.0, gpid0 / wz - kp / kf)
        return PidParameters(kp=kp, ki=ki, kd=kd, kf=kf)

    def _plant_at(self, omega_rad_s: float) -> complex:
        p = self.plant
        s = 1j * omega_rad_s
        w0 = 1.0 / math.sqrt(p.inductance_h * p.capacitance_f)
        q = math.sqrt(p.inductance_h / p.capacitance_f) / (p.capacitor_esr_ohm + p.inductor_dcr_ohm)
        wesr = 1.0 / (p.capacitor_esr_ohm * p.capacitance_f)
        numerator = p.vdc * (1.0 + s / wesr)
        denominator = 1.0 + s / (q * w0) + (s / w0) ** 2
        return numerator / denominator
