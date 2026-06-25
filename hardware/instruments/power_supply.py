"""SCPI drivers for programmable DC power supplies."""

from __future__ import annotations

from dataclasses import dataclass

from .visa_resource import VisaInstrument


@dataclass
class PowerSupplyReadback:
    identity: str
    output_enabled: bool | None
    voltage_setpoint_v: float | None
    current_limit_a: float | None
    measured_voltage_v: float | None
    measured_current_a: float | None
    error: str | None = None


@dataclass
class KeysightN5700PowerSupply(VisaInstrument):
    """Keysight/Agilent N5700 series supply driver.

    For master/slave parallel wiring, connect to the master instrument only.
    This class intentionally avoids commands that alter front-panel, analog,
    tracking, or parallel-system topology.
    """

    model_voltage_limit_v: float = 60.0
    model_current_limit_a: float = 25.0
    configured_parallel_units: int = 2

    @property
    def command_current_limit_a(self) -> float:
        return self.model_current_limit_a

    def readback(self) -> PowerSupplyReadback:
        return PowerSupplyReadback(
            identity=self.idn(),
            output_enabled=self.get_output_enabled(),
            voltage_setpoint_v=_safe_float_query(self, "VOLT?"),
            current_limit_a=_safe_float_query(self, "CURR?"),
            measured_voltage_v=_safe_float_query(self, "MEAS:VOLT?"),
            measured_current_a=_safe_float_query(self, "MEAS:CURR?"),
            error=self.get_error(),
        )

    def get_output_enabled(self) -> bool | None:
        try:
            raw = self.query("OUTP?").strip().upper()
        except Exception:
            return None
        if raw in {"1", "ON"}:
            return True
        if raw in {"0", "OFF"}:
            return False
        return None

    def set_voltage(self, voltage_v: float) -> None:
        if not 0.0 <= voltage_v <= self.model_voltage_limit_v:
            raise ValueError(
                f"Voltage {voltage_v} V is outside safe range 0-{self.model_voltage_limit_v} V."
            )
        self.write(f"VOLT {voltage_v:.12g}")

    def set_current_limit(self, current_a: float) -> None:
        if not 0.0 <= current_a <= self.command_current_limit_a:
            raise ValueError(
                f"Current {current_a} A is outside safe command range "
                f"0-{self.command_current_limit_a} A. Increase the explicit "
                "software limit only after confirming the master/slave current "
                "programming semantics."
            )
        self.write(f"CURR {current_a:.12g}")

    def configure_output_limits(self, voltage_v: float, current_limit_a: float) -> None:
        """Set voltage and current while leaving output state unchanged."""

        self.set_voltage(voltage_v)
        self.set_current_limit(current_limit_a)

    def output_on(self) -> None:
        self.write("OUTP ON")

    def output_off(self) -> None:
        self.write("OUTP OFF")

    def get_error(self) -> str:
        return self.query("SYST:ERR?")


def _safe_float_query(instrument: VisaInstrument, command: str) -> float | None:
    try:
        return float(instrument.query(command))
    except Exception:
        return None
