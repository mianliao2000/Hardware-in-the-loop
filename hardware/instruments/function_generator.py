"""Generic SCPI function-generator driver.

The command set here intentionally uses common SCPI commands supported by many
Keysight, Rigol, Siglent, Tektronix, and Aim-TTi function generators. Model
specific quirks can be added later as subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

from .visa_resource import VisaInstrument


@dataclass
class FunctionGenerator(VisaInstrument):
    """Minimal USB/VISA function generator driver."""

    output_channel: int = 1

    def configure_sine(
        self,
        frequency_hz: float,
        amplitude_vpp: float,
        offset_v: float = 0.0,
        phase_deg: float | None = None,
        channel: int | None = None,
    ) -> None:
        """Configure a sine wave while leaving output state unchanged."""

        ch = channel or self.output_channel
        self.write(f"SOUR{ch}:FUNC SIN")
        self.write(f"SOUR{ch}:FREQ {frequency_hz:.12g}")
        self.write(f"SOUR{ch}:VOLT {amplitude_vpp:.12g}")
        self.write(f"SOUR{ch}:VOLT:OFFS {offset_v:.12g}")
        if phase_deg is not None:
            self.set_phase(phase_deg, ch)

    def configure_square(
        self,
        frequency_hz: float,
        amplitude_vpp: float,
        offset_v: float = 0.0,
        duty_percent: float = 50.0,
        channel: int | None = None,
    ) -> None:
        """Configure a square wave while leaving output state unchanged."""

        ch = channel or self.output_channel
        self.write(f"SOUR{ch}:FUNC SQU")
        self.write(f"SOUR{ch}:FREQ {frequency_hz:.12g}")
        self.write(f"SOUR{ch}:VOLT {amplitude_vpp:.12g}")
        self.write(f"SOUR{ch}:VOLT:OFFS {offset_v:.12g}")

    def configure_square_levels(
        self,
        frequency_hz: float,
        low_v: float,
        high_v: float,
        duty_percent: float = 50.0,
        channel: int | None = None,
    ) -> None:
        ch = channel or self.output_channel
        self.write(f"SOUR{ch}:FUNC SQU")
        self.write(f"SOUR{ch}:FREQ {frequency_hz:.12g}")
        self.write(f"SOUR{ch}:VOLT:LOW {low_v:.12g}")
        self.write(f"SOUR{ch}:VOLT:HIGH {high_v:.12g}")

    def configure_pulse_levels(
        self,
        frequency_hz: float,
        low_v: float,
        high_v: float,
        width_s: float | None = None,
        duty_percent: float | None = None,
        channel: int | None = None,
    ) -> None:
        ch = channel or self.output_channel
        self.write(f"SOUR{ch}:FUNC PULS")
        self.write(f"SOUR{ch}:FREQ {frequency_hz:.12g}")
        self.write(f"SOUR{ch}:VOLT:LOW {low_v:.12g}")
        self.write(f"SOUR{ch}:VOLT:HIGH {high_v:.12g}")
        if width_s is not None:
            self.write(f"SOUR{ch}:PULS:WIDT {width_s:.12g}")

    def configure_dc(self, level_v: float, channel: int | None = None) -> None:
        ch = channel or self.output_channel
        self.write(f"SOUR{ch}:FUNC DC")
        self.write(f"SOUR{ch}:VOLT:OFFS {level_v:.12g}")

    def output_on(self, channel: int | None = None) -> None:
        ch = channel or self.output_channel
        self.write(f"OUTP{ch} ON")

    def output_off(self, channel: int | None = None) -> None:
        ch = channel or self.output_channel
        self.write(f"OUTP{ch} OFF")

    def set_frequency(self, frequency_hz: float, channel: int | None = None) -> None:
        ch = channel or self.output_channel
        self.write(f"SOUR{ch}:FREQ {frequency_hz:.12g}")

    def set_phase(self, phase_deg: float, channel: int | None = None) -> None:
        ch = channel or self.output_channel
        self.write(f"SOUR{ch}:PHAS {phase_deg:.12g}DEG")

    def set_voltage_unit(self, unit: str, channel: int | None = None) -> None:
        ch = channel or self.output_channel
        normalized = unit.strip().upper()
        if normalized not in {"VPP", "VRMS", "DBM"}:
            raise ValueError(f"Unsupported voltage unit: {unit}")
        self.write(f"SOUR{ch}:VOLT:UNIT {normalized}")

    def get_frequency(self, channel: int | None = None) -> str:
        ch = channel or self.output_channel
        return self.query(f"SOUR{ch}:FREQ?")

    def get_phase(self, channel: int | None = None) -> str:
        ch = channel or self.output_channel
        return self.query(f"SOUR{ch}:PHAS?")

    def get_function(self, channel: int | None = None) -> str:
        ch = channel or self.output_channel
        return self.query(f"SOUR{ch}:FUNC?")

    def get_voltage(self, channel: int | None = None) -> str:
        ch = channel or self.output_channel
        return self.query(f"SOUR{ch}:VOLT?")

    def get_offset(self, channel: int | None = None) -> str:
        ch = channel or self.output_channel
        return self.query(f"SOUR{ch}:VOLT:OFFS?")

    def get_output_state(self, channel: int | None = None) -> str:
        ch = channel or self.output_channel
        return self.query(f"OUTP{ch}?")

    def get_error(self) -> str:
        return self.query("SYST:ERR?")
