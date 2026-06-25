"""Hardware backend skeleton for PID tuning iterations.

The simulation project has backend classes that return waveform and Bode data.
This module defines the equivalent shape for the real lab bench. The methods are
deliberately conservative stubs until each instrument is brought up and safety
limits are confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .testbench import HardwareTestbench, HardwareTestbenchConfig


@dataclass
class PidParameters:
    Kp: float
    Ki: float
    Kd: float
    Kf: float


@dataclass
class WaveformData:
    header: list[str]
    rows: list[list[float]]


@dataclass
class HardwareIterationResult:
    pid: PidParameters
    waveform: WaveformData
    bode: Optional[object] = None


@dataclass
class HardwareBackendConfig(HardwareTestbenchConfig):
    vout_target_v: float = 5.0
    vin_v: float = 12.0
    input_current_limit_a: float = 1.0
    enable_power_output: bool = False
    power_supply_resource: Optional[str] = None
    power_supply_voltage_limit_v: float = 60.0
    power_supply_current_limit_a: float = 25.0
    power_supply_parallel_units: int = 2
    bode_scpi_host: str = "127.0.0.1"
    bode_scpi_port: int = 5025
    bode_timeout_ms: int = 20000
    board_controller_kind: str = "generic"
    board_i2c_adapter: str = "mock"
    board_i2c_address: str = "0x40"
    board_name: str = "board"
    aardvark_port: int = 0
    i2c_bitrate_khz: int = 100


class HardwareBackend:
    """Future real-hardware replacement for PLECS/LTspice/SIMPLIS backends."""

    def __init__(self, config: HardwareBackendConfig | None = None):
        self.config = config or HardwareBackendConfig()
        self.testbench = HardwareTestbench(self.config)

    def setup(self) -> None:
        """Connect instruments that are safe to connect without energizing the board."""

        self.testbench.connect_function_generator()
        self.testbench.connect_board_controller()

    def run_iteration(self, pid: PidParameters) -> HardwareIterationResult:
        """Run one hardware PID experiment.

        Planned sequence:
        1. Program PID into the board controller.
        2. Enable supply with current and voltage limits.
        3. Trigger/load-step the board.
        4. Capture scope waveform as Time, IL, Vout.
        5. Optionally run loop-gain/Bode measurement.
        6. Shut down or return to a known safe state.
        """

        raise NotImplementedError("Scope, supply, and board controller drivers are not connected yet.")

    def close(self) -> None:
        self.testbench.close()
