"""High-level hardware testbench skeleton.

This is the future replacement for the simulation backend. For now it only owns
the function generator connection; scope, Bode analyzer, power supply, and board
controller will be added beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .instruments.bode_analyzer import BodeScpiClient
from .instruments.board_controller import BoardController, BoardControllerConfig, create_board_controller
from .instruments.function_generator import FunctionGenerator
from .instruments.i2c_adapters import create_i2c_adapter
from .instruments.power_supply import KeysightN5700PowerSupply
from .instruments.visa_resource import find_first_usb_resource, list_visa_resources


@dataclass
class HardwareTestbenchConfig:
    function_generator_resource: Optional[str] = None
    power_supply_resource: Optional[str] = None
    visa_timeout_ms: int = 5000
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


class HardwareTestbench:
    def __init__(self, config: HardwareTestbenchConfig | None = None):
        self.config = config or HardwareTestbenchConfig()
        self.function_generator: Optional[FunctionGenerator] = None
        self.power_supply: Optional[KeysightN5700PowerSupply] = None
        self.bode: Optional[BodeScpiClient] = None
        self.board: Optional[BoardController] = None

    def connect_function_generator(self) -> FunctionGenerator:
        resource = self.config.function_generator_resource
        if resource is None:
            resource = find_first_usb_resource(list_visa_resources())
        if resource is None:
            raise RuntimeError("No USB VISA instrument found.")

        fg = FunctionGenerator(resource, timeout_ms=self.config.visa_timeout_ms)
        fg.connect()
        self.function_generator = fg
        return fg

    def connect_power_supply(self) -> KeysightN5700PowerSupply:
        resource = self.config.power_supply_resource
        if resource is None:
            resource = _find_keysight_n5700_resource(list_visa_resources())
        if resource is None:
            raise RuntimeError("No Keysight N5700/N5767A VISA resource found.")

        supply = KeysightN5700PowerSupply(
            resource,
            timeout_ms=self.config.visa_timeout_ms,
            model_voltage_limit_v=self.config.power_supply_voltage_limit_v,
            model_current_limit_a=self.config.power_supply_current_limit_a,
            configured_parallel_units=self.config.power_supply_parallel_units,
        )
        supply.connect()
        self.power_supply = supply
        return supply

    def connect_bode(self) -> BodeScpiClient:
        bode = BodeScpiClient(
            host=self.config.bode_scpi_host,
            port=self.config.bode_scpi_port,
            timeout_ms=self.config.bode_timeout_ms,
        )
        bode.connect()
        self.bode = bode
        return bode

    def connect_board_controller(self) -> BoardController:
        adapter_kwargs = {}
        if self.config.board_i2c_adapter.lower() == "aardvark":
            adapter_kwargs = {
                "port": self.config.aardvark_port,
                "bitrate_khz": self.config.i2c_bitrate_khz,
            }
        adapter = create_i2c_adapter(self.config.board_i2c_adapter, **adapter_kwargs)
        board = create_board_controller(
            self.config.board_controller_kind,
            adapter,
            BoardControllerConfig(
                address=self.config.board_i2c_address,
                name=self.config.board_name,
            ),
        )
        board.connect()
        self.board = board
        return board

    def close(self) -> None:
        if self.function_generator is not None:
            self.function_generator.close()
            self.function_generator = None
        if self.power_supply is not None:
            self.power_supply.close()
            self.power_supply = None
        if self.bode is not None:
            self.bode.close()
            self.bode = None
        if self.board is not None:
            self.board.close()
            self.board = None


def _find_keysight_n5700_resource(resources: tuple[str, ...]) -> Optional[str]:
    for resource in resources:
        text = resource.upper()
        if "0X0957" in text:
            return resource
    return None
