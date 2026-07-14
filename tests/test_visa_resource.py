from __future__ import annotations

import unittest
from unittest.mock import patch

from hardware.instruments import visa_resource


class _FakeInstrument:
    def __init__(self, name: str) -> None:
        self.name = name
        self.timeout = 0
        self.read_termination = ""
        self.write_termination = ""
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def query(self, command: str) -> str:
        if self.closed:
            raise RuntimeError("instrument is closed")
        return f"{self.name}:{command}"


class _FakeResourceManager:
    def __init__(self) -> None:
        self.session = 1
        self.close_count = 0
        self.instruments: list[_FakeInstrument] = []

    def open_resource(self, name: str) -> _FakeInstrument:
        instrument = _FakeInstrument(name)
        self.instruments.append(instrument)
        return instrument

    def close(self) -> None:
        self.close_count += 1
        self.session = None


class _FakePyvisa:
    def __init__(self) -> None:
        self.manager = _FakeResourceManager()
        self.resource_manager_calls = 0

    def ResourceManager(self, backend=None) -> _FakeResourceManager:
        self.resource_manager_calls += 1
        return self.manager


class VisaResourceManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        visa_resource.close_shared_resource_managers()

    def tearDown(self) -> None:
        visa_resource.close_shared_resource_managers()

    def test_closing_one_instrument_keeps_shared_manager_and_other_session_alive(self) -> None:
        pyvisa = _FakePyvisa()
        with patch.object(visa_resource, "_load_pyvisa", return_value=pyvisa):
            first = visa_resource.VisaInstrument("USB::FIRST").connect()
            second = visa_resource.VisaInstrument("USB::SECOND").connect()

            first.close()

            self.assertEqual(pyvisa.resource_manager_calls, 1)
            self.assertEqual(pyvisa.manager.close_count, 0)
            self.assertEqual(second.query("*IDN?"), "USB::SECOND:*IDN?")

            second.close()
            self.assertEqual(pyvisa.manager.close_count, 0)

        visa_resource.close_shared_resource_managers()
        self.assertEqual(pyvisa.manager.close_count, 1)


if __name__ == "__main__":
    unittest.main()
