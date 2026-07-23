from __future__ import annotations

import unittest

from hardware.instruments.oscilloscope import TektronixOscilloscope


class _FakeTekResource:
    def __init__(self, *, history: bool = True, record_length: int = 656_250) -> None:
        self.history = history
        self.record_length = record_length
        self.writes: list[str] = []

    def write(self, command: str) -> None:
        self.writes.append(command)
        normalized = command.strip().upper()
        if normalized.startswith("HORIZONTAL:HISTORY:STATE "):
            self.history = normalized.endswith(" ON")
        if "RECORDLENGTH " in normalized:
            self.record_length = int(normalized.rsplit(" ", 1)[1])

    def query(self, command: str) -> str:
        normalized = command.strip().upper()
        if normalized == "HORIZONTAL:HISTORY:STATE?":
            return "1" if self.history else "0"
        if "RECORDLENGTH?" in normalized or normalized == "HOR:RECO?":
            return str(self.record_length)
        if normalized == "WFMOUTPRE:NR_PT?":
            return str(self.record_length)
        raise RuntimeError(f"Unsupported query: {command}")


class TektronixAutotuneMemoryGuardTest(unittest.TestCase):
    def _scope(self, resource: _FakeTekResource) -> TektronixOscilloscope:
        scope = TektronixOscilloscope("FAKE::SCOPE")
        scope._inst = resource
        return scope

    def test_guard_disables_history_and_caps_record_length(self) -> None:
        resource = _FakeTekResource(history=True, record_length=656_250)
        scope = self._scope(resource)

        result = scope.prepare_autotune_acquisition(max_record_length=250_000)

        self.assertTrue(result["history_before"])
        self.assertFalse(result["history_after"])
        self.assertEqual(result["record_length_before"], 656_250)
        self.assertEqual(result["record_length_after"], 250_000)
        self.assertFalse(resource.history)
        self.assertEqual(resource.record_length, 250_000)
        self.assertIn("ACQUIRE:MODE SAMPLE", resource.writes)
        self.assertIn("*CLS", resource.writes)

    def test_forced_recovery_changes_length_to_flush_then_restores_cap(self) -> None:
        resource = _FakeTekResource(history=False, record_length=250_000)
        scope = self._scope(resource)

        result = scope.prepare_autotune_acquisition(
            max_record_length=250_000,
            force_flush=True,
        )

        record_writes = [command for command in resource.writes if "RECORDLENGTH " in command.upper()]
        self.assertGreaterEqual(len(record_writes), 2)
        self.assertIn("100000", record_writes[0])
        self.assertIn("250000", record_writes[-1])
        self.assertTrue(result["force_flush"])
        self.assertEqual(resource.record_length, 250_000)

    def test_periodic_maintenance_does_not_grow_record_length(self) -> None:
        resource = _FakeTekResource(history=False, record_length=125_000)
        scope = self._scope(resource)
        scope._autotune_acquisition_count = 99

        result = scope.prepare_autotune_acquisition(
            max_record_length=250_000,
            maintenance_interval=100,
        )

        self.assertFalse(result["maintenance_due"])
        self.assertEqual(resource.record_length, 125_000)
        next_result = scope.prepare_autotune_acquisition(
            max_record_length=250_000,
            maintenance_interval=100,
        )
        self.assertTrue(next_result["maintenance_due"])
        self.assertEqual(resource.record_length, 125_000)
        self.assertIn("ACQUIRE:MODE SAMPLE", resource.writes)


if __name__ == "__main__":
    unittest.main()
