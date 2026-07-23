from __future__ import annotations

import json
import unittest

from gui.server import _compact_loaded_status_for_client, _incremental_status_for_client


class GuiStatusDeltaTest(unittest.TestCase):
    def _status(self, count: int = 4) -> dict:
        return {
            "state": "running",
            "history": [
                {"iteration": iteration, "metrics": {"score": float(iteration)}}
                for iteration in range(1, count + 1)
            ],
        }

    def test_matching_cursor_returns_only_new_iterations(self) -> None:
        payload = _incremental_status_for_client(
            self._status(),
            "after_iteration=2&history_token=run-token",
            "run-token",
        )
        self.assertTrue(payload["history_delta"])
        self.assertEqual([record["iteration"] for record in payload["history"]], [3, 4])
        self.assertEqual(payload["history_total"], 4)
        self.assertEqual(payload["history_last_iteration"], 4)
        self.assertEqual(payload["history_after_iteration"], 2)

    def test_token_or_cursor_mismatch_falls_back_to_full_history(self) -> None:
        token_mismatch = _incremental_status_for_client(
            self._status(),
            "after_iteration=2&history_token=old-run",
            "new-run",
        )
        cursor_ahead = _incremental_status_for_client(
            self._status(),
            "after_iteration=99&history_token=new-run",
            "new-run",
        )
        self.assertFalse(token_mismatch["history_delta"])
        self.assertFalse(cursor_ahead["history_delta"])
        self.assertEqual(len(token_mismatch["history"]), 4)
        self.assertEqual(len(cursor_ahead["history"]), 4)

    def test_loaded_archive_history_drops_diagnostics_but_keeps_artifact_references(self) -> None:
        status = {
            "history": [
                {
                    "iteration": 1,
                    "waveform": {"time_s": list(range(1000)), "vout_v": list(range(1000))},
                    "write_results": {"large": list(range(1000))},
                    "optimizer_metadata": {"diagnostics": list(range(1000))},
                    "bode_result": {
                        "data_file": "data/bode.npz",
                        "frequency_hz": list(range(1000)),
                        "magnitude_db": list(range(1000)),
                    },
                    "scope_result": {
                        "capture_id": "capture-1",
                        "waveforms": [
                            {
                                "source": "CH3",
                                "data_file": "data/scope.npz",
                                "x": list(range(1000)),
                                "y": list(range(1000)),
                            }
                        ],
                    },
                }
            ]
        }
        before_bytes = len(json.dumps(status))
        compact = _compact_loaded_status_for_client(status)
        record = compact["history"][0]
        self.assertEqual(record["write_results"], {})
        self.assertEqual(record["optimizer_metadata"], {})
        self.assertEqual(record["waveform"]["time_s"], [])
        self.assertEqual(record["bode_result"]["data_file"], "data/bode.npz")
        self.assertNotIn("frequency_hz", record["bode_result"])
        self.assertEqual(record["scope_result"]["waveforms"][0]["data_file"], "data/scope.npz")
        self.assertNotIn("x", record["scope_result"]["waveforms"][0])
        self.assertLess(len(json.dumps(compact)), before_bytes // 10)


if __name__ == "__main__":
    unittest.main()
