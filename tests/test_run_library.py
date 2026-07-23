import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gui.server import AutotuneRunStore


class RunLibraryListingTest(unittest.TestCase):
    def test_existing_summary_does_not_load_full_run_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recent = root / "recent"
            saved = root / "saved"
            run_dir = recent / "Recent_2026-07-22_01"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "algorithm": "DRL",
                        "iteration_count": 2568,
                        "updated_at": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            # A library listing must not parse this potentially huge file.
            (run_dir / "run_status.json").write_text("not-json", encoding="utf-8")
            store = AutotuneRunStore(recent, saved, recent_limit=10)

            with mock.patch.object(store, "_read_json", wraps=store._read_json) as read_json:
                listed = store._list_kind(recent, "recent")

            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["iteration_count"], 2568)
            self.assertEqual(listed[0]["algorithm"], "DRL")
            read_paths = [Path(call.args[0]).name for call in read_json.call_args_list]
            self.assertEqual(read_paths, ["summary.json"])


if __name__ == "__main__":
    unittest.main()
