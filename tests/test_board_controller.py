from __future__ import annotations

import unittest

from hardware.instruments.board_controller import BoardController, XDPE_MOD0_LL_BW_ADDRESS


class Mod0LowLoadBandwidthTest(unittest.TestCase):
    def test_one_write_sets_ls_and_lr_to_the_same_value(self) -> None:
        controller = object.__new__(BoardController)
        memory = {XDPE_MOD0_LL_BW_ADDRESS: 0xA5A50000 | 74 | (84 << 7)}
        writes: list[tuple[int, int]] = []

        controller._read_xdpe_ahb_word = lambda address: memory[address]  # type: ignore[method-assign]

        def write(address: int, value: int) -> None:
            writes.append((address, value))
            memory[address] = value

        controller._write_xdpe_ahb_word = write  # type: ignore[method-assign]
        result = controller.set_mod0_ll_bandwidth(93, page=0)

        self.assertEqual(len(writes), 1)
        self.assertEqual(memory[XDPE_MOD0_LL_BW_ADDRESS] & 0x7F, 93)
        self.assertEqual((memory[XDPE_MOD0_LL_BW_ADDRESS] >> 7) & 0x7F, 93)
        self.assertTrue(result["readback"]["equal"])
        self.assertEqual(result["readback"]["value"], 93)

    def test_loop_b_is_rejected(self) -> None:
        controller = object.__new__(BoardController)
        with self.assertRaisesRegex(ValueError, "Loop A"):
            controller.set_mod0_ll_bandwidth(74, page=1)


if __name__ == "__main__":
    unittest.main()
