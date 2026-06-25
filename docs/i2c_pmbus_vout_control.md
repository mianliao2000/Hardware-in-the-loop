# I2C / PMBus Vout Control Notes

This document records the current approach for controlling the Infineon
XDPE1A2G5C multiphase buck controller over I2C/PMBus. It is meant to give a
new developer enough context to understand the communication path, the PMBus
commands involved in Vout control, and why writing `VOUT_COMMAND` alone is not
always enough to change the real output voltage.

## Communication Path

The current hardware path is:

```text
PC Python / GUI
  -> Infineon XDP USB dongle
    -> I2C / SMBus
      -> XDPE1A2G5C controller
        -> Loop A / Loop B buck output
```

PMBus is a power-management command protocol that runs on top of I2C/SMBus.
I2C/SMBus provides the physical transfer. PMBus defines which command code to
read or write, what each command means, and how the payload bytes are encoded.

On the current Yosemite board, XDP Designer reports the active controller as:

```text
I2C address: 0x5E
Loop A: PAGE 0
Loop B: PAGE 1
```

The automation code currently allows Vout writes only on `PAGE 0 / Loop A`.
Loop B should stay disabled until its safety limits and intended use are
confirmed.

## Key PMBus Commands

| Command | Code | Access | Purpose |
| --- | --- | --- | --- |
| `PAGE` | `0x00` | read/write byte | Selects the loop. `0` is Loop A, `1` is Loop B. |
| `OPERATION` | `0x01` | read/write byte | Controls output state and selects the active voltage-control source. |
| `ON_OFF_CONFIG` | `0x02` | read/write byte | Defines how the CONTROL pin and `OPERATION` command turn the unit on/off. |
| `VOUT_MODE` | `0x20` | read byte | Defines how Vout-related commands are encoded. |
| `VOUT_COMMAND` | `0x21` | read/write word | PMBus nominal target Vout. |
| `VOUT_MARGIN_HIGH` | `0x25` | read/write word | Target voltage used when margin-high mode is commanded by `OPERATION`. |
| `VOUT_MARGIN_LOW` | `0x26` | read/write word | Target voltage used when margin-low mode is commanded by `OPERATION`. |
| `STATUS_WORD` | `0x79` | read word | Summary status and fault bits. |
| `READ_VOUT` | `0x8B` | read word | Actual Vout telemetry. |
| `READ_IOUT` | `0x8C` | read word | Actual Iout telemetry. |

Do not send `STORE_*` commands during tuning. `STORE_*` commands copy settings
to non-volatile memory, which is not appropriate for early bring-up or
autotuning experiments.

## Vout Encoding

Always read `VOUT_MODE` before encoding or decoding Vout commands.

The current board reports:

```text
VOUT_MODE = 0x17
mode      = linear
exponent  = -9
```

This means Vout uses PMBus LINEAR16 format:

```text
voltage = raw * 2^exponent
raw     = round(voltage / 2^exponent)
```

With `exponent = -9`, the resolution is:

```text
2^-9 V = 1/512 V = 0.001953125 V
```

Example:

```text
0.85 V -> raw = round(0.85 * 512) = 435 = 0x01B3
decoded = 435 / 512 = 0.849609375 V
```

Therefore, setting `0.85 V` from the GUI or a script and reading back
`VOUT_COMMAND = 0.849609375 V` is normal quantization behavior.

## Correct Vout Write Sequence

A before/after PMBus snapshot diff against XDP Designer showed that XDP
Designer does not only write `VOUT_COMMAND`. It also returns `OPERATION` to
`0x80`, which makes the output follow the PMBus nominal voltage target.

Recommended sequence:

```text
1. PAGE = 0
2. Read VOUT_MODE
3. Encode the target voltage using VOUT_MODE
4. Write VOUT_COMMAND = encoded target voltage
5. Write OPERATION = 0x80
6. Poll READ_VOUT until it is close to the target
```

Example: set Loop A Vout to approximately `0.85 V`.

```text
PAGE         = 0
VOUT_COMMAND = 0.85 V  -> raw 0x01B3
OPERATION    = 0x80
```

Expected readback is similar to:

```text
VOUT_COMMAND = 0.849609375 V
READ_VOUT    = 0.84765625 V
OPERATION    = 0x80
```

`VOUT_COMMAND` is the target setpoint. `READ_VOUT` is measured telemetry. They
should be close, but they do not need to be bit-for-bit identical.

## OPERATION 0x80 vs 0xB0

`OPERATION` is PMBus command `0x01`. It controls both the output state and the
active voltage-control source.

### `OPERATION = 0x80`

```text
0x80 = 1000 0000b
bit 7   = 1   output on
bit 5:4 = 00  use PMBus VOUT_COMMAND / nominal voltage
```

Meaning:

```text
Turn the output on and use VOUT_COMMAND as the target voltage.
```

This is the mode used by the current GUI Vout control path.

### `OPERATION = 0xB0`

```text
0xB0 = 1011 0000b
bit 7   = 1   output on
bit 5:4 = 11  high-speed interface controls output voltage
```

The XDPE1A2G5C datasheet describes `OPERATION` as controlling whether the
output voltage is controlled by `VOUT_COMMAND` or by the high-speed interface.

On this board, `0xB0` can be treated as:

```text
The output is on, but the target voltage is not controlled by PMBus
VOUT_COMMAND. It is controlled by the high-speed interface / VID / SVID / SVI3
path instead.
```

This explains the earlier failure mode: when `OPERATION = 0xB0`, the
`VOUT_COMMAND` register can still be written and read back, but the real output
voltage may not follow it.

## Why Use PMBus Snapshot Diffing

At first, we did not know exactly which commands XDP Designer wrote when it
changed Vout. Guessing the write sequence can miss important state, such as
`OPERATION`.

The safer method was a read-only snapshot diff:

```text
1. Use Python to read a known set of PMBus commands and save a before snapshot.
2. Use XDP Designer to change Vout manually.
3. Use Python to read the same PMBus commands and save an after snapshot.
4. Diff before vs after.
```

The important observed changes were:

```text
VOUT_COMMAND: 0.935546875 V -> 0.900390625 V
OPERATION:    0xB0         -> 0x80
READ_VOUT:    0.93359375 V -> 0.900390625 V
```

This confirmed that PMBus Vout writes need `VOUT_COMMAND` followed by
`OPERATION = 0x80` to make the real output follow the PMBus setpoint.

Related script:

```powershell
python scripts\snapshot_xdpe_pmbus.py --adapter xdp --address 0x5E --pages 0 1 --output snapshots\xdpe_before_xdp.json
python scripts\snapshot_xdpe_pmbus.py --adapter xdp --address 0x5E --pages 0 1 --output snapshots\xdpe_after_xdp.json
python scripts\snapshot_xdpe_pmbus.py --diff snapshots\xdpe_before_xdp.json snapshots\xdpe_after_xdp.json
```

## Code Locations

Low-level PMBus / I2C:

```text
hardware/instruments/pmbus.py
hardware/instruments/i2c_adapters.py
hardware/instruments/board_controller.py
```

GUI API:

```text
gui/server.py
```

Vout write path:

```text
gui/server.py::_set_vout()
  -> board.set_vout_command(voltage, page=0)
  -> board.set_operation(0x80, page=0)
  -> poll board.read_vout(page=0)
```

Read-only PMBus snapshot / diff:

```text
scripts/snapshot_xdpe_pmbus.py
```

## Safety Rules

1. Control only `PAGE 0 / Loop A` by default.
2. Before writing Vout, verify input supply state, current limit, scope probes,
   and load condition.
3. Do not automatically turn bench instrument outputs on or off unless the user
   explicitly requests it.
4. Do not send `STORE_*` commands.
5. Do not modify PID registers until the XDPE PID register map is confirmed.
6. XDP Designer and Python cannot control the XDP dongle at the same time.
   Close XDP Designer before using the GUI/API.
7. After every Vout write, verify `READ_VOUT`; do not trust only
   `VOUT_COMMAND` readback.

## Future PID Register Bring-Up

Use the same diffing approach when PID parameter writes are added:

```text
1. Change one PID parameter manually in XDP Designer.
2. Capture before/after PMBus or internal-register snapshots.
3. Diff the snapshots to find the real write path and encoding.
4. Implement the same volatile write in Python.
5. Read back and verify. Do not write to NVM.
```

This avoids guessing the register map and reduces the risk of writing the wrong
settings to the board.
