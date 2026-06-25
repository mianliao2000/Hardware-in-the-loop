# TPU A1 Hardware PID Auto-Tuner

This project is the hardware-testbench version of the simulation auto-tuner.
The first milestone is instrument connectivity over USB, starting with a
function generator.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

You also need a VISA backend installed on Windows, usually one of:

- NI-VISA
- Keysight IO Libraries Suite
- TekVISA

## First Function Generator Check

List visible VISA resources:

```powershell
python scripts/list_visa_resources.py
```

Connect to the first USB instrument and read its identity:

```powershell
python scripts/connect_function_generator.py
```

Or pass a resource explicitly:

```powershell
python scripts/connect_function_generator.py --resource "USB0::0x0957::0x2807::MY12345678::INSTR"
```

Try a safe output-off configuration:

```powershell
python scripts/connect_function_generator.py --resource "USB0::..." --configure-sine --frequency 1000 --amplitude 0.2 --offset 0
```

The script leaves output disabled unless `--enable-output` is provided.

## Board Controller Over I2C/PMBus

Infineon XDP Designer and ADI Power Studio are useful for manual validation and
exporting configuration, but the auto-tuner controls the board directly from
Python over I2C/PMBus. Keep tuning writes volatile until the exact controller,
register map, and safe limits are confirmed.

Validate the Python path with the mock adapter:

```powershell
python scripts/connect_board.py --adapter mock --address 0x40 --identify --status
```

When an I2C adapter is connected, start with read-only checks:

```powershell
python scripts/connect_board.py --adapter aardvark --address 0x40 --kind infineon_xdp --identify --status
```

The board script does not enable outputs or change PID values by default.
State-changing commands such as `--clear-faults` must be requested explicitly.

## Keysight N5767A Parallel Power Supply

The two N5767A supplies are physically wired in master/slave parallel mode. The
software talks to the master VISA resource only and does not configure the
parallel wiring, tracking, analog programming, or master/slave role.

Read the current state without changing output:

```powershell
python scripts/connect_power_supply.py --resource "USB0::0x0957::0xA407::US17M5136P::INSTR"
```

Set conservative limits while leaving output state unchanged:

```powershell
python scripts/connect_power_supply.py --resource "USB0::0x0957::0xA407::US17M5136P::INSTR" --voltage 12 --current-limit 1
```

Enable output only when voltage and current limit are both provided:

```powershell
python scripts/connect_power_supply.py --resource "USB0::0x0957::0xA407::US17M5136P::INSTR" --voltage 12 --current-limit 1 --output-on
```

## Infineon XDPE1A2G5C Vout Over PMBus

The XDPE1A2G5C controller uses PMBus over SMBus/I2C. For the Yosemite board,
the active controller address shown by XDP Designer is `0x5E`.

Relevant PMBus commands:

- `PAGE` `0x00`: page `0` = Loop A, page `1` = Loop B
- `VOUT_MODE` `0x20`: read this first to determine VOUT encoding
- `VOUT_COMMAND` `0x21`: set target Vout in the format selected by `VOUT_MODE`
- `READ_VOUT` `0x8B`: read Vout telemetry in the same VOUT format

Dry-run the encoding path:

```powershell
python scripts/set_xdpe_vout.py --adapter mock --address 0x5E --page 0 --voltage 0.95 --read --dry-run
```

When a supported I2C adapter is connected, read before writing:

```powershell
python scripts/set_xdpe_vout.py --adapter aardvark --address 0x5E --page 0 --read
```

The Total Phase `aardvark_py` package currently works with Python 3.5-3.8.
Use a Python 3.8 environment for Aardvark hardware access:

```powershell
py -3.8 -m pip install aardvark_py
py -3.8 scripts/list_aardvark.py
py -3.8 scripts/set_xdpe_vout.py --adapter aardvark --address 0x5E --page 0 --read
```

For the current Yosemite/XDPE setup, Vout control must follow the XDP Designer
PMBus sequence: write the operating-memory `VOUT_COMMAND`, then set
`OPERATION = 0x80` so the output follows the PMBus nominal target. See the
bring-up notes for the full reasoning:

```text
docs/i2c_pmbus_vout_control.md
```

The older standalone script writes only `VOUT_COMMAND` and is mainly useful for
encoding/readback checks:

```powershell
python scripts/set_xdpe_vout.py --adapter aardvark --address 0x5E --page 0 --voltage 0.95 --write
```

Do not use `STORE_*` commands during tuning; those copy settings to non-volatile
memory.

## Local PID Auto-Tuner GUI

The browser GUI lives in the standalone `gui/` folder. The Python server exposes
the instrument APIs and serves the React/Vite workbench build.

- `gui/server.py`: localhost API and React build server
- `gui/frontend/`: React/Vite/TypeScript workbench
- `hardware/tuning/`: PID autotuning framework and stub experiment runner

Build the frontend once after changing GUI code:

```powershell
cd gui\frontend
npm install
npm run build
cd ..\..
```

Start the local server with:

```powershell
python gui/server.py --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

The current auto-tuner GUI runs PID iterations with a placeholder experiment
runner. PID programming is deliberately disabled until the XDPE PID register map
is verified. The Vout panel still uses the XDP USB dongle path and performs
volatile PMBus writes only: `VOUT_COMMAND` followed by `OPERATION = 0x80`.
Close XDP Designer before using the Vout controls, because the XDP dongle is
exclusive and cannot be controlled by both programs at the same time.
