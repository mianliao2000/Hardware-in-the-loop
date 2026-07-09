# Hardware-in-the-Loop Power Tuning Lab Notes

This document summarizes the practical lessons learned while converting the original simulation autotuner idea into a real hardware experiment workbench. It is intended to help another developer reproduce the setup, understand the control paths, and avoid the failure modes we found during bring-up.

## System Goal

The project is a localhost-based hardware-in-the-loop workbench for tuning and observing a multiphase buck converter. The GUI coordinates several bench instruments and the board controller:

- Tektronix AFG31000 function generator
- Tektronix MSO58 oscilloscope
- OMICRON Bode 100 analyzer
- Keysight N5767A DC power supply pair in parallel master/slave mode
- Infineon XDPE1A2G5C board controller through XDP / PMBus

The PID autotuning path is scaffolded, but direct PID writes are still treated as explicit manual writes until the full register map and safe operating procedure are verified.

## Local GUI

Start the server from the repository root:

```powershell
python gui/server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/?tab=manual
```

The frontend is React/Vite/TypeScript. The backend is a lightweight Python HTTP server. The production frontend bundle is served by `gui/server.py`.

## Logging Strategy

High-rate telemetry polling can generate thousands of HTTP access lines. The server intentionally aggregates request logs and prints a compact summary approximately every 5 seconds instead of logging every single `/api/read` and `/api/tuning/status` request.

This keeps the terminal readable when Live Data runs at high rates.

## Scope Data File Format

Scope captures are saved in a compact NPZ format so long autotune runs do not create excessively large result folders.

Current format:

- One capture is saved as one `.npz` file.
- The time axis is not stored as a full array. The file stores `x_start`, `x_increment`, and `points`; the full time axis is reconstructed as `x_start + arange(points) * x_increment`.
- Each channel waveform is stored as `float32`, for example `y_CH1` and `y_CH3`.
- Metadata stores channel names, units, capture ID, timestamp, original point count, and transfer encoding.

This keeps the full waveform available for future analysis while avoiding repeated `float64` time arrays and per-channel duplicate files. Older per-channel files can be converted with:

```powershell
python scripts/compact_scope_npz.py results/autotune_runs/recent/Recent_YYYY-MM-DD_NN
```

## XDP / PMBus Board Control

The board is controlled through PMBus over the XDP dongle. The active controller address used during testing was:

```text
PMBus address: 0x5E
Page: 0
```

### VOUT Control

VOUT is controlled with the PMBus `VOUT_COMMAND` command. The device uses `VOUT_MODE` linear encoding; in the tested configuration the effective voltage resolution was `1 / 512 V`.

For example, a requested target near `0.93 V` is rounded to the nearest supported register voltage.

Important practical detail:

- User-entered Vout values are snapped to the nearest representable register value.
- Spinner up/down changes by one register step, not by an arbitrary decimal step.
- Real-time writing writes only when explicitly enabled.
- When real-time writing is turned on, the current Vout target box is immediately matched to hardware.

### PMBus Output Control

The standard PMBus `OPERATION` command at code `0x01` was used for output control experiments:

```text
OPERATION bit 7 = 1: output on
OPERATION bit 7 = 0: output off
```

The GUI exposes this as `PMBus Output Control` with `Enable` and `Disable`.

The earlier XDP-specific VREN high/low method was kept as a debugging reference during bring-up, but the preferred visible control path is PMBus `OPERATION`.

### STATUS_WORD

`STATUS_WORD` is the PMBus status summary word. It is useful for checking whether the controller reports faults such as undervoltage, overvoltage, current, temperature, communication, or generic fault states. It should be read after control actions that affect output enable or Vout.

## XDP Connection Recovery

XDP Designer often failed to reconnect automatically after a board power cycle unless the XDP USB dongle was physically replugged. The custom Python/Node PMBus path recovered more reliably because it tears down and recreates the USB bridge/session after failures. In practice:

- XDP Designer appears to keep stale state after the board disappears.
- The custom bridge can detect access errors, release its handle, and reconnect.
- Auto-reconnect is especially important when the DUT power rail is being intentionally toggled or faulted.

This is not necessarily proof of a vendor bug, but it is a practical limitation of the GUI workflow.

## Bode 100 Integration

The Bode 100 is controlled through the OMICRON SCPI TCP runner, normally reachable at:

```text
TCPIP::127.0.0.1::5025::SOCKET
```

If no listener exists, the backend tries to start:

```text
C:\Program Files\OMICRON\BodeAnalyzerSuite\OmicronLab.VectorNetworkAnalysis.ScpiRunner.exe
```

### Critical SCPI Lesson

The correct Gain/Phase measurement creation command is:

```text
:CALC:PAR:DEF GAIN, DEF
```

Not:

```text
:CALC:PAR:DEF GAINphase
```

`GAINphase` appears in OMICRON naming, but `GAIN, DEF` is the working SCPI command for creating the measurement.

Another important ordering detail:

1. Create the measurement with `:CALC:PAR:DEF GAIN, DEF`
2. Then set start/stop frequency, point count, sweep type, RBW, source level
3. Trigger the sweep
4. Read frequency and trace data

Defining the measurement after setting sweep parameters can reset the sweep configuration back to defaults.

### Tested Bode Settings

The default sweep configured in the GUI matches the Bode Analyzer Suite reference view:

```text
Start: 1 kHz
Stop: 1 MHz
Points: 201
RBW: 300 Hz
Source: 0 dBm
Sweep: logarithmic
Measurement: Gain / Phase
```

The GUI accepts compact frequency text such as `1k` and `1M`, then converts it to numeric Hz.

## Oscilloscope Integration

The MSO58 is controlled through VISA USB. The scope module can:

- Select channels to capture
- Select immediate measurements such as mean, pk-pk, frequency, RMS, min, max
- Capture waveform points and plot them in the GUI

The self-test path avoids enabling or disabling outputs. It only makes reversible setting changes and restores them.

## Function Generator Integration

The AFG31000 is controlled through VISA USB. The GUI supports common modes:

- Square
- Pulse
- DC
- Sine

For square and pulse modes, the GUI can set high/low levels directly. A known working example was:

```text
100 kHz square wave
Low: 365 mV
High: 2 V
```

The backend does not toggle output state unless a dedicated control is intentionally added.

## Power Supply Integration

The Keysight N5767A pair is physically configured in parallel master/slave mode. Software should talk to the master supply only.

Safety rule:

- Do not change parallel topology from software.
- Do not automatically enable or disable output in routine tests.
- Do not live-write voltage/current limits.

The GUI supports:

- Read measured voltage/current
- Read voltage setpoint/current limit
- Manually set voltage and current limit only when `Write Supply` is clicked
- Live read at a user-controlled rate

Live read is read-only and does not update the write-input boxes. This prevents accidental writes and avoids confusing readback state with pending user commands.

## Self Testing

The Self Testing tab checks instrument communication without opening or closing outputs. Each module can be tested individually. The tests should:

- Query identity or status
- Make a small reversible setting change when appropriate
- Restore the original setting
- Report pass/fail immediately for each instrument

This is safer than one monolithic test because high-power hardware may be connected.

## Live Data

Live Data displays:

```text
Vout
Iout
Vout_Command
```

The visible values and plotted traces use a 0.5 second moving average. Raw PMBus samples remain unchanged internally.

The live graph keeps a bounded history window so old samples are dropped instead of growing memory indefinitely.

## GUI Design Notes

Manual Tuning is the main hardware workbench. Each module is collapsible so the operator can keep only relevant instruments open.

The current practical order during experiments is:

1. Verify XDP connection
2. Verify Live Data readback
3. Set Vout only when safe
4. Configure AFG
5. Capture scope data
6. Run Bode 100 sweep
7. Monitor or set power supply limits only by explicit action

## Safety Notes

- Treat all output-enable and power-supply writes as explicit operator actions.
- Default to read-only behavior for live panels.
- Keep hardware PID writes opt-in until register semantics are fully verified.
- For PMBus register reverse engineering, always snapshot before and after a vendor GUI action, then diff the register map.
- Restore reversible self-test settings before reporting success.

## Debugging Checklist

If an instrument does not respond:

1. Confirm the physical cable and power state.
2. Confirm VISA resource visibility for USB instruments.
3. Confirm no vendor GUI is holding exclusive access.
4. For Bode 100, confirm the SCPI TCP runner is listening on port 5025.
5. For XDP, restart the bridge session before physically replugging the dongle.
6. For PMBus writes, read back both the command register and measured telemetry.
7. Check `STATUS_WORD` after output-control or Vout operations.

## Known Open Work

- Confirm the complete XDPE PID register map before enabling automated PID programming.
- Add explicit safety interlocks around output enable flows.
- Add persistent experiment logs and exported reports.
- Add automatic Bode/scope/power-supply data association per tuning iteration.
- Add scripted calibration and fixture metadata tracking.
