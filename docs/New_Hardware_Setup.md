# New Hardware Setup and Migration Playbook

## Purpose

This document is a reusable engineering playbook for bringing new laboratory hardware into the Google Cloud Power Auto Tuner. It records the architecture, connection workflow, failure modes, debugging lessons, and safety rules learned while integrating the current Infineon controller board, XDP USB dongle, Bode 100, Tektronix scope and function generator, and Keysight power supply.

Use it when replacing any of the following:

- the power controller, for example Infineon to Renesas or MPS;
- the USB-to-I2C/PMBus adapter;
- the power supply;
- the oscilloscope;
- the function generator or electronic load control source;
- the frequency-response analyzer;
- the physical board while keeping the same instruments.

The objective is not merely to make a new device answer one command. A successful migration preserves the complete workflow:

1. Self Testing can prove communication without changing output state.
2. Manual Tuning can read, write, plot, and recover safely.
3. Full Data Acquisition can coordinate all instruments in a deterministic order.
4. PID Auto-Tune can run many iterations, survive recoverable trips, and save reproducible evidence.
5. Every hardware action has a clear owner, timeout, cleanup path, and error message.

## Non-Negotiable Safety Rules

These rules apply before any new driver is connected to real power hardware.

1. Never toggle an output as part of a connection test.
2. Never infer that an output is off from a low voltage reading. Query its output state explicitly.
3. Read and preserve the current setting before a reversible Self Test change.
4. Restore the setting in `finally`, even if verification fails.
5. Disable the function generator in `finally` after every transient experiment.
6. Keep power-supply writes explicit. Background readback must not perform writes.
7. Apply address, page, loop, voltage, current, and raw-field bounds in the backend, not only in the GUI.
8. For controller writes, use read-modify-write when multiple fields share one register.
9. Verify write readback before starting Bode or transient acquisition.
10. Treat stale sessions, USB ownership conflicts, and timeouts as different failures. Recovery must match the failure.
11. A failed candidate must not silently become a valid optimization point.
12. Any recovery that can energize hardware must be explicit and logged.

## System Architecture

The application is intentionally layered. New hardware should replace one layer without leaking vendor-specific commands into React or the tuning search.

```text
React GUI
  -> JSON API in gui/server.py
  -> orchestration and safety policy
  -> hardware/testbench.py and tuning runner
  -> instrument/controller driver
  -> transport adapter (VISA, TCP, USB, I2C, PMBus)
  -> physical instrument or controller
```

The current board path is:

```text
GUI/backend
  -> BoardController
  -> PmbusDevice
  -> XdpNodeUsbI2cAdapter
  -> hardware/instruments/xdp_usb_bridge.js
  -> Infineon XDP USB dongle
  -> PMBus/I2C
  -> XDPE1A2G5C controller
```

The current bench-instrument path is:

```text
GUI/backend
  -> device-specific Python driver
  -> VisaInstrument
  -> PyVISA resource manager
  -> USBTMC or vendor VISA transport
  -> instrument
```

The Bode 100 path is special:

```text
GUI/backend
  -> BodeScpiClient
  -> TCPIP::127.0.0.1::5025::SOCKET
  -> Omicron ScpiRunner
  -> Bode Analyzer Suite/driver
  -> Bode 100 USB hardware
```

## Repository Map

### Frontend

- `gui/frontend/src/main.tsx`: page state, panels, instrument controls, Auto-Tune controls, plots, result loading, and API calls.
- `gui/frontend/src/api.ts`: typed HTTP request functions. Keep endpoint details here instead of scattering `fetch()` calls through components.
- `gui/frontend/src/types.ts`: frontend copies of API contracts. Update this whenever a backend response or request changes.
- `gui/frontend/src/styles.css`: layout, themes, responsive sizing, controls, charts, and result states.
- `gui/frontend/dist/`: built frontend served by Python. Source changes are not visible until `npm run build` updates this directory.

### HTTP and orchestration

- `gui/server.py`: HTTP routes, connection caches, hardware action helpers, coordinated acquisitions, result storage, PNG generation, and LLM integration.
- `ServerHardwareExperimentRunner`: executes one Auto-Tune candidate against real hardware.
- `AutotuneRunStore`: manages recent/permanent runs, history, artifacts, and friendly result names.
- `_connect_board()`: creates an adapter and controller for board actions.
- `_run_bode_sweep()`: obtains a Bode connection, configures a sweep, transfers data, calculates metrics, and creates artifacts.
- `_capture_scope()`: configures single acquisition, transfers selected channels, stores complete data, and creates display artifacts.
- `_set_function_generator()` and `_set_power_supply()`: explicit instrument writes.

### Hardware abstraction

- `hardware/testbench.py`: owns the set of bench instruments and provides connect/close methods.
- `hardware/instruments/visa_resource.py`: shared PyVISA resource managers and the base `VisaInstrument` class.
- `hardware/instruments/function_generator.py`: Tektronix AFG waveform configuration and output control.
- `hardware/instruments/power_supply.py`: Keysight N5700 voltage/current configuration, readback, and output control.
- `hardware/instruments/oscilloscope.py`: Tektronix channel, trigger, acquisition, and binary waveform transfer.
- `hardware/instruments/bode100.py`: Bode 100 discovery and ScpiRunner startup.
- `hardware/instruments/bode_analyzer.py`: Bode SCPI commands and stability calculations.
- `hardware/instruments/pmbus.py`: vendor-neutral PMBus transactions and numeric formats.
- `hardware/instruments/i2c_adapters.py`: adapter implementations and XDP persistent bridge lifecycle.
- `hardware/instruments/board_controller.py`: controller-level telemetry, command, and raw register field operations.
- `hardware/instruments/self_test.py`: reversible connection checks and cleanup.

### Tuning

- `hardware/tuning/runner.py`: session state, background execution, pause/resume, candidate experiment order, and recovery.
- `hardware/tuning/search.py`: coarse combinations and refined candidate generation.
- `hardware/tuning/analyzer.py`: transient metrics, filtered waveform analysis, and penalty inputs.
- `hardware/tuning/pid_programmer.py`: PID candidate write interface.
- `hardware/tuning/models.py`: request, candidate, metric, and history data structures.

## Connection Ownership Model

Most intermittent hardware bugs are ownership bugs rather than command bugs.

For every device, define one connection owner and one cleanup owner:

| Device | Connection owner | Shared/cached | Required cleanup |
| --- | --- | --- | --- |
| XDP dongle | persistent Node bridge | yes | close adapter or reset bridge on stale USB access |
| Bode 100 | server Bode connection cache | yes | unlock session; drop stale client only on error |
| Scope | server scope connection cache | yes | preserve live VISA session; drop stale handle on error |
| AFG | driver action/testbench | short or cached by current flow | always turn output off after experiment |
| Power supply | driver action/testbench | short or cached by current flow | preserve output unless user explicitly changes it |

Do not allow Self Testing and Auto-Tune to keep separate live handles to an exclusive USB device. Self Testing must release its board adapter before returning its JSON response.

## VISA Device Integration

### Discovery

Start by listing VISA resources with `list_visa_resources()` in `visa_resource.py`. Record:

- the exact resource string;
- `*IDN?` response;
- USB vendor ID and product ID;
- serial number;
- VISA backend used (`@py`, NI-VISA, or vendor VISA);
- query/write termination requirements;
- timeout needed for the slowest command.

Never rely solely on resource order. Match a known VID/PID, serial number, or identity substring.

### Base class contract

New VISA drivers should inherit `VisaInstrument` and use:

- `connect()` to open and configure the resource;
- `close()` to release it;
- `write()` for commands with no response;
- `query()` for commands that return text.

The shared resource-manager code prevents repeated PyVISA initialization, but each instrument session can still become stale. A cache hit is not proof that the underlying session handle remains valid.

### Common VISA failures

#### Invalid session handle

Symptom:

```text
VI_ERROR_INV_OBJECT
Invalid session handle. The resource might be closed.
```

Cause: a cached Python object refers to a VISA session closed by another path, USB reset, instrument reboot, or vendor software.

Recovery:

1. Catch only recognized stale-session errors.
2. Remove the cached connection.
3. Close the old object defensively.
4. Reopen once.
5. Retry the operation once, not indefinitely.

#### First operation is slow

The first action may initialize USBTMC, load a VISA backend, enumerate resources, or wake the instrument parser. A warm connection endpoint is useful for the scope. Do not hide a ten-second first action with a huge timeout; measure each stage separately.

#### Query unterminated (`-420`)

This usually means query/write sequencing is wrong. Typical causes are issuing a second query before consuming the first response, mixing raw and line reads, or using the wrong termination. Keep one command-response pair in flight per instrument.

## Function Generator Migration

The current `FunctionGenerator` driver supports waveform configuration and explicit output state. A replacement driver must provide at least:

```python
connect()
identify()
configure_square(channel, frequency_hz, low_v, high_v, ...)
configure_pulse(channel, frequency_hz, low_v, high_v, width_s, ...)
configure_sine(...)
configure_dc(...)
set_output(channel, enabled)
readback(channel)
close()
```

Migration checklist:

1. Confirm whether amplitude is expressed as Vpp, peak, RMS, high/low, or amplitude/offset.
2. Confirm whether the voltage unit command changes interpretation of existing values.
3. Verify channel numbering and output command syntax.
4. Verify frequency suffix parsing in the frontend and numeric Hz at the backend boundary.
5. Read back waveform, frequency, levels, and output state after write.
6. Make output enable and disable separate explicit operations.
7. In every transient experiment, enable immediately before scope acquisition and disable in `finally`.
8. Do not leave output enabled after an exception, pause, or trip recovery.

For full acquisition, the scope window is derived from the active function-generator frequency. Therefore, the frontend setting used to configure the generator must be the same value passed to `_capture_scope()`.

## Power Supply Migration

The current Keysight driver separates readback from writes. Preserve this behavior for any replacement supply.

Minimum interface:

```python
connect()
identify()
read_measured_voltage()
read_measured_current()
read_voltage_setpoint()
read_current_limit()
read_output_enabled()
set_voltage(value)
set_current_limit(value)
set_output(enabled)
close()
```

Important rules:

- A live-read toggle must call read methods only.
- Voltage and current-limit fields are written only when the user presses Write.
- A connection test may alter a harmless setpoint only while output is confirmed off, and must restore the exact original value.
- Never use a measurement of nearly zero volts as evidence that output is disabled.
- Clamp setpoints to both the instrument rating and the test fixture rating.
- Log whether a write reached the instrument and whether readback matched.

When changing supply vendors, check command aliases, channel selection, protection-clear behavior, output coupling, and whether remote sense affects readback.

## Oscilloscope Migration

### Required behavior

The scope abstraction must support:

- enabling selected channels and disabling unselected channels;
- selecting an edge-trigger source and slope;
- setting horizontal scale/window and trigger position;
- run, stop, and single acquisition;
- determining acquisition completion;
- binary waveform transfer;
- waveform preamble/scaling metadata;
- optional scalar measurements.

### Current acquisition sequence

For Auto-Tune, the safe sequence is:

1. Reuse or open the VISA session.
2. Enable required channels, normally CH1 and CH3.
3. Set CH1 rising-edge trigger.
4. Set the horizontal window from function-generator frequency.
5. Put the first trigger near the left side of the window.
6. Arm a single acquisition.
7. Wait for acquisition completion or force a trigger after a bounded delay.
8. Transfer every selected channel.
9. Store full raw data.
10. Create reduced display data and PNG separately.

### Binary waveform transfer

Use Tektronix RIBinary/RPBinary or the vendor equivalent. ASCII transfer converts every sample to text and is dramatically slower for large records.

The generic binary process is:

1. Query waveform preamble (`x0`, `dx`, `y offset`, `y scale`, byte width, byte order).
2. Request the curve as an IEEE 488.2 binary block.
3. Parse the raw byte buffer with `numpy.frombuffer()`.
4. Convert ADC codes to engineering units using the preamble.
5. Reconstruct x as `x0 + arange(n) * dx` only when needed.

For stored captures, this repository uses one capture per NPZ file, `x0/dx/n` instead of a full x array, and float32 y arrays. This preserves time range and waveform shape while reducing disk use.

### Scope pitfalls

- Reading a disabled channel can fail or return stale data. Enable selected channels first.
- Reading while acquisition is running can produce inconsistent records. Use single acquisition for experiments.
- A fixed x-axis can compress data at one edge. Derive limits from actual `x0/dx/n`.
- Plotting one million points directly can overflow the browser stack. Keep full data in storage and use edge-focused display reduction.
- The first capture is slower because of session and acquisition setup. Cache configuration signatures only when hardware state is known unchanged.
- If vendor software touched the scope, invalidate the cache and reconfigure.

## Bode 100 Integration

### Why the GUI may fail while Bode Analyzer Suite works

Bode Analyzer Suite talking to the device does not imply a TCP SCPI listener exists. The Python client connects to `127.0.0.1:5025`, which is provided by Omicron ScpiRunner. Opening or closing the Suite can initialize the driver/runner and make later tests pass.

Required configuration:

- `BODE100_SERIAL`, for example `Bode100R2-PN287H`;
- valid ScpiRunner path;
- local TCP port, normally 5025;
- firewall permission for localhost;
- no competing client holding an exclusive lock.

### Sweep sequence

1. Ensure ScpiRunner is listening.
2. Connect `BodeScpiClient` to the TCP socket.
3. Query identity.
4. Lock the measurement session if required by the API.
5. Configure gain/phase mode, start/stop frequency, point count, receiver bandwidth, and source level.
6. Run the sweep.
7. Transfer frequency, magnitude, and phase arrays.
8. Unlock in `finally`.
9. Keep the TCP session only if it is healthy.

Source level in the GUI is Vpp. Convert it at the backend boundary if the Omicron SCPI API expects dBm and state the assumed impedance.

### Bode pitfalls

- A second test can fail if the previous lock/session was not released.
- Reopening ScpiRunner every iteration destroys performance.
- A cached object may hold a dead TCP/VISA handle. Detect stale-session errors and reconnect once.
- Gain and phase must be aligned to the same frequency vector.
- Multiple 0 dB crossovers indicate a potentially invalid loop response; treat this explicitly instead of selecting one silently.
- Gain margin conventions in this project may differ from textbook `-180 deg` conventions. Document the exact phase reference used by the fixture.

## PMBus and Controller Integration

### PMBus transport versus controller semantics

`PmbusDevice` provides generic PMBus transactions:

- `send_byte()`;
- `write_byte()` / `read_byte()`;
- `write_word()` / `read_word()`;
- `write_block()` / `read_block()`;
- page selection;
- Linear11 and Linear16 conversion.

`BoardController` gives those commands device meaning:

- identify device;
- select and verify page;
- read `STATUS_WORD` and `OPERATION`;
- read/write `VOUT_COMMAND`;
- read output voltage and current telemetry;
- read/write PID fields;
- read/write current-emulation inductance fields;
- read/write XDP AHB register fields.

Do not place Renesas- or MPS-specific register addresses in `PmbusDevice`. Put them in a new controller subclass or register-map object.

### VOUT command flow

1. Select the intended PMBus page.
2. Read `VOUT_MODE`.
3. Decode Linear16 exponent.
4. Quantize requested voltage to the nearest representable raw register value.
5. Write `VOUT_COMMAND`.
6. Set `OPERATION` mode only when required and verified for this controller.
7. Read back command and telemetry.
8. Abort before Bode/transient activity if the readback is outside tolerance.

`OPERATION = 0x80` sets bit 7 and requests output on under the PMBus-defined source selection. Other values such as `0xB0` include additional source/margin bits. Never generalize a vendor-observed value without checking that controller's PMBus documentation and `ON_OFF_CONFIG`.

### Raw XDP fields

The Infineon PID and inductance fields are not all standard PMBus commands. They are fields inside vendor memory/AHB registers. The implementation uses:

1. read the containing 32-bit word;
2. mask and replace only the target bit range;
3. write the complete word;
4. read it back;
5. decode the field and compare.

This read-modify-write rule is essential because one address may contain several unrelated controls.

### Reverse-engineering a vendor GUI safely

When documentation is incomplete, use controlled before/after snapshots:

1. Put hardware in a stable, low-risk state.
2. Read a bounded register set and save a timestamped snapshot.
3. Change exactly one vendor-GUI parameter by one known step.
4. Read the same register set again.
5. Diff address, word, and changed bits.
6. Repeat with a second value to infer scaling and signedness.
7. Restore the original value.
8. Validate readback through both the vendor GUI and this application.

Do not infer a writable command from a telemetry register. Do not write an unknown whole word when only one field changed.

## XDP USB Bridge Lifecycle

The XDP dongle is exclusive-access USB hardware. `XdpNodeUsbI2cAdapter` delegates transfers to a persistent Node process running `xdp_usb_bridge.js`.

Advantages of the persistent bridge:

- no Node startup for every PMBus transaction;
- stable USB initialization;
- much higher telemetry rate;
- one place to serialize transfers;
- controlled reconnect after board power cycling.

The common error is:

```text
LIBUSB_ERROR_ACCESS
```

Likely causes:

- XDP Designer owns the dongle;
- a previous bridge process is still alive;
- Self Testing returned without releasing its adapter;
- Windows has not completed USB re-enumeration;
- two backend requests opened separate bridges concurrently.

Recovery order:

1. Finish or cancel the current transfer.
2. Close the adapter that owns the bridge.
3. Stop the repository's persistent bridge.
4. Wait briefly for Windows to release the handle.
5. Reopen once and identify the dongle.
6. Only if necessary, stop known external bridge processes.
7. Never kill arbitrary Node processes.

XDP Designer may require physical unplug/replug after board power cycling because its process keeps stale device state. This backend can recover without unplugging because it tears down and recreates the bridge and USB handle.

## Self Testing Design

Self Testing answers: can the computer exchange commands with every device and restore state?

It must not answer: can the program energize the system?

Recommended tests:

- AFG: query identity, read phase, write a harmless phase change, verify, restore.
- Bode: confirm TCP listener, query identity, acquire/release lock, restore session.
- Supply: query identity and readbacks; only test a setpoint change when output is explicitly off; restore.
- Scope: query identity, switch waveform source to another enabled/valid channel, verify, restore.
- Board: identify, read page/status/telemetry, make only a verified reversible command change, restore.

Every result should report:

- resource present;
- identity;
- duration;
- actions performed;
- restored state;
- visible error and likely corrective action.

After the board test, close the controller and adapter and reset only the Self Test bridge ownership. Auto-Tune and Manual Tuning must be able to connect immediately afterward.

## Auto-Tune Hardware Order

One serial iteration should be easy to audit:

1. Prepare and verify Vout.
2. Write candidate controller parameters.
3. Read back candidate parameters.
4. Run enabled Bode analysis.
5. Configure the function generator from current GUI settings.
6. Enable the function-generator output.
7. Perform a single scope acquisition.
8. Disable the function-generator output in `finally`.
9. Analyze transient and Bode data.
10. Calculate penalty/pass state.
11. Persist record and artifacts.
12. Generate the next candidate.

Run every candidate end to end in hardware order: apply settings, capture the transient response, run the Bode sweep, calculate metrics, and store the result before starting the next candidate. This keeps hardware ownership, displayed history, pause behavior, and recovery aligned.

Pause/Stop must be checked between candidates. It should take effect after the current candidate finishes without interrupting a Bode sweep or scope capture midway.

## Data and Artifact Rules

Keep measurement truth separate from GUI display data:

- Full scope data: compact NPZ, one capture per file.
- Full Bode data: frequency, gain, phase, and metadata.
- Display data: bounded adaptive sample count.
- PNG: human-readable evidence, not the source of metrics.
- GIF: generated only by explicit user action.
- Run status/history: JSON with paths relative to the project where possible.

When moving a recent run to permanent storage, update every path reference. Do not keep a second recent copy or stale paths pointing to deleted files.

When loading old results, resolve legacy absolute paths, current relative paths, and moved permanent paths. If raw data exists but PNG is missing, rebuild the PNG from raw data instead of declaring the run unusable.

## Adding a New Controller Vendor

### Step 1: collect authoritative information

Obtain:

- PMBus/I2C address and page model;
- supported PMBus revision;
- `VOUT_MODE` format;
- telemetry commands and scaling;
- operation/output-control semantics;
- status and fault decoding;
- PID/compensation register map;
- nonvolatile-store behavior;
- write protection and unlock sequence;
- adapter requirements.

### Step 2: implement a controller class

Create a new subclass or implementation alongside `InfineonXdpController`, for example:

```python
class RenesasPowerController(BoardController):
    ...

class MpsPowerController(BoardController):
    ...
```

Add the kind to `create_board_controller()`. Keep common PMBus behavior in the base class and vendor-specific register access in the subclass.

### Step 3: define capability metadata

Not every controller has the same seven tuning fields. Expose capabilities such as:

- readable/writable Vout;
- PID fields and ranges;
- current-mode fields;
- inductance fields;
- output control;
- page/loop count;
- supported recovery action.

The frontend should render controls from capabilities instead of pretending unsupported fields exist.

### Step 4: add mock and conversion tests

Before real hardware:

- test command bytes;
- test endianness;
- test field masks;
- test signed values and exponents;
- test min/max quantization;
- test read-modify-write preservation;
- test failure cleanup.

### Step 5: stage real-hardware validation

1. Identify only.
2. Read status and telemetry.
3. Read all intended writable settings.
4. Change one low-risk setting and restore it.
5. Verify output-control behavior separately.
6. Run Manual Tuning with writes opt-in.
7. Run one transient-only iteration.
8. Run one Bode-only iteration.
9. Run one combined iteration.
10. Increase iteration count only after recovery is proven.

## Adding a New Scope, Supply, or AFG

For each replacement instrument:

1. Create a device-specific driver with the existing method contract.
2. Add identity-based resource discovery.
3. Add configuration fields only when the hardware supports them.
4. Add API request/response fields in backend and TypeScript types together.
5. Add a reversible Self Test.
6. Add stale-session retry and exact timeout handling.
7. Add one mock driver unit test for every command sequence.
8. Verify dark/light UI at desktop and narrow width.
9. Verify no control changes output state during page load.
10. Verify shutdown/exception cleanup with the real instrument.

## Debugging Workflow

### 1. Isolate the layer

Ask in order:

1. Does Windows see the device?
2. Does VISA/libusb see it?
3. Can the transport open it?
4. Does identity query work?
5. Does one direct driver read work?
6. Does the backend endpoint return valid JSON?
7. Does the frontend render that JSON?
8. Does the complete orchestration sequence work?

Do not debug React layout while the endpoint returns HTML or a transport exception.

### 2. Measure stages

Record separate durations for:

- connection/session acquisition;
- configuration;
- hardware acquisition;
- waveform transfer;
- numeric analysis;
- raw-data write;
- PNG generation;
- run-store persistence.

An iteration that grows from 7 seconds to 40 seconds usually indicates queued artifact work, repeated full-history rewriting, stale-session retries, or a growing payload. Total duration alone does not identify the cause.

### 3. Preserve the first useful error

The GUI should not replace an action error with an optional polling failure. Clear stale errors when a new action starts, debounce startup connection warnings, and log optional panel failures without using the global error banner.

### 4. Reproduce outside orchestration

Use a focused script or direct Python call to test one driver. Once reliable, test the HTTP endpoint. Only then run a full iteration.

## Failure Matrix

| Symptom | Likely cause | Correct response |
| --- | --- | --- |
| `LIBUSB_ERROR_ACCESS` | competing XDP owner/stale bridge | close owner, reset bridge, reopen once |
| Bode listener unreachable | ScpiRunner missing or serial unset | set serial/path, start runner, verify port 5025 |
| second Bode run fails | lock/session not released | unlock in `finally`, drop stale cache only |
| scope invalid session | stale VISA handle | evict cached scope, reconnect once |
| first scope read is slow | VISA/setup/acquisition warmup | warm session and measure stages |
| AFG `-420` | overlapping/unconsumed query | serialize command-response pairs |
| page loads HTML for API | old server/build or wrong route | rebuild frontend, restart backend, inspect endpoint |
| transient red startup banner | polling race during restart | require repeated status failures; clear on action |
| output trips at first iteration | unsafe default/write order or stale state | read baseline, quantize, verify before stimulus |
| Auto-Tune fails after Self Test | Self Test retained XDP ownership | close controller/adapter and reset bridge |
| plot missing for old run | stale/moved artifact path | resolve run-relative path or rebuild from raw data |
| later iterations slow down | queue/history rewrite/resource leak | bound workers, persist deltas, keep serial hardware |

## Definition of Done for a New Hardware Setup

A hardware migration is complete only when all items pass:

- [ ] Device discovery uses identity, not resource ordering.
- [ ] Driver has connect, read/write, timeout, and close behavior.
- [ ] Output state never changes on page load or Self Test.
- [ ] Self Test restores all changed settings.
- [ ] Self Test releases exclusive resources.
- [ ] Manual reads and writes use the new driver.
- [ ] Live readback does not write.
- [ ] Full Data Acquisition completes and cleans up outputs.
- [ ] One combined Auto-Tune iteration completes.
- [ ] Pause/Stop takes effect at the next safe boundary.
- [ ] Recoverable stale sessions retry once.
- [ ] Protection trips are recorded and recovered according to policy.
- [ ] Raw data and plots reload after server restart.
- [ ] Frontend types match backend JSON.
- [ ] Python compilation and unit tests pass.
- [ ] Frontend production build passes.
- [ ] Real-hardware command order is documented.

## Verification Commands

```powershell
python -m compileall hardware gui scripts
python -m unittest tests.test_tuning
Set-Location gui/frontend
npm run build
```

Start the GUI from the repository root:

```powershell
python gui/server.py --host 127.0.0.1 --port 8765
```

Stop only a real listener process, not PID 0:

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen |
  Where-Object { $_.OwningProcess -gt 0 } |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

## Final Engineering Principle

Treat every instrument integration as a state-management problem, not only a command-mapping problem. The difficult bugs occur at boundaries: two owners, stale handles, implicit output changes, missing restoration, mismatched units, and data files that move after capture. A thin vendor driver, explicit ownership, deterministic orchestration, and reversible tests make the same GUI and Auto-Tune workflow portable to a new laboratory bench.
