# TPU A1 Hardware-in-the-Loop Power Auto-Tuner

This repository contains a local hardware-in-the-loop workbench for tuning a
TPU power-board multiphase buck regulator. The main application is a localhost
web GUI named **Google Cloud Power Auto Tuner (V1.0)**. It can manually control the
lab instruments, run repeatable data acquisition, and perform real hardware PID
auto-tuning against Bode and transient-response targets.

The system is intended for bench use by people who do not want to edit Python
or SCPI scripts for every experiment. The GUI exposes the important knobs, while
the backend handles PMBus writes, instrument sequencing, safety checks, data
capture, plotting, result storage, and GIF generation.

## Safety

This code controls real power hardware. Before running any write operation:

- Confirm the power stage, load, probes, and injection setup are correct.
- Use the GUI self-test before starting tuning.
- Keep conservative search ranges until the board behavior is understood.
- Remember that only one program should own the XDP USB dongle at a time.
- The function generator output is disabled in `finally` blocks after
  transient captures, but the operator is still responsible for bench safety.

## Hardware Stack

The current bench integration covers:

- Infineon XDPE1A2G5C controller over PMBus/I2C through an Infineon XDP USB
  dongle.
- Tektronix AFG31000 function generator over VISA.
- Tektronix MSO58 oscilloscope over VISA using binary waveform transfer.
- OMICRON Bode 100 through Bode Analyzer Suite's SCPI runner.
- Keysight N5767A power supply over VISA. In the current bench this represents
  the master unit of a physically paralleled supply pair.

The board communication path is:

```text
Browser GUI
  -> gui/server.py
  -> BoardController / PmbusDevice
  -> XdpNodeUsbI2cAdapter
  -> hardware/instruments/xdp_usb_bridge.js
  -> Infineon XDP USB dongle
  -> PMBus/I2C
  -> XDPE1A2G5C controller
```

The instrument communication path is:

```text
Browser GUI
  -> gui/server.py
  -> hardware/instruments/*.py
  -> VISA / TCP SCPI
  -> AFG31000, MSO58, Bode 100, N5767A
```

## Repository Layout

```text
gui/
  server.py                 Local API server and static frontend server.
  frontend/                 React + Vite + TypeScript GUI.

hardware/instruments/
  board_controller.py       High-level XDPE board control.
  pmbus.py                  PMBus command encoding/decoding.
  i2c_adapters.py           XDP USB adapter process management.
  xdp_usb_bridge.js         Node USB bridge for the Infineon XDP dongle.
  oscilloscope.py           Tektronix MSO58 control and binary waveform reads.
  function_generator.py     Tektronix AFG31000 control.
  bode_analyzer.py          Bode 100 SCPI runner integration.
  bode100.py                Bode sweep helpers and metrics.
  power_supply.py           Keysight N5767A control.
  self_test.py              Reversible connection checks.

hardware/tuning/
  models.py                 Tuning dataclasses and result models.
  search.py                 Coarse/pairwise/refined candidate generation.
  analyzer.py               Transient, Bode, and penalty calculations.
  runner.py                 Real hardware tuning session orchestration.
  pid_programmer.py         PID programming interface.
  drl/                      Fixed-condition surrogate and Safe SAC workflow.

scripts/
  compact_scope_npz.py      Repack old scope captures into compact NPZ files.
  recompute_tuning_results.py
                            Recompute saved metrics/plots from stored data.

docs/
  hardware_in_loop_lab_notes.md
                            Lab notes and reverse-engineering history.
  i2c_pmbus_vout_control.md PMBus VOUT control notes.

results/                    Runtime tuning results. Ignored by git.
data/                       Runtime captures/cache. Ignored by git.
snapshots/                  Local reverse-engineering snapshots. Ignored by git.
```

## Setup

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Safe SAC is optional and uses a separate CPU-only dependency group. The normal
heuristic server does not require PyTorch:

```powershell
pip install -r requirements-ml.txt
```

Install frontend dependencies:

```powershell
cd gui/frontend
npm install
npm run build
cd ../..
```

The XDP bridge uses Node.js and USB access. The Bode 100 path expects Bode
Analyzer Suite to be installed and able to start or expose its SCPI runner.

## Environment

Copy `.env.example` to `.env` and fill in local values. `.env` is ignored by
git and must not be committed.

The GUI assistant uses an OpenAI-compatible chat-completions style API. The
default example is configured for MiniMax:

```env
LLM_API_BASE_URL=https://api.minimax.io/v1
LLM_API_KEY=your_minimax_api_key_here
LLM_MODEL=MiniMax-M3
LLM_TIMEOUT_S=30
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=700

MINIMAX_API_KEY=your_minimax_api_key_here
MINIMAX_BASE_URL=https://api.minimax.io/v1
MINIMAX_MODEL=MiniMax-M3
```

## Running The GUI

Start the localhost server:

```powershell
python gui/server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

If the port is already in use, stop the old server process or start on another
port.

## GUI Tabs

### PID Auto-Tune

Runs the real hardware tuning flow. The tab includes:

- Run control: start, single iteration, pause, resume, stop, reset, GIF save,
  and live iteration playback.
- Transient and Bode enable checkboxes. At least one analysis must be enabled.
- Function generator settings used by the tuning transient step.
- Bode analysis settings used by the tuning Bode sweep.
- Targets for transient and loop metrics.
- Search space for the hardware parameters.
- Result library for recent and permanent runs.
- Transient, Bode, penalty trend, iteration history, current result, and best
  result panels.
- Deep Reinforcement Learning control for the guarded 240-point collection,
  CPU model training, and four-episode (up to 60 point) hardware validation.

### Manual Tuning

Provides direct manual control of the bench:

- VOUT command and PMBus output control.
- PID/current-emulation register writes.
- Function generator setup and output enable/disable.
- Bode 100 sweep and plot.
- Scope capture with channel-axis assignment and binary waveform transfer.
- Power supply read/write and output enable/disable.
- Full Data Acquisition: Bode sweep, function generator step, scope capture,
  and function generator shutdown in one sequence.

### Self Testing

Runs reversible connection checks. These checks avoid turning outputs on or off
for safety. For devices where a setting check is needed, the test changes a
small reversible setting and restores it.

### AI Copilot

The assistant panel explains GUI concepts and workflows using the configured
LLM API. It is a helper only; it does not directly operate the hardware.

## Auto-Tune Hardware Flow

Each real tuning iteration uses the same backend path as Manual Tuning:

1. Write the target `VOUT_COMMAND`.
2. Write candidate PID/current-emulation parameters to the XDPE controller.
3. Apply the current function-generator settings.
4. Enable the function-generator output.
5. Capture the scope using single acquisition if transient analysis is enabled.
6. Disable the function-generator output in a `finally` block.
7. Only after the transient safety check passes, run the Bode 100 sweep if
   Bode analysis is enabled.
8. Compute metrics, update the penalty, store plots/data, and update best
   candidate.

If a transient trips protection or a candidate is invalid, the point is marked
invalid and bypassed. Invalid/tripped points use a bounded penalty instead of
stopping the whole run. The output-control recovery path disables both outputs,
restores the configured safe baseline, and re-enables the outputs. A DRL
workflow pauses after recovery and requires an explicit operator resume; it
never advances automatically.

## Fixed-Condition Safe SAC

The first DRL workflow is intentionally restricted to the current board,
approximately 0.93 V Vout, the 10 kHz / 0-1 V load step, and the current
1 kHz-1 MHz Bode setup. In the GUI, select **Deep Reinforcement Learning** and
use the controls in this order:

1. **Collect** creates and runs 60 repeats, 120 local Sobol candidates, and 60
   uncertainty candidates. Every candidate runs transient first.
2. **Train** fits five bootstrap surrogate MLPs and a 1,000,000-step SAC policy
   on CPU. Training runs in a background thread.
3. **Validate** freezes the model and policy, then runs four hardware episodes
   with at most 15 measurements each. A final candidate must pass three times
   consecutively.

Artifacts and resumable workflow state are stored under
`results/autotune_ml/`. A model is unavailable to normal **Start Auto-Tune**
until at least three of the four hardware episodes succeed. Configuration or
artifact hash mismatches fail closed.

## Search Strategy

The search operates on real hardware parameters:

- `mod0_kp`
- `mod0_ki`
- `mod0_kd`
- `mod0_kpole1`
- `mod0_kpole2`
- `mod0_cm_gain`
- output inductance
- effective LC inductance

The UI separates **Max Coarse Iterations** and **Max Refined Iterations**.
Coarse search samples combinations from the configured min/max/points table.
Pairwise exploration is part of the coarse stage. Refined search then performs
local refinement around the best candidates.

Current default PID ranges are intentionally broad:

- `mod0_kp`: 100 to 255
- `mod0_ki`: 150 to 255
- `mod0_kd`: 100 to 200
- `mod0_kpole1` and `mod0_kpole2`: usually tested as matched 3 or 6 values

## Metrics And Penalty

Lower penalty is better.

Transient metrics are computed from the scope capture:

- CH1 is the function-generator/load-step signal.
- CH3 is the output-voltage response.
- CH3 is low-pass filtered at 5 MHz for OS/US/settling calculations.
- Raw CH3 is still plotted for visibility, but the filtered trace is the
  metric trace.
- Rising and falling CH1 edges define the load segments.
- Low-load and high-load steady-state Vout are computed from the corresponding
  load windows.
- Overshoot and undershoot are measured relative to the load-state steady
  voltage.
- OS settling and US settling are reported separately in microseconds.

Current default transient targets:

- Overshoot max: 3 percent.
- Undershoot max: 3 percent.
- Settling target: 2 us.

Bode metrics are computed from the Bode 100 sweep:

- Phase margin target: greater than 45 deg.
- Crossover frequency upper limit: 200 kHz.
- Crossover below the upper limit can receive a small reward.
- Crossover above the upper limit receives a percent-based penalty.
- Gain margin is displayed, but it is not currently used in the penalty.
- A duplicated gain crossover is treated as an invalid Bode result.

Invalid/tripped candidates use a bounded penalty so they remain visible in the
history without dominating plots.

## Result Storage

Runtime results are stored under:

```text
results/autotune_runs/recent/
results/autotune_runs/saved/
results/autotune_ml/
```

Recent runs are temporary. Saving a run permanently moves it out of recent and
into saved results. Permanent runs are only removed by user action.

Each iteration may store:

- Scope full-data capture.
- Bode full-data capture.
- Scope PNG.
- Bode PNG.
- Combined GIF frames.
- Iteration metadata and metrics.

Scope full-data files use a compact NPZ format:

- One capture per file.
- `x` is stored as `x0`, `dx`, and `n` instead of a full array.
- Each channel `y` array is stored as `float32`.

This keeps complete data for future automation while reducing disk usage.

## Plotting And GIFs

Transient plots show CH1, raw CH3, and CH3 5 MHz LPF. Edge and settling markers
are drawn to make the settling-time decision visible.

Bode plots show gain and phase together. GIF generation supports:

- transient-only runs,
- Bode-only runs,
- combined transient + Bode + penalty-trend runs.

Trip/invalid points can be skipped in the GIF animation.

## Instrument Notes

### XDP / PMBus

The Infineon XDP dongle is accessed through a Node USB bridge. XDP Designer and
this backend should not own the dongle at the same time. If XDP Designer grabs
the dongle, close it and let the backend reconnect.

### Bode 100

Bode control uses Bode Analyzer Suite's SCPI runner over TCP. If the SCPI
listener is missing, open Bode Analyzer Suite once or verify the runner path and
device serial.

### MSO58 Scope

Scope waveform reads use binary transfer instead of ASCII. This is much faster
for large captures and allows the backend to retain complete waveform data.

### Power Supply

For the paralleled Keysight N5767A setup, control the master supply interface.
Do not reconfigure the physical parallel mode from this software.

## Useful Commands

Compile Python:

```powershell
python -m compileall hardware gui scripts
```

Run Python tests:

```powershell
python -m unittest tests.test_tuning
```

Build the frontend:

```powershell
cd gui/frontend
npm run build
```

Recompute saved tuning results after analyzer changes:

```powershell
python scripts/recompute_tuning_results.py --help
```

Compact older scope capture files:

```powershell
python scripts/compact_scope_npz.py --help
```

## Development Rules

- Keep secrets in `.env`; never commit them.
- Keep runtime data under ignored `results/`, `data/`, and `snapshots/`
  directories unless a specific artifact is intentionally exported.
- Prefer adding new hardware behavior behind backend helpers instead of wiring
  raw SCPI or PMBus commands directly into the frontend.
- Treat GUI defaults as bench defaults. If a default changes, update the
  frontend, backend model defaults, and this README together.
