import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactECharts from "echarts-for-react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Gauge,
  Pause,
  Play,
  RefreshCw,
  ShieldOff,
  SkipForward,
  SlidersHorizontal,
  StopCircle,
  LoaderCircle,
  XCircle,
  Zap
} from "lucide-react";
import { getTuningStatus, readVout, runSelfTestDevice, setVout, startTuning, stepTuning, stopTuning } from "./api";
import type { InstrumentKey, InstrumentTestResult, IterationRecord, SelfTestResponse, TuningConfig, TuningStatus, VoutReadback } from "./types";
import "./styles.css";

const defaultConfig: TuningConfig = {
  plant: {
    vdc: 12,
    inductance_h: 30e-6,
    capacitance_f: 15e-6,
    capacitor_esr_ohm: 7.5e-3,
    inductor_dcr_ohm: 50e-3
  },
  targets: {
    vout_target_v: 0.9,
    overshoot_pct: 4,
    undershoot_pct: 4,
    settling_time_s: 100e-6,
    oscillations: 0
  },
  search: {
    wc_min_rad_s: 94248,
    wc_max_rad_s: 314159,
    phi_min_deg: 30,
    phi_max_deg: 80,
    initial_wc_rad_s: 157080,
    initial_phi_deg: 60,
    max_iterations: 40
  }
};

const selfTestOrder: InstrumentKey[] = ["afg", "bode", "power_supply", "scope", "board_i2c"];

function App() {
  const [config, setConfig] = useState<TuningConfig>(defaultConfig);
  const [status, setStatus] = useState<TuningStatus | null>(null);
  const [activeTab, setActiveTab] = useState<"autotune" | "selftest">("autotune");
  const [error, setError] = useState("");
  const [vout, setVoutState] = useState<VoutReadback | null>(null);
  const [voutRequest, setVoutRequest] = useState({ address: "0x5E", page: 0, adapter: "xdp", voltage: 0.9 });
  const [selfTest, setSelfTest] = useState<SelfTestResponse | null>(null);
  const [selfTestRunning, setSelfTestRunning] = useState(false);
  const [activeSelfTestKey, setActiveSelfTestKey] = useState<InstrumentKey | null>(null);

  const refresh = async () => {
    try {
      const next = await getTuningStatus();
      setStatus(next);
      setConfig(next.config);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => window.clearInterval(timer);
  }, []);

  const runAction = async (action: "start" | "stop" | "step") => {
    try {
      const next =
        action === "start" ? await startTuning(config) : action === "stop" ? await stopTuning() : await stepTuning(config);
      setStatus(next);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const readBoardVout = async () => {
    try {
      setVoutState(await readVout(voutRequest.address, voutRequest.page, voutRequest.adapter));
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const writeBoardVout = async () => {
    try {
      setVoutState(await setVout(voutRequest.address, voutRequest.page, voutRequest.adapter, voutRequest.voltage));
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const testInstruments = async () => {
    const started = Date.now();
    try {
      setSelfTestRunning(true);
      setActiveSelfTestKey(null);
      setSelfTest({
        ok: true,
        timestamp: Date.now() / 1000,
        duration_s: 0,
        visa_resources: [],
        visa_resource_error: null,
        tests: [],
        all_passed: false
      });
      setError("");
      for (const key of selfTestOrder) {
        setActiveSelfTestKey(key);
        const result = await runSelfTestDevice(key);
        setSelfTest((current) => mergeSelfTestResult(current, result, started));
      }
    } catch (exc) {
      setError(String(exc));
    } finally {
      setActiveSelfTestKey(null);
      setSelfTestRunning(false);
    }
  };

  const testSingleInstrument = async (key: InstrumentKey) => {
    const started = Date.now();
    try {
      setSelfTestRunning(true);
      setActiveSelfTestKey(key);
      setSelfTest((current) => removeSelfTestResult(current, key));
      setError("");
      const result = await runSelfTestDevice(key);
      setSelfTest((current) => mergeSelfTestResult(current, result, started));
    } catch (exc) {
      setError(String(exc));
    } finally {
      setActiveSelfTestKey(null);
      setSelfTestRunning(false);
    }
  };

  const current = status?.current ?? null;
  const best = status?.best ?? null;
  const history = status?.history ?? [];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><Gauge size={24} /></div>
          <div>
            <h1>Buck PID Auto-Tuner</h1>
            <p>Hardware experiment workbench</p>
          </div>
        </div>
        <div className="status-strip">
          <StatusPill label="Backend" value={status?.state ?? "loading"} tone={status?.state === "error" ? "bad" : "good"} />
          <StatusPill label="PID programming" value="Stub / Disabled" tone="warn" />
          <StatusPill label="Iterations" value={String(history.length)} tone="neutral" />
        </div>
      </header>

      <nav className="tabs">
        <button className={activeTab === "autotune" ? "active" : ""} onClick={() => setActiveTab("autotune")}>
          <Gauge size={16} /> PID Auto-Tune
        </button>
        <button className={activeTab === "selftest" ? "active" : ""} onClick={() => setActiveTab("selftest")}>
          <CheckCircle2 size={16} /> Self Testing
        </button>
      </nav>

      {error && (
        <div className="banner error">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      <section className="notice">
        <ShieldOff size={18} />
        <span>No hardware PID writes are being sent. The current PID path is a stub until the XDP/I2C register map is verified.</span>
      </section>

      {activeTab === "autotune" ? <AutotuneWorkbench
        config={config}
        setConfig={setConfig}
        status={status}
        current={current}
        best={best}
        history={history}
        vout={vout}
        voutRequest={voutRequest}
        setVoutRequest={setVoutRequest}
        runAction={runAction}
        readBoardVout={readBoardVout}
        writeBoardVout={writeBoardVout}
      /> : <SelfTestingView
        result={selfTest}
        running={selfTestRunning}
        activeKey={activeSelfTestKey}
        onRun={testInstruments}
        onRunDevice={testSingleInstrument}
      />}
    </main>
  );
}

function AutotuneWorkbench({
  config,
  setConfig,
  status,
  current,
  best,
  history,
  vout,
  voutRequest,
  setVoutRequest,
  runAction,
  readBoardVout,
  writeBoardVout
}: {
  config: TuningConfig;
  setConfig: React.Dispatch<React.SetStateAction<TuningConfig>>;
  status: TuningStatus | null;
  current: IterationRecord | null;
  best: IterationRecord | null;
  history: IterationRecord[];
  vout: VoutReadback | null;
  voutRequest: { address: string; page: number; adapter: string; voltage: number };
  setVoutRequest: React.Dispatch<React.SetStateAction<{ address: string; page: number; adapter: string; voltage: number }>>;
  runAction: (action: "start" | "stop" | "step") => Promise<void>;
  readBoardVout: () => Promise<void>;
  writeBoardVout: () => Promise<void>;
}) {
  return (
      <div className="workspace">
        <aside className="control-rail">
          <Panel title="Run Control" icon={<Activity size={17} />}>
            <div className="button-row">
              <button className="primary" onClick={() => runAction("start")} disabled={status?.state === "running"}>
                <Play size={16} /> Start
              </button>
              <button onClick={() => runAction("step")} disabled={status?.state === "running"}>
                <SkipForward size={16} /> Step
              </button>
              <button onClick={() => runAction("stop")} disabled={status?.state !== "running"}>
                <StopCircle size={16} /> Stop
              </button>
            </div>
            <p className="message-line">{status?.message ?? "Connecting to local backend..."}</p>
          </Panel>

          <Panel title="Targets" icon={<SlidersHorizontal size={17} />}>
            <NumberField label="Vout target (V)" value={config.targets.vout_target_v} onChange={(value) => updateConfig(setConfig, "targets", "vout_target_v", value)} />
            <NumberField label="Overshoot max (%)" value={config.targets.overshoot_pct} onChange={(value) => updateConfig(setConfig, "targets", "overshoot_pct", value)} />
            <NumberField label="Undershoot max (%)" value={config.targets.undershoot_pct} onChange={(value) => updateConfig(setConfig, "targets", "undershoot_pct", value)} />
            <NumberField label="Settling target (us)" value={config.targets.settling_time_s * 1e6} onChange={(value) => updateConfig(setConfig, "targets", "settling_time_s", value / 1e6)} />
          </Panel>

          <Panel title="Search Space" icon={<RefreshCw size={17} />}>
            <NumberField label="wc min (rad/s)" value={config.search.wc_min_rad_s} onChange={(value) => updateConfig(setConfig, "search", "wc_min_rad_s", value)} />
            <NumberField label="wc max (rad/s)" value={config.search.wc_max_rad_s} onChange={(value) => updateConfig(setConfig, "search", "wc_max_rad_s", value)} />
            <NumberField label="initial phase margin (deg)" value={config.search.initial_phi_deg} onChange={(value) => updateConfig(setConfig, "search", "initial_phi_deg", value)} />
            <NumberField label="max iterations" value={config.search.max_iterations} onChange={(value) => updateConfig(setConfig, "search", "max_iterations", Math.max(1, Math.round(value)))} />
          </Panel>

          <Panel title="Vout Control" icon={<Zap size={17} />}>
            <TextField label="Address" value={voutRequest.address} onChange={(value) => setVoutRequest({ ...voutRequest, address: value })} />
            <NumberField label="Page" value={voutRequest.page} onChange={(value) => setVoutRequest({ ...voutRequest, page: Math.max(0, Math.round(value)) })} />
            <NumberField label="Set Vout (V)" value={voutRequest.voltage} onChange={(value) => setVoutRequest({ ...voutRequest, voltage: value })} />
            <div className="button-row">
              <button onClick={readBoardVout}><RefreshCw size={16} /> Read</button>
              <button onClick={writeBoardVout}><Zap size={16} /> Set</button>
            </div>
            <p className="readback">{formatVout(vout)}</p>
          </Panel>
        </aside>

        <section className="plot-deck">
          <Panel title="Transient Response" icon={<Activity size={17} />}>
            <ReactECharts option={waveformOption(current, config.targets.vout_target_v)} className="chart tall" />
          </Panel>
          <div className="split-grid">
            <Panel title="Score Trend" icon={<Pause size={17} />}>
              <ReactECharts option={scoreOption(history)} className="chart" />
            </Panel>
            <Panel title="Loop Gain Placeholder" icon={<Gauge size={17} />}>
              <ReactECharts option={bodePlaceholderOption(history)} className="chart" />
            </Panel>
          </div>
          <Panel title="Iteration History" icon={<RefreshCw size={17} />}>
            <IterationTable history={history} />
          </Panel>
        </section>

        <aside className="metrics-rail">
          <PidPanel title="Current Candidate" record={current} />
          <PidPanel title="Best Result" record={best} />
          <Panel title="Current Metrics" icon={<Activity size={17} />}>
            <Metrics record={current} />
          </Panel>
          <Panel title="PID Port" icon={<ShieldOff size={17} />}>
            <dl className="kv">
              <dt>Mode</dt>
              <dd>{status?.pid_programming.mode ?? "stub"}</dd>
              <dt>Available</dt>
              <dd>{status?.pid_programming.available ? "yes" : "no"}</dd>
              <dt>Writes</dt>
              <dd>{status?.pid_programming.write_attempts ?? 0}</dd>
            </dl>
          </Panel>
        </aside>
      </div>
  );
}

function SelfTestingView({
  result,
  running,
  activeKey,
  onRun,
  onRunDevice
}: {
  result: SelfTestResponse | null;
  running: boolean;
  activeKey: InstrumentKey | null;
  onRun: () => void;
  onRunDevice: (key: InstrumentKey) => void;
}) {
  const tests = result?.tests ?? [];
  return (
    <section className="selftest-page">
      <div className="selftest-header">
        <div>
          <h2>Self Testing</h2>
          <p>Reversible setting checks only. The test never turns instrument outputs on or off.</p>
        </div>
        <button className="primary" onClick={onRun} disabled={running}>
          <RefreshCw size={16} /> {running ? "Testing..." : "Run Connection Test"}
        </button>
      </div>

      <div className="instrument-grid">
        {selfTestOrder.map((key) => (
          <InstrumentCard
            key={key}
            deviceKey={key}
            result={tests.find((item) => item.key === key) ?? null}
            label={fallbackLabel(key)}
            testing={activeKey === key}
            running={running}
            onRun={onRunDevice}
          />
        ))}
      </div>

      <div className="split-grid">
        <Panel title="VISA Resources" icon={<Activity size={17} />}>
          {result?.visa_resource_error && <p className="message-line bad-text">{result.visa_resource_error}</p>}
          <div className="resource-list">
            {(result?.visa_resources ?? []).map((resource) => <code key={resource}>{resource}</code>)}
            {!result && <p className="muted">Run a connection test to list visible VISA resources.</p>}
          </div>
        </Panel>
        <Panel title="Summary" icon={<CheckCircle2 size={17} />}>
          <dl className="kv">
            <dt>Overall</dt>
            <dd>{running ? "running..." : result ? (result.all_passed ? "passed" : "needs attention") : "--"}</dd>
            <dt>Duration</dt>
            <dd>{result ? `${result.duration_s.toFixed(2)} s` : "--"}</dd>
            <dt>Passed</dt>
            <dd>{tests.filter((item) => item.status === "passed").length} / {selfTestOrder.length}</dd>
          </dl>
        </Panel>
      </div>
    </section>
  );
}

function InstrumentCard({
  deviceKey,
  result,
  label,
  testing,
  running,
  onRun
}: {
  deviceKey: InstrumentKey;
  result: InstrumentTestResult | null;
  label: string;
  testing: boolean;
  running: boolean;
  onRun: (key: InstrumentKey) => void;
}) {
  const passed = result?.status === "passed";
  return (
    <section className={`panel instrument-card ${testing ? "testing" : result ? (passed ? "passed" : "failed") : ""}`}>
      <div className="instrument-head">
        <div>
          <h2>{result?.label ?? label}</h2>
          <p>{result?.resource ?? "Default resource"}</p>
        </div>
        {testing ? <LoaderCircle className="spin" size={24} /> : result ? passed ? <CheckCircle2 size={24} /> : <XCircle size={24} /> : <RefreshCw size={22} />}
      </div>
      <dl className="kv">
        <dt>Status</dt>
        <dd>{testing ? "testing..." : result?.status ?? "not tested"}</dd>
        <dt>Resource present</dt>
        <dd>{result ? (result.resource_present ? "yes" : "no") : "--"}</dd>
        <dt>Duration</dt>
        <dd>{result ? `${result.duration_s.toFixed(2)} s` : "--"}</dd>
        <dt>Restored</dt>
        <dd>{result ? (result.restored ? "yes" : "no") : "--"}</dd>
      </dl>
      <p className="identity-line">{testing ? "Testing this instrument now..." : result?.identity || result?.error || "Run test to query identity."}</p>
      <div className="instrument-actions">
        <button onClick={() => onRun(deviceKey)} disabled={running}>
          <RefreshCw size={14} /> {testing ? "Testing..." : "Test"}
        </button>
      </div>
      {result && result.actions.length > 0 && (
        <div className="action-list">
          {result.actions.map((action, index) => (
            <div key={`${action.name}-${index}`}>
              <span>{action.name}</span>
              <strong>{action.value}</strong>
            </div>
          ))}
        </div>
      )}
      {result && Object.keys(result.details).length > 0 && (
        <dl className="kv details-kv">
          {Object.entries(result.details).map(([key, value]) => (
            <React.Fragment key={key}>
              <dt>{key}</dt>
              <dd>{value}</dd>
            </React.Fragment>
          ))}
        </dl>
      )}
    </section>
  );
}

function mergeSelfTestResult(current: SelfTestResponse | null, result: SelfTestResponse, started: number): SelfTestResponse {
  const previous = current ?? {
    ok: true,
    timestamp: Date.now() / 1000,
    duration_s: 0,
    visa_resources: [],
    visa_resource_error: null,
    tests: [],
    all_passed: false
  };
  const incoming = result.tests[0];
  const tests = incoming
    ? [...previous.tests.filter((item) => item.key !== incoming.key), incoming].sort(
        (a, b) => selfTestOrder.indexOf(a.key as InstrumentKey) - selfTestOrder.indexOf(b.key as InstrumentKey)
      )
    : previous.tests;
  return {
    ok: previous.ok && result.ok,
    timestamp: result.timestamp,
    duration_s: (Date.now() - started) / 1000,
    visa_resources: result.visa_resources.length ? result.visa_resources : previous.visa_resources,
    visa_resource_error: result.visa_resource_error ?? previous.visa_resource_error,
    tests,
    all_passed: tests.length === selfTestOrder.length && tests.every((item) => item.status === "passed")
  };
}

function removeSelfTestResult(current: SelfTestResponse | null, key: InstrumentKey): SelfTestResponse {
  const previous = current ?? {
    ok: true,
    timestamp: Date.now() / 1000,
    duration_s: 0,
    visa_resources: [],
    visa_resource_error: null,
    tests: [],
    all_passed: false
  };
  const tests = previous.tests.filter((item) => item.key !== key);
  return {
    ...previous,
    timestamp: Date.now() / 1000,
    tests,
    all_passed: tests.length === selfTestOrder.length && tests.every((item) => item.status === "passed")
  };
}

function fallbackLabel(key: string) {
  if (key === "afg") return "Tektronix AFG31000";
  if (key === "bode") return "OMICRON Bode 100";
  if (key === "power_supply") return "Keysight N5767A";
  if (key === "scope") return "Tektronix MSO58";
  return "Board I2C / XDPE1A2G5C";
}

function Panel({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-title">
        {icon}
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function StatusPill({ label, value, tone }: { label: string; value: string; tone: "good" | "warn" | "bad" | "neutral" }) {
  return (
    <div className={`status-pill ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function NumberField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" value={Number.isFinite(value) ? value : 0} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function TextField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function PidPanel({ title, record }: { title: string; record: IterationRecord | null }) {
  return (
    <Panel title={title} icon={<Gauge size={17} />}>
      <dl className="kv mono">
        <dt>Kp</dt>
        <dd>{formatSci(record?.pid.kp)}</dd>
        <dt>Ki</dt>
        <dd>{formatSci(record?.pid.ki)}</dd>
        <dt>Kd</dt>
        <dd>{formatSci(record?.pid.kd)}</dd>
        <dt>Kf</dt>
        <dd>{formatSci(record?.pid.kf)}</dd>
      </dl>
    </Panel>
  );
}

function Metrics({ record }: { record: IterationRecord | null }) {
  if (!record) return <p className="muted">No iteration has run yet.</p>;
  return (
    <dl className="kv">
      <dt>Overshoot</dt>
      <dd>{record.metrics.overshoot_pct.toFixed(2)}%</dd>
      <dt>Undershoot</dt>
      <dd>{record.metrics.undershoot_pct.toFixed(2)}%</dd>
      <dt>Settling</dt>
      <dd>{(record.metrics.settling_time_s * 1e6).toFixed(1)} us</dd>
      <dt>Oscillations</dt>
      <dd>{record.metrics.oscillations}</dd>
      <dt>Score</dt>
      <dd>{record.metrics.score.toFixed(3)}</dd>
      <dt>Pass</dt>
      <dd>{record.metrics.passed ? "yes" : "no"}</dd>
    </dl>
  );
}

function IterationTable({ history }: { history: IterationRecord[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Phase</th>
            <th>wc rad/s</th>
            <th>PM deg</th>
            <th>Kp</th>
            <th>Ki</th>
            <th>OS %</th>
            <th>US %</th>
            <th>Ts us</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {history.slice().reverse().map((item) => (
            <tr key={item.iteration} className={item.metrics.passed ? "passed" : ""}>
              <td>{item.iteration}</td>
              <td>{item.phase}</td>
              <td>{item.wc_rad_s.toFixed(0)}</td>
              <td>{item.phi_deg.toFixed(1)}</td>
              <td>{formatSci(item.pid.kp)}</td>
              <td>{formatSci(item.pid.ki)}</td>
              <td>{item.metrics.overshoot_pct.toFixed(2)}</td>
              <td>{item.metrics.undershoot_pct.toFixed(2)}</td>
              <td>{(item.metrics.settling_time_s * 1e6).toFixed(1)}</td>
              <td>{item.metrics.score.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function waveformOption(record: IterationRecord | null, target: number) {
  const time = record?.waveform.time_s.map((value) => value * 1e6) ?? [];
  const vout = record?.waveform.vout_v ?? [];
  return {
    animation: false,
    tooltip: { trigger: "axis" },
    grid: { left: 56, right: 20, top: 24, bottom: 42 },
    xAxis: { type: "category", name: "us", data: time.map((value) => value.toFixed(1)) },
    yAxis: { type: "value", name: "Vout" },
    series: [
      { name: "Vout", type: "line", smooth: true, symbol: "none", data: vout, lineStyle: { width: 2, color: "#1a73e8" } },
      { name: "Target", type: "line", symbol: "none", data: time.map(() => target), lineStyle: { width: 1, color: "#34a853", type: "dashed" } }
    ]
  };
}

function scoreOption(history: IterationRecord[]) {
  return {
    animation: false,
    tooltip: { trigger: "axis" },
    grid: { left: 48, right: 18, top: 20, bottom: 38 },
    xAxis: { type: "category", data: history.map((item) => item.iteration) },
    yAxis: { type: "value", name: "score" },
    series: [{ name: "Score", type: "line", smooth: true, data: history.map((item) => item.metrics.score), areaStyle: {}, lineStyle: { color: "#ea4335" } }]
  };
}

function bodePlaceholderOption(history: IterationRecord[]) {
  const last = history.at(-1);
  const points = [100, 300, 1000, 3000, 10000, 30000, 100000];
  return {
    animation: false,
    tooltip: { trigger: "axis" },
    grid: { left: 48, right: 18, top: 20, bottom: 38 },
    xAxis: { type: "category", name: "Hz", data: points },
    yAxis: { type: "value", name: "dB" },
    series: [
      {
        name: "Loop gain placeholder",
        type: "line",
        symbol: "none",
        data: points.map((_, index) => (last ? 24 - index * 7 + Math.log10(last.wc_rad_s / 100000) * 3 : null)),
        lineStyle: { color: "#fbbc04" }
      }
    ]
  };
}

function updateConfig<K extends keyof TuningConfig, F extends keyof TuningConfig[K]>(
  setConfig: React.Dispatch<React.SetStateAction<TuningConfig>>,
  group: K,
  field: F,
  value: TuningConfig[K][F]
) {
  setConfig((current) => ({
    ...current,
    [group]: {
      ...current[group],
      [field]: value
    }
  }));
}

function formatSci(value: number | undefined) {
  if (value === undefined || value === null) return "--";
  if (value === 0) return "0";
  return value.toExponential(3);
}

function formatVout(value: VoutReadback | null) {
  if (!value) return "No board readback yet.";
  if (!value.ok) return value.error ?? "Vout read failed.";
  return `${value.loop ?? "Loop"} command ${value.vout_command_v?.toFixed(6)} V, measured ${value.read_vout_v?.toFixed(6)} V`;
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
