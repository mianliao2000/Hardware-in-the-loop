import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactECharts from "echarts-for-react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Gauge,
  Pause,
  Play,
  Power as PowerIcon,
  RefreshCw,
  RotateCcw,
  ShieldOff,
  SkipForward,
  SlidersHorizontal,
  StopCircle,
  LoaderCircle,
  Moon,
  XCircle,
  Sun,
  Zap
} from "lucide-react";
import { captureScope, getTuningStatus, readFunctionGenerator, readInductance, readPmbusOutput, readPowerSupply, readVout, readXdpOutput, readXdpPid, runBodeSweep, runSelfTestDevice, setFunctionGenerator, setInductance, setPmbusOutput, setPowerSupply, setScopeAcquisition, setVout, setXdpOutput, setXdpPid, startTuning, stepTuning, stopTuning, warmScope } from "./api";
import type { BodeSweepConfig, BodeSweepReadback, FunctionGeneratorReadback, InductanceField, InductanceReadback, InstrumentKey, InstrumentTestResult, IterationRecord, PmbusOutputAction, PmbusOutputReadback, PowerSupplyReadback, ScopeCaptureReadback, SelfTestResponse, TuningConfig, TuningStatus, VoutReadback, XdpOutputAction, XdpOutputReadback, XdpPidReadback } from "./types";
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
    vout_target_v: 0.93,
    overshoot_pct: 4,
    undershoot_pct: 4,
    settling_time_s: 100e-6,
    oscillations: 0,
    phase_margin_deg: 60,
    crossover_frequency_hz: 100000
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
const xdpPidFieldNames = ["mod0_kp", "mod0_ki", "mod0_kd", "mod0_kpole1", "mod0_kpole2"] as const;
const defaultXdpPidLimits: Record<(typeof xdpPidFieldNames)[number], { min: number; max: number }> = {
  mod0_kp: { min: 0, max: 255 },
  mod0_ki: { min: 0, max: 255 },
  mod0_kd: { min: 0, max: 255 },
  mod0_kpole1: { min: 0, max: 15 },
  mod0_kpole2: { min: 0, max: 15 }
};
const liveTelemetryMinHz = 1;
const liveTelemetryDefaultHz = 10;
const liveTelemetryMaxHz = 1000;
const defaultTelemetryWindowSeconds = 15;
const telemetryHistoryWindowSeconds = 120;
const telemetryAxisStepMs = 5000;
const telemetryMovingAverageSeconds = 0.5;
const telemetryVoutAxisMax = 1.2;
const telemetryIoutAxisMax = 1000;
const defaultVoutExponent = -9;
const voutDisplayDigits = 4;
const defaultManualVoutRequest = { address: "0x5E", page: 0, adapter: "xdp", voltage: 0.9296875 };
const defaultManualInductanceRequest = { output_nh: 100.024, effective_lc_nh: 369.276 };
const outputInductanceRangeLabel = "14.29-117028.57 nH";
const effectiveLcInductanceRangeLabel = "229.02-117028.57 nH";
const outputInductanceLimitText = `Range: ${outputInductanceRangeLabel}`;
const effectiveLcInductanceLimitText = `Range: ${effectiveLcInductanceRangeLabel}`;
const defaultManualXdpPidRequest: Record<string, number> = {
  mod0_kp: 165,
  mod0_ki: 220,
  mod0_kd: 175,
  mod0_kpole1: 3,
  mod0_kpole2: 3
};
const defaultBodeSweepConfig: BodeSweepConfig = {
  host: "127.0.0.1",
  port: 5025,
  start_hz: 1000,
  stop_hz: 1_000_000,
  points: 201,
  bandwidth_hz: 300,
  source_dbm: 0,
  timeout_ms: 60000
};
const defaultScopeChannels = ["CH1", "CH3"];
const defaultScopeMeasurements: string[] = [];
const scopeFrontendMaxPoints = 200_000;
const defaultFunctionGeneratorConfig = {
  channel: 1,
  mode: "square",
  voltage_unit: "VPP",
  frequency_hz: 25000,
  low_v: 0,
  high_v: 1,
  pulse_width_s: 5e-6,
  dc_level_v: 0,
  amplitude_vpp: 1,
  offset_v: 0,
  phase_deg: 0
};
type AppTab = "autotune" | "manual" | "selftest";
type Language = "zh" | "en";
type ThemeMode = "light" | "dark";
type ManualWriteSelection = {
  vout: boolean;
  output_inductance: boolean;
  effective_lc_inductance: boolean;
  xdp_pid: boolean;
};
type ManualWriteSnapshot = {
  address: string;
  page: number;
  adapter: string;
  voltage: number;
  output_nh: number;
  effective_lc_nh: number;
  pid: Record<string, number>;
  selection: ManualWriteSelection;
};
type ChartAxisSettings = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  y2Min: number;
  y2Max: number;
};
type TelemetryAxisSettings = {
  durationSeconds: number;
  yMin: number;
  yMax: number;
  y2Min: number;
  y2Max: number;
};
type ScopeAxisSide = "left" | "right";
type ScopeAxisSettings = {
  leftMin: number;
  leftMax: number;
  rightMin: number;
  rightMax: number;
  channelAxes: Record<string, ScopeAxisSide>;
};
const tabStorageKey = "tpu_a1_active_tab";
const languageStorageKey = "tpu_a1_language";
const themeStorageKey = "tpu_a1_theme";
const appTabs: AppTab[] = ["autotune", "manual", "selftest"];
const defaultTelemetryAxisSettings: TelemetryAxisSettings = {
  durationSeconds: defaultTelemetryWindowSeconds,
  yMin: 0,
  yMax: telemetryVoutAxisMax,
  y2Min: 0,
  y2Max: telemetryIoutAxisMax
};
const defaultBodeAxisSettings: ChartAxisSettings = {
  xMin: defaultBodeSweepConfig.start_hz,
  xMax: defaultBodeSweepConfig.stop_hz,
  yMin: -100,
  yMax: 100,
  y2Min: -200,
  y2Max: 200
};
const defaultScopeAxisSettings: ScopeAxisSettings = {
  leftMin: -0.5,
  leftMax: 3,
  rightMin: 0.7,
  rightMax: 1.1,
  channelAxes: {
    CH1: "left",
    CH2: "left",
    CH3: "right",
    CH4: "right",
    CH5: "left",
    CH6: "left",
    CH7: "right",
    CH8: "right"
  }
};

function getInitialTab(): AppTab {
  const params = new URLSearchParams(window.location.search);
  const queryTab = params.get("tab");
  if (queryTab && appTabs.includes(queryTab as AppTab)) return queryTab as AppTab;
  const storedTab = window.localStorage.getItem(tabStorageKey);
  if (storedTab && appTabs.includes(storedTab as AppTab)) return storedTab as AppTab;
  return "autotune";
}

function getInitialLanguage(): Language {
  const stored = window.localStorage.getItem(languageStorageKey);
  return stored === "zh" || stored === "en" ? stored : "en";
}

function getInitialTheme(): ThemeMode {
  const stored = window.localStorage.getItem(themeStorageKey);
  return stored === "light" || stored === "dark" ? stored : "light";
}

const copy = {
  en: {
    platform: "AI Powered Hardware-in-the-loop Platform",
    copyright: "Copyright © Google LLC",
    author: "Author: Jackson (Mian) Liao",
    backend: "Backend",
    pidProgramming: "PID programming",
    iterations: "Iterations",
    autoTune: "PID Auto-Tune",
    manualTuning: "Manual Tuning",
    selfTesting: "Self Testing",
    pidNotice: "No hardware PID writes are being sent. The current PID path is a stub until the XDP/I2C register map is verified.",
    xdpConnection: "XDP Connection",
    edit: "Edit",
    hide: "Hide",
    liveReadback: "Real Time Feedback",
    readAll: "Read Outputs/Inductance/PID",
    manualParameters: "Manual Parameters",
    writeVout: "Write VOUT_COMMAND",
    voutTarget: "Vout target (V)",
    writeOutputInductance: "Write Output Inductance",
    outputInductance: "Output Inductance (nH)",
    writeEffectiveLc: "Write Effective Lc Inductance",
    effectiveLc: "Effective Lc Inductance (nH)",
    writePid: "Write mod0 PID register",
    writing: "Writing...",
    writeSelected: "Write Selected to XDP",
    writeNote: "Only VOUT_COMMAND is selected by default. PID and inductance writes are raw register writes and remain opt-in.",
    liveData: "Live Data"
  },
  zh: {
    platform: "AI 驱动的硬件在环平台",
    copyright: "版权 © Google LLC",
    author: "Author: Jackson (Mian) Liao",
    backend: "后端",
    pidProgramming: "PID 写入",
    iterations: "迭代次数",
    autoTune: "PID 自动调参",
    manualTuning: "手动调参",
    selfTesting: "自检",
    pidNotice: "当前不会写入真实硬件 PID。PID 通路会保持 stub，直到 XDP/I2C register map 完成验证。",
    xdpConnection: "XDP 连接",
    edit: "编辑",
    hide: "收起",
    liveReadback: "Real Time Feedback",
    readAll: "读取 Outputs/Inductance/PID",
    manualParameters: "手动参数",
    writeVout: "写入 VOUT_COMMAND",
    voutTarget: "Vout 目标值 (V)",
    writeOutputInductance: "写入 Output Inductance",
    outputInductance: "Output Inductance (nH)",
    writeEffectiveLc: "写入 Effective Lc Inductance",
    effectiveLc: "Effective Lc Inductance (nH)",
    writePid: "写入 mod0 PID register",
    writing: "写入中...",
    writeSelected: "写入选中参数到 XDP",
    writeNote: "默认只选中 VOUT_COMMAND。PID 和电感写入是 raw register 写入，需要手动勾选。",
    liveData: "实时数据"
  }
} satisfies Record<Language, Record<string, string>>;

function appendTelemetrySample(current: VoutReadback[], next: VoutReadback) {
  const timestamp = next.timestamp ?? Date.now() / 1000;
  const sample = { ...next, timestamp };
  const cutoff = timestamp - telemetryHistoryWindowSeconds;
  return [...current, sample].filter((item) => (item.timestamp ?? timestamp) >= cutoff);
}

function smoothTelemetryHistory(history: VoutReadback[], windowSeconds = telemetryMovingAverageSeconds) {
  if (history.length === 0) return [];
  return history.map((sample, index) => {
    const timestamp = sample.timestamp ?? Date.now() / 1000;
    const start = timestamp - windowSeconds;
    const window = history
      .slice(0, index + 1)
      .filter((item) => {
        const itemTimestamp = item.timestamp ?? timestamp;
        return itemTimestamp >= start && itemTimestamp <= timestamp;
      });
    return {
      ...sample,
      read_vout_v: averageDefined(window.map((item) => item.read_vout_v)),
      read_iout_a: averageDefined(window.map((item) => item.read_iout_a)),
      vout_command_v: averageDefined(window.map((item) => item.vout_command_v))
    };
  });
}

function averageDefined(values: Array<number | undefined>) {
  const valid = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (valid.length === 0) return undefined;
  return valid.reduce((total, value) => total + value, 0) / valid.length;
}

function clampTelemetryDuration(value: number) {
  if (!Number.isFinite(value)) return defaultTelemetryWindowSeconds;
  return Math.max(1, Math.min(telemetryHistoryWindowSeconds, value));
}

function clampTelemetryRate(value: number) {
  if (!Number.isFinite(value)) return liveTelemetryDefaultHz;
  return Math.max(liveTelemetryMinHz, Math.min(liveTelemetryMaxHz, value));
}

function parseFrequencyText(value: string) {
  const trimmed = value.trim();
  const match = trimmed.match(/^([+-]?\d+(?:\.\d+)?|\.\d+)\s*([kKmM]?)\s*(?:hz)?$/i);
  if (!match) return Number.NaN;
  const base = Number(match[1]);
  const suffix = match[2].toLowerCase();
  const scale = suffix === "k" ? 1_000 : suffix === "m" ? 1_000_000 : 1;
  return base * scale;
}

function formatFrequencyInput(value: number) {
  if (!Number.isFinite(value)) return "1k";
  if (value >= 1_000_000 && Math.abs(value % 1_000_000) < 1e-9) return `${trimTick(value / 1_000_000)}M`;
  if (value >= 1_000 && Math.abs(value % 1_000) < 1e-9) return `${trimTick(value / 1_000)}k`;
  return trimTick(value);
}

function voutExponentFromReadback(vout: VoutReadback | null) {
  return Number.isFinite(vout?.exponent) ? Number(vout?.exponent) : defaultVoutExponent;
}

function snapVoutToRegister(voltage: number, exponent: number) {
  const lsb = 2 ** exponent;
  if (!Number.isFinite(voltage) || !Number.isFinite(lsb) || lsb <= 0) return 0;
  const raw = Math.max(0, Math.min(0xffff, Math.round(voltage / lsb)));
  return Number((raw * lsb).toFixed(voutDisplayDigits));
}

function snapVoutToRegisterAbove(voltage: number, exponent: number) {
  const lsb = 2 ** exponent;
  if (!Number.isFinite(voltage) || !Number.isFinite(lsb) || lsb <= 0) return 0;
  const raw = Math.max(0, Math.min(0xffff, Math.floor(voltage / lsb) + 1));
  return Number((raw * lsb).toFixed(voutDisplayDigits));
}

function voutRegisterStepFromExponent(exponent: number) {
  const lsb = 2 ** exponent;
  return Number.isFinite(lsb) && lsb > 0 ? lsb : 2 ** defaultVoutExponent;
}

function xdpPidLimitLabel(name: (typeof xdpPidFieldNames)[number], xdpPid: XdpPidReadback | null) {
  const field = xdpPid?.pid_registers?.fields[name];
  const limits = field ? { min: field.min, max: field.max } : defaultXdpPidLimits[name];
  return `${name} (${limits.min}-${limits.max})`;
}

function App() {
  const [config, setConfig] = useState<TuningConfig>(defaultConfig);
  const [status, setStatus] = useState<TuningStatus | null>(null);
  const [activeTab, setActiveTab] = useState<AppTab>(getInitialTab);
  const [language, setLanguage] = useState<Language>(getInitialLanguage);
  const [themeMode, setThemeMode] = useState<ThemeMode>(getInitialTheme);
  const [error, setError] = useState("");
  const [vout, setVoutState] = useState<VoutReadback | null>(null);
  const [voutRequest, setVoutRequest] = useState({ ...defaultManualVoutRequest });
  const [inductance, setInductanceState] = useState<InductanceReadback | null>(null);
  const [inductanceRequest, setInductanceRequest] = useState({ ...defaultManualInductanceRequest });
  const [xdpPid, setXdpPidState] = useState<XdpPidReadback | null>(null);
  const [xdpPidRequest, setXdpPidRequest] = useState<Record<string, number>>({ ...defaultManualXdpPidRequest });
  const [pmbusOutput, setPmbusOutputState] = useState<PmbusOutputReadback | null>(null);
  const [pmbusOutputRunning, setPmbusOutputRunning] = useState(false);
  const [xdpOutput, setXdpOutputState] = useState<XdpOutputReadback | null>(null);
  const [xdpOutputRunning, setXdpOutputRunning] = useState(false);
  const [bodeSweepConfig, setBodeSweepConfig] = useState<BodeSweepConfig>(defaultBodeSweepConfig);
  const [bodeSweep, setBodeSweep] = useState<BodeSweepReadback | null>(null);
  const [bodeSweepRunning, setBodeSweepRunning] = useState(false);
  const [bodeSweepStatus, setBodeSweepStatus] = useState("Ready");
  const [telemetryHistory, setTelemetryHistory] = useState<VoutReadback[]>([]);
  const [manualLiveRefresh, setManualLiveRefresh] = useState(false);
  const [manualLiveRateHz, setManualLiveRateHz] = useState(liveTelemetryDefaultHz);
  const [manualWriteSelection, setManualWriteSelection] = useState<ManualWriteSelection>({
    vout: true,
    output_inductance: false,
    effective_lc_inductance: false,
    xdp_pid: false
  });
  const [manualRealTimeWriting, setManualRealTimeWriting] = useState(false);
  const [manualWriteRunning, setManualWriteRunning] = useState(false);
  const realTimeWriteTimer = useRef<number | null>(null);
  const realTimeWriteInFlight = useRef(false);
  const pendingRealTimeWrite = useRef<ManualWriteSnapshot | null>(null);
  const lastRealTimeWriteSignature = useRef("");
  const [selfTest, setSelfTest] = useState<SelfTestResponse | null>(null);
  const [selfTestRunning, setSelfTestRunning] = useState(false);
  const [activeSelfTestKey, setActiveSelfTestKey] = useState<InstrumentKey | null>(null);
  const t = copy[language];

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

  useEffect(() => {
    window.localStorage.setItem(tabStorageKey, activeTab);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", activeTab);
    window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
  }, [activeTab]);

  useEffect(() => {
    window.localStorage.setItem(languageStorageKey, language);
  }, [language]);

  useEffect(() => {
    window.localStorage.setItem(themeStorageKey, themeMode);
    document.body.classList.toggle("theme-dark", themeMode === "dark");
  }, [themeMode]);

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
      const next = await readVout(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setVoutState(next);
      if (next.ok && next.read_vout_v !== undefined) {
        setTelemetryHistory((current) => appendTelemetrySample(current, next));
      }
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const writeBoardVout = async () => {
    try {
      const requestedVoltage = snapVoutToRegister(voutRequest.voltage, voutExponentFromReadback(vout));
      const next = await setVout(voutRequest.address, voutRequest.page, voutRequest.adapter, requestedVoltage);
      setVoutState(next);
      setVoutRequest((current) => ({ ...current, voltage: requestedVoltage }));
      if (next.ok && next.read_vout_v !== undefined) {
        setTelemetryHistory((current) => appendTelemetrySample(current, next));
      }
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const readBoardInductance = async () => {
    try {
      const next = await readInductance(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setInductanceState(next);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const readBoardXdpPid = async () => {
    try {
      const next = await readXdpPid(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setXdpPidState(next);
      const fields = next.pid_registers?.fields;
      if (fields) {
        setXdpPidRequest((current) => ({
          ...current,
          ...Object.fromEntries(xdpPidFieldNames.map((name) => [name, fields[name]?.raw ?? current[name]]))
        }));
      }
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const readBoardPmbusOutput = async () => {
    try {
      const next = await readPmbusOutput(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setPmbusOutputState(next);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const readBoardXdpOutput = async () => {
    try {
      const next = await readXdpOutput(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setXdpOutputState(next);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const resetManualDefaults = () => {
    setVoutRequest((current) => ({ ...current, voltage: defaultManualVoutRequest.voltage }));
    setInductanceRequest({ ...defaultManualInductanceRequest });
    setXdpPidRequest({ ...defaultManualXdpPidRequest });
    setError("");
  };

  const writeBoardPmbusOutput = async (action: PmbusOutputAction) => {
    try {
      setPmbusOutputRunning(true);
      const next = await setPmbusOutput(voutRequest.address, voutRequest.page, voutRequest.adapter, action);
      setPmbusOutputState(next);
      const refreshed = await readVout(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setVoutState(refreshed);
      if (refreshed.ok && refreshed.read_vout_v !== undefined) {
        setTelemetryHistory((current) => appendTelemetrySample(current, refreshed));
      }
      setError("");
    } catch (exc) {
      setError(String(exc));
    } finally {
      setPmbusOutputRunning(false);
    }
  };

  const writeBoardXdpOutput = async (action: XdpOutputAction) => {
    try {
      setXdpOutputRunning(true);
      const next = await setXdpOutput(voutRequest.address, voutRequest.page, voutRequest.adapter, action);
      setXdpOutputState(next);
      const refreshed = await readVout(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setVoutState(refreshed);
      if (refreshed.ok && refreshed.read_vout_v !== undefined) {
        setTelemetryHistory((current) => appendTelemetrySample(current, refreshed));
      }
      setError("");
    } catch (exc) {
      setError(String(exc));
    } finally {
      setXdpOutputRunning(false);
    }
  };

  useEffect(() => {
    if (activeTab !== "manual" || !manualLiveRefresh) return;
    let cancelled = false;

    const wait = (durationMs: number) => new Promise((resolve) => window.setTimeout(resolve, durationMs));

    const telemetryLoop = async () => {
      while (!cancelled) {
        const started = performance.now();
        await readBoardVout();
        const elapsed = performance.now() - started;
        await wait(Math.max(0, 1000 / clampTelemetryRate(manualLiveRateHz) - elapsed));
      }
    };

    telemetryLoop();
    return () => {
      cancelled = true;
    };
  }, [activeTab, manualLiveRefresh, manualLiveRateHz, voutRequest.address, voutRequest.page, voutRequest.adapter]);

  useEffect(() => {
    if (activeTab !== "manual") return;
    readBoardVout();
    readBoardInductance();
    readBoardXdpPid();
    readBoardPmbusOutput();
    readBoardXdpOutput();
  }, [activeTab, voutRequest.address, voutRequest.page, voutRequest.adapter]);

  const writeOutputInductance = async () => {
    try {
      const next = await setInductance(voutRequest.address, voutRequest.page, voutRequest.adapter, {
        output_inductance_nh: inductanceRequest.output_nh
      });
      setInductanceState(next);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const writeEffectiveLcInductance = async () => {
    try {
      const next = await setInductance(voutRequest.address, voutRequest.page, voutRequest.adapter, {
        effective_lc_inductance_nh: inductanceRequest.effective_lc_nh
      });
      setInductanceState(next);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const buildManualWriteSnapshot = (overrides: Partial<ManualWriteSnapshot> = {}): ManualWriteSnapshot => {
    const base: ManualWriteSnapshot = {
      address: voutRequest.address,
      page: voutRequest.page,
      adapter: voutRequest.adapter,
      voltage: snapVoutToRegister(voutRequest.voltage, voutExponentFromReadback(vout)),
      output_nh: inductanceRequest.output_nh,
      effective_lc_nh: inductanceRequest.effective_lc_nh,
      pid: { ...xdpPidRequest },
      selection: { ...manualWriteSelection }
    };
    return {
      ...base,
      ...overrides,
      pid: overrides.pid ? { ...overrides.pid } : base.pid,
      selection: overrides.selection ? { ...overrides.selection } : base.selection
    };
  };

  const writeManualSnapshot = async (snapshot: ManualWriteSnapshot, showSelectionError = true) => {
    const inductancePayload: { output_inductance_nh?: number; effective_lc_inductance_nh?: number } = {};
    if (snapshot.selection.output_inductance) {
      inductancePayload.output_inductance_nh = snapshot.output_nh;
    }
    if (snapshot.selection.effective_lc_inductance) {
      inductancePayload.effective_lc_inductance_nh = snapshot.effective_lc_nh;
    }
    const xdpPidPayload = snapshot.selection.xdp_pid
      ? Object.fromEntries(xdpPidFieldNames.map((name) => [name, snapshot.pid[name]]))
      : {};
    if (!snapshot.selection.vout && Object.keys(inductancePayload).length === 0 && Object.keys(xdpPidPayload).length === 0) {
      if (showSelectionError) setError("Select at least one parameter to write.");
      return;
    }
    try {
      setManualWriteRunning(true);
      if (snapshot.selection.vout) {
        const nextVout = await setVout(snapshot.address, snapshot.page, snapshot.adapter, snapshot.voltage);
        setVoutState(nextVout);
        setVoutRequest((current) => ({ ...current, voltage: snapshot.voltage }));
        if (nextVout.ok && nextVout.read_vout_v !== undefined) {
          setTelemetryHistory((current) => appendTelemetrySample(current, nextVout));
        }
      }
      if (Object.keys(inductancePayload).length > 0) {
        const nextInductance = await setInductance(
          snapshot.address,
          snapshot.page,
          snapshot.adapter,
          inductancePayload
        );
        setInductanceState(nextInductance);
      }
      if (Object.keys(xdpPidPayload).length > 0) {
        const nextPid = await setXdpPid(snapshot.address, snapshot.page, snapshot.adapter, xdpPidPayload);
        setXdpPidState(nextPid);
      }
      const refreshed = await readVout(snapshot.address, snapshot.page, snapshot.adapter);
      setVoutState(refreshed);
      if (refreshed.ok && refreshed.read_vout_v !== undefined) {
        setTelemetryHistory((current) => appendTelemetrySample(current, refreshed));
      }
      setError("");
    } catch (exc) {
      setError(String(exc));
    } finally {
      setManualWriteRunning(false);
    }
  };

  const writeManualTuning = async () => {
    await writeManualSnapshot(buildManualWriteSnapshot());
  };

  const writeAllManualParametersForAcquisition = async () => {
    await writeManualSnapshot(
      buildManualWriteSnapshot({
        selection: {
          vout: true,
          output_inductance: true,
          effective_lc_inductance: true,
          xdp_pid: true
        }
      }),
      false
    );
  };

  const handleRealTimeWritingChange = (enabled: boolean) => {
    setManualRealTimeWriting(enabled);
    if (!enabled || activeTab !== "manual") return;
    const voltage = snapVoutToRegister(voutRequest.voltage, voutExponentFromReadback(vout));
    const selection = { ...manualWriteSelection, vout: true };
    setManualWriteSelection(selection);
    setVoutRequest((current) => ({ ...current, voltage }));
    pendingRealTimeWrite.current = buildManualWriteSnapshot({ voltage, selection });
    if (realTimeWriteTimer.current !== null) {
      window.clearTimeout(realTimeWriteTimer.current);
    }
    realTimeWriteTimer.current = window.setTimeout(flushRealTimeWrite, 0);
  };

  const flushRealTimeWrite = async () => {
    if (realTimeWriteInFlight.current) {
      return;
    }
    const snapshot = pendingRealTimeWrite.current;
    if (!snapshot) {
      return;
    }
    const signature = JSON.stringify(snapshot);
    pendingRealTimeWrite.current = null;
    if (signature === lastRealTimeWriteSignature.current) {
      return;
    }
    realTimeWriteInFlight.current = true;
    lastRealTimeWriteSignature.current = signature;
    try {
      await writeManualSnapshot(snapshot, false);
    } finally {
      realTimeWriteInFlight.current = false;
      if (pendingRealTimeWrite.current) {
        window.setTimeout(flushRealTimeWrite, 0);
      }
    }
  };

  const queueRealTimeWrite = (overrides: Partial<ManualWriteSnapshot>) => {
    if (activeTab !== "manual" || !manualRealTimeWriting) return;
    pendingRealTimeWrite.current = buildManualWriteSnapshot(overrides);
    if (realTimeWriteTimer.current !== null) {
      window.clearTimeout(realTimeWriteTimer.current);
    }
    realTimeWriteTimer.current = window.setTimeout(flushRealTimeWrite, 80);
  };

  useEffect(() => () => {
    if (realTimeWriteTimer.current !== null) {
      window.clearTimeout(realTimeWriteTimer.current);
    }
  }, []);

  const runManualBodeSweep = async () => {
    try {
      setBodeSweepRunning(true);
      setBodeSweepStatus("Running Bode 100 sweep...");
      setError("");
      const next = await runBodeSweep(bodeSweepConfig);
      setBodeSweep(next);
      if (next.ok === false) {
        setBodeSweepStatus(next.error ?? "Bode sweep failed");
      } else {
        setBodeSweepStatus(next.duration_s ? `Done in ${next.duration_s.toFixed(2)} s` : "Sweep complete");
      }
      return next;
    } catch (exc) {
      const message = String(exc);
      setError(message);
      setBodeSweepStatus(message);
      return null;
    } finally {
      setBodeSweepRunning(false);
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
          <GoogleMark />
          <div>
            <h1 className="google-title">
              <span aria-label="Google">
                <span className="google-blue">G</span>
                <span className="google-red">o</span>
                <span className="google-yellow">o</span>
                <span className="google-blue">g</span>
                <span className="google-green">l</span>
                <span className="google-red">e</span>
              </span>{" "}
              <span className="product-title">Power Auto-Tuner (V1.0)</span>
            </h1>
            <p>{t.platform} ({t.copyright})</p>
            <p>{t.author}</p>
          </div>
        </div>
        <div className="topbar-actions">
          <HeaderToggles
            language={language}
            setLanguage={setLanguage}
            themeMode={themeMode}
            setThemeMode={setThemeMode}
          />
        </div>
      </header>

      <nav className="tabs">
        <button className={activeTab === "autotune" ? "active" : ""} onClick={() => setActiveTab("autotune")}>
          <Gauge size={16} /> {t.autoTune}
        </button>
        <button className={activeTab === "manual" ? "active" : ""} onClick={() => setActiveTab("manual")}>
          <SlidersHorizontal size={16} /> {t.manualTuning}
        </button>
        <button className={activeTab === "selftest" ? "active" : ""} onClick={() => setActiveTab("selftest")}>
          <CheckCircle2 size={16} /> {t.selfTesting}
        </button>
      </nav>

      {error && (
        <div className="banner error">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      {activeTab === "autotune" ? <AutotuneWorkbench
        config={config}
        setConfig={setConfig}
        status={status}
        current={current}
        best={best}
        history={history}
        vout={vout}
        inductance={inductance}
        inductanceRequest={inductanceRequest}
        setInductanceRequest={setInductanceRequest}
        xdpPid={xdpPid}
        xdpPidRequest={xdpPidRequest}
        setXdpPidRequest={setXdpPidRequest}
        runAction={runAction}
        readBoardInductance={readBoardInductance}
        writeOutputInductance={writeOutputInductance}
        writeEffectiveLcInductance={writeEffectiveLcInductance}
      /> : activeTab === "manual" ? <ManualTuningView
        vout={vout}
        voutRequest={voutRequest}
        setVoutRequest={setVoutRequest}
        inductance={inductance}
        inductanceRequest={inductanceRequest}
        setInductanceRequest={setInductanceRequest}
        xdpPid={xdpPid}
        xdpPidRequest={xdpPidRequest}
        setXdpPidRequest={setXdpPidRequest}
        pmbusOutput={pmbusOutput}
        pmbusOutputRunning={pmbusOutputRunning}
        readBoardPmbusOutput={readBoardPmbusOutput}
        writeBoardPmbusOutput={writeBoardPmbusOutput}
        xdpOutput={xdpOutput}
        xdpOutputRunning={xdpOutputRunning}
        readBoardXdpOutput={readBoardXdpOutput}
        writeBoardXdpOutput={writeBoardXdpOutput}
        bodeSweep={bodeSweep}
        bodeSweepConfig={bodeSweepConfig}
        setBodeSweepConfig={setBodeSweepConfig}
        setBodeSweep={setBodeSweep}
        bodeSweepRunning={bodeSweepRunning}
        setBodeSweepRunning={setBodeSweepRunning}
        bodeSweepStatus={bodeSweepStatus}
        setBodeSweepStatus={setBodeSweepStatus}
        runManualBodeSweep={runManualBodeSweep}
        setError={setError}
        telemetryHistory={telemetryHistory}
        liveRefresh={manualLiveRefresh}
        setLiveRefresh={setManualLiveRefresh}
        liveRateHz={manualLiveRateHz}
        setLiveRateHz={setManualLiveRateHz}
        realTimeWriting={manualRealTimeWriting}
        setRealTimeWriting={handleRealTimeWritingChange}
        queueRealTimeWrite={queueRealTimeWrite}
        writeSelection={manualWriteSelection}
        setWriteSelection={setManualWriteSelection}
        writeRunning={manualWriteRunning}
        readBoardVout={readBoardVout}
        readBoardInductance={readBoardInductance}
        readBoardXdpPid={readBoardXdpPid}
        writeManualTuning={writeManualTuning}
        writeAllManualParametersForAcquisition={writeAllManualParametersForAcquisition}
        resetManualDefaults={resetManualDefaults}
        labels={t}
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

function HeaderToggles({
  language,
  setLanguage,
  themeMode,
  setThemeMode
}: {
  language: Language;
  setLanguage: (language: Language) => void;
  themeMode: ThemeMode;
  setThemeMode: (themeMode: ThemeMode) => void;
}) {
  return (
    <div className="header-toggles" aria-label="Display controls">
      <div className="pill-toggle language-toggle" role="group" aria-label="Language">
        <button
          type="button"
          className={language === "zh" ? "active" : ""}
          aria-pressed={language === "zh"}
          onClick={() => setLanguage("zh")}
        >
          中
        </button>
        <button
          type="button"
          className={language === "en" ? "active" : ""}
          aria-pressed={language === "en"}
          onClick={() => setLanguage("en")}
        >
          EN
        </button>
      </div>
      <div className="pill-toggle icon-toggle" role="group" aria-label="Theme">
        <button
          type="button"
          className={themeMode === "light" ? "active" : ""}
          aria-pressed={themeMode === "light"}
          onClick={() => setThemeMode("light")}
        >
          <Sun size={15} />
        </button>
        <button
          type="button"
          className={themeMode === "dark" ? "active" : ""}
          aria-pressed={themeMode === "dark"}
          onClick={() => setThemeMode("dark")}
        >
          <Moon size={15} />
        </button>
      </div>
    </div>
  );
}

function GoogleMark() {
  return (
    <svg className="google-mark" viewBox="0 0 48 48" aria-label="Google logo" role="img">
      <path
        fill="#FFC107"
        d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"
      />
      <path
        fill="#FF3D00"
        d="M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z"
      />
      <path
        fill="#4CAF50"
        d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238C29.211 35.091 26.715 36 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z"
      />
      <path
        fill="#1976D2"
        d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571l.003-.002 6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z"
      />
    </svg>
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
  inductance,
  inductanceRequest,
  setInductanceRequest,
  xdpPid,
  xdpPidRequest,
  setXdpPidRequest,
  runAction,
  readBoardInductance,
  writeOutputInductance,
  writeEffectiveLcInductance
}: {
  config: TuningConfig;
  setConfig: React.Dispatch<React.SetStateAction<TuningConfig>>;
  status: TuningStatus | null;
  current: IterationRecord | null;
  best: IterationRecord | null;
  history: IterationRecord[];
  vout: VoutReadback | null;
  inductance: InductanceReadback | null;
  inductanceRequest: { output_nh: number; effective_lc_nh: number };
  setInductanceRequest: React.Dispatch<React.SetStateAction<{ output_nh: number; effective_lc_nh: number }>>;
  xdpPid: XdpPidReadback | null;
  xdpPidRequest: Record<string, number>;
  setXdpPidRequest: React.Dispatch<React.SetStateAction<Record<string, number>>>;
  runAction: (action: "start" | "stop" | "step") => Promise<void>;
  readBoardInductance: () => Promise<void>;
  writeOutputInductance: () => Promise<void>;
  writeEffectiveLcInductance: () => Promise<void>;
}) {
  const voutExponent = voutExponentFromReadback(vout);
  const voutRegisterStep = voutRegisterStepFromExponent(voutExponent);
  return (
      <div className="workspace">
        <aside className="control-rail">
          <Panel title="Run Control" icon={<Activity size={17} />}>
            <div className="autotune-controls">
              <button className="autotune-control-button start" onClick={() => runAction("start")} disabled={status?.state === "running"}>
                Start Auto-Tune
              </button>
              <button className="autotune-control-button iterate" onClick={() => runAction("step")} disabled={status?.state === "running"}>
                Run Single Iteration
              </button>
              <div className="autotune-control-grid">
                <button className="autotune-control-button" disabled>
                  Pause
                </button>
                <button className="autotune-control-button" disabled>
                  Resume
                </button>
                <button className="autotune-control-button" onClick={() => runAction("stop")} disabled={status?.state !== "running"}>
                  Stop
                </button>
              </div>
              <button className="autotune-control-button" disabled>
                Save Animation GIF
              </button>
              <button className="autotune-control-button reset" onClick={() => setConfig(defaultConfig)}>
                Reset to Defaults
              </button>
            </div>
            <p className="message-line">{status?.message ?? "Connecting to local backend..."}</p>
          </Panel>

          <Panel title="Targets" icon={<SlidersHorizontal size={17} />}>
            <NumberField
              label="Vout target (V)"
              value={snapVoutToRegister(config.targets.vout_target_v, voutExponent)}
              step={voutRegisterStep}
              displayDigits={voutDisplayDigits}
              onChange={(value) => updateConfig(setConfig, "targets", "vout_target_v", snapVoutToRegister(value, voutExponent))}
            />
            <NumberField label="Overshoot max (%)" value={config.targets.overshoot_pct} onChange={(value) => updateConfig(setConfig, "targets", "overshoot_pct", value)} />
            <NumberField label="Undershoot max (%)" value={config.targets.undershoot_pct} onChange={(value) => updateConfig(setConfig, "targets", "undershoot_pct", value)} />
            <NumberField label="Settling target (us)" value={config.targets.settling_time_s * 1e6} onChange={(value) => updateConfig(setConfig, "targets", "settling_time_s", value / 1e6)} />
            <NumberField label="Phase Margin target (deg)" value={config.targets.phase_margin_deg} onChange={(value) => updateConfig(setConfig, "targets", "phase_margin_deg", value)} />
            <NumberField label="Crossover Frequency target (Hz)" value={config.targets.crossover_frequency_hz} onChange={(value) => updateConfig(setConfig, "targets", "crossover_frequency_hz", value)} />
          </Panel>

          <Panel title="Search Space" icon={<RefreshCw size={17} />}>
            <NumberField label="max iterations" value={config.search.max_iterations} onChange={(value) => updateConfig(setConfig, "search", "max_iterations", Math.max(1, Math.round(value)))} />
            {xdpPidFieldNames.map((name) => (
              <NumberField
                key={name}
                label={`${name} ${xdpPidLimitLabel(name, xdpPid).replace(`${name} `, "")}`}
                value={xdpPidRequest[name] ?? 0}
                step={1}
                onChange={(value) => {
                  const limit = xdpPid?.pid_registers?.fields?.[name] ?? defaultXdpPidLimits[name];
                  setXdpPidRequest({
                    ...xdpPidRequest,
                    [name]: clamp(Math.round(value), limit.min, limit.max)
                  });
                }}
              />
            ))}
            <NumberField
              label="Output Inductance (nH)"
              value={inductanceRequest.output_nh}
              onChange={(value) => setInductanceRequest({ ...inductanceRequest, output_nh: value })}
            />
            <p className="message-line">{outputInductanceLimitText}</p>
            <NumberField
              label="Effective Lc Inductance (nH)"
              value={inductanceRequest.effective_lc_nh}
              onChange={(value) => setInductanceRequest({ ...inductanceRequest, effective_lc_nh: value })}
            />
            <p className="message-line">{effectiveLcInductanceLimitText}</p>
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

function ManualTuningView({
  vout,
  voutRequest,
  setVoutRequest,
  inductance,
  inductanceRequest,
  setInductanceRequest,
  xdpPid,
  xdpPidRequest,
  setXdpPidRequest,
  pmbusOutput,
  pmbusOutputRunning,
  readBoardPmbusOutput,
  writeBoardPmbusOutput,
  xdpOutput,
  xdpOutputRunning,
  readBoardXdpOutput,
  writeBoardXdpOutput,
  bodeSweep,
  bodeSweepConfig,
  setBodeSweepConfig,
  setBodeSweep,
  bodeSweepRunning,
  setBodeSweepRunning,
  bodeSweepStatus,
  setBodeSweepStatus,
  runManualBodeSweep,
  setError,
  telemetryHistory,
  liveRefresh,
  setLiveRefresh,
  liveRateHz,
  setLiveRateHz,
  realTimeWriting,
  setRealTimeWriting,
  queueRealTimeWrite,
  writeSelection,
  setWriteSelection,
  writeRunning,
  readBoardVout,
  readBoardInductance,
  readBoardXdpPid,
  writeManualTuning,
  writeAllManualParametersForAcquisition,
  resetManualDefaults,
  labels
}: {
  vout: VoutReadback | null;
  voutRequest: { address: string; page: number; adapter: string; voltage: number };
  setVoutRequest: React.Dispatch<React.SetStateAction<{ address: string; page: number; adapter: string; voltage: number }>>;
  inductance: InductanceReadback | null;
  inductanceRequest: { output_nh: number; effective_lc_nh: number };
  setInductanceRequest: React.Dispatch<React.SetStateAction<{ output_nh: number; effective_lc_nh: number }>>;
  xdpPid: XdpPidReadback | null;
  xdpPidRequest: Record<string, number>;
  setXdpPidRequest: React.Dispatch<React.SetStateAction<Record<string, number>>>;
  pmbusOutput: PmbusOutputReadback | null;
  pmbusOutputRunning: boolean;
  readBoardPmbusOutput: () => Promise<void>;
  writeBoardPmbusOutput: (action: PmbusOutputAction) => Promise<void>;
  xdpOutput: XdpOutputReadback | null;
  xdpOutputRunning: boolean;
  readBoardXdpOutput: () => Promise<void>;
  writeBoardXdpOutput: (action: XdpOutputAction) => Promise<void>;
  bodeSweep: BodeSweepReadback | null;
  bodeSweepConfig: BodeSweepConfig;
  setBodeSweepConfig: React.Dispatch<React.SetStateAction<BodeSweepConfig>>;
  setBodeSweep: React.Dispatch<React.SetStateAction<BodeSweepReadback | null>>;
  bodeSweepRunning: boolean;
  setBodeSweepRunning: React.Dispatch<React.SetStateAction<boolean>>;
  bodeSweepStatus: string;
  setBodeSweepStatus: React.Dispatch<React.SetStateAction<string>>;
  runManualBodeSweep: () => Promise<BodeSweepReadback | null>;
  setError: React.Dispatch<React.SetStateAction<string>>;
  telemetryHistory: VoutReadback[];
  liveRefresh: boolean;
  setLiveRefresh: React.Dispatch<React.SetStateAction<boolean>>;
  liveRateHz: number;
  setLiveRateHz: React.Dispatch<React.SetStateAction<number>>;
  realTimeWriting: boolean;
  setRealTimeWriting: React.Dispatch<React.SetStateAction<boolean>>;
  queueRealTimeWrite: (overrides: Partial<ManualWriteSnapshot>) => void;
  writeSelection: { vout: boolean; output_inductance: boolean; effective_lc_inductance: boolean; xdp_pid: boolean };
  setWriteSelection: React.Dispatch<React.SetStateAction<{ vout: boolean; output_inductance: boolean; effective_lc_inductance: boolean; xdp_pid: boolean }>>;
  writeRunning: boolean;
  readBoardVout: () => Promise<void>;
  readBoardInductance: () => Promise<void>;
  readBoardXdpPid: () => Promise<void>;
  writeManualTuning: () => Promise<void>;
  writeAllManualParametersForAcquisition: () => Promise<void>;
  resetManualDefaults: () => void;
  labels: (typeof copy)[Language];
}) {
  const voutExponent = voutExponentFromReadback(vout);
  const voutRegisterStep = voutRegisterStepFromExponent(voutExponent);
  const [showConnectionSettings, setShowConnectionSettings] = useState(false);
  const [axisEditor, setAxisEditor] = useState<"telemetry" | "scope" | "bode" | null>(null);
  const [telemetryAxisSettings, setTelemetryAxisSettings] = useState<TelemetryAxisSettings>({ ...defaultTelemetryAxisSettings });
  const [scopeAxisSettings, setScopeAxisSettings] = useState<ScopeAxisSettings>({ ...defaultScopeAxisSettings });
  const [bodeAxisSettings, setBodeAxisSettings] = useState<ChartAxisSettings>({ ...defaultBodeAxisSettings });
  const [scopeChannels, setScopeChannels] = useState<string[]>(defaultScopeChannels);
  const [scopeMeasurements, setScopeMeasurements] = useState<string[]>(defaultScopeMeasurements);
  const [scopeCapture, setScopeCapture] = useState<ScopeCaptureReadback | null>(null);
  const [scopeWebPlotOpen, setScopeWebPlotOpen] = useState(true);
  const [scopePngPlotOpen, setScopePngPlotOpen] = useState(true);
  const [bodeWebPlotOpen, setBodeWebPlotOpen] = useState(true);
  const [bodePngPlotOpen, setBodePngPlotOpen] = useState(true);
  const [scopeRunning, setScopeRunning] = useState(false);
  const [scopeAcquisitionRunning, setScopeAcquisitionRunning] = useState(false);
  const [scopeAcquisitionBusy, setScopeAcquisitionBusy] = useState(false);
  const [fgConfig, setFgConfig] = useState({ ...defaultFunctionGeneratorConfig });
  const [fgReadback, setFgReadback] = useState<FunctionGeneratorReadback | null>(null);
  const [fgRunning, setFgRunning] = useState(false);
  const [fullAcquisitionRunning, setFullAcquisitionRunning] = useState(false);
  const [fullAcquisitionStatus, setFullAcquisitionStatus] = useState("");
  const [fullAcquisitionDurationS, setFullAcquisitionDurationS] = useState<number | null>(null);
  const [powerSupply, setPowerSupplyState] = useState<PowerSupplyReadback | null>(null);
  const powerSupplyRateHz = 5;
  const [powerSupplyLive, setPowerSupplyLive] = useState(false);
  const [powerSupplyRequest, setPowerSupplyRequest] = useState({ voltage_v: 54.5, current_limit_a: 8 });
  const [powerSupplyRunning, setPowerSupplyRunning] = useState(false);
  const powerSupplyPollInFlight = useRef(false);

  useEffect(() => {
    warmScope().catch(() => undefined);
  }, []);

  const runScopeCapture = async () => {
    try {
      setScopeRunning(true);
      setScopeCapture(null);
      setScopeCapture(await captureScope({
        channels: scopeChannels,
        measurements: scopeMeasurements,
        function_generator_frequency_hz: fgConfig.frequency_hz,
        scope_axis_settings: scopeAxisSettings
      }));
      setScopeAcquisitionRunning(false);
    } catch (exc) {
      setScopeCapture({ ok: false, error: String(exc), timestamp: Date.now() / 1000 });
    } finally {
      setScopeRunning(false);
    }
  };

  const setScopeRunState = async (running: boolean) => {
    try {
      setScopeAcquisitionBusy(true);
      const result = await setScopeAcquisition(running);
      if (result.ok) {
        setScopeAcquisitionRunning(running);
      } else {
        setScopeCapture({ ok: false, error: result.error ?? "Scope acquisition command failed.", timestamp: Date.now() / 1000 });
      }
    } catch (exc) {
      setScopeCapture({ ok: false, error: String(exc), timestamp: Date.now() / 1000 });
    } finally {
      setScopeAcquisitionBusy(false);
    }
  };

  const readAfGenerator = async () => {
    try {
      setFgRunning(true);
      setFgReadback(await readFunctionGenerator(undefined, fgConfig.channel));
    } catch (exc) {
      setFgReadback({ ok: false, error: String(exc), timestamp: Date.now() / 1000 });
    } finally {
      setFgRunning(false);
    }
  };

  const writeAfGenerator = async () => {
    try {
      setFgRunning(true);
      setFgReadback(await setFunctionGenerator(fgConfig));
    } catch (exc) {
      setFgReadback({ ok: false, error: String(exc), timestamp: Date.now() / 1000 });
    } finally {
      setFgRunning(false);
    }
  };

  const setAfGeneratorOutput = async (output_enabled: boolean) => {
    try {
      setFgRunning(true);
      setFgReadback(
        await setFunctionGenerator({
          channel: fgConfig.channel,
          mode: fgConfig.mode,
          output_enabled
        })
      );
    } catch (exc) {
      setFgReadback({ ok: false, error: String(exc), timestamp: Date.now() / 1000 });
    } finally {
      setFgRunning(false);
    }
  };

  const readSupply = async (showBusy = true) => {
    if (!showBusy && powerSupplyPollInFlight.current) return;
    try {
      if (showBusy) {
        setPowerSupplyRunning(true);
      } else {
        powerSupplyPollInFlight.current = true;
      }
      const next = await readPowerSupply();
      setPowerSupplyState(next);
    } catch (exc) {
      setPowerSupplyState({ ok: false, error: String(exc), timestamp: Date.now() / 1000 });
    } finally {
      if (showBusy) {
        setPowerSupplyRunning(false);
      } else {
        powerSupplyPollInFlight.current = false;
      }
    }
  };

  const writeSupply = async () => {
    try {
      setPowerSupplyRunning(true);
      setPowerSupplyState(await setPowerSupply(powerSupplyRequest));
    } catch (exc) {
      setPowerSupplyState({ ok: false, error: String(exc), timestamp: Date.now() / 1000 });
    } finally {
      setPowerSupplyRunning(false);
    }
  };

  const setSupplyOutput = async (output_enabled: boolean) => {
    try {
      setPowerSupplyRunning(true);
      setPowerSupplyState(await setPowerSupply({ output_enabled }));
    } catch (exc) {
      setPowerSupplyState({ ok: false, error: String(exc), timestamp: Date.now() / 1000 });
    } finally {
      setPowerSupplyRunning(false);
    }
  };

  useEffect(() => {
    if (!powerSupplyLive) return;
    readSupply(false);
    const timer = window.setInterval(() => readSupply(false), 1000 / Math.max(1, Math.min(10, powerSupplyRateHz)));
    return () => window.clearInterval(timer);
  }, [powerSupplyLive, powerSupplyRateHz]);

  const readAllManualBlocks = async () => {
    await readBoardVout();
    await readBoardInductance();
    await readBoardXdpPid();
    await readBoardPmbusOutput();
    await readBoardXdpOutput();
  };

  const runFullDataAcquisition = async () => {
    const started = performance.now();
    const acquisitionFgConfig = { ...fgConfig };
    let afgEnabled = false;
    try {
      setFullAcquisitionRunning(true);
      setFullAcquisitionDurationS(null);
      setError("");
      setFullAcquisitionStatus("Writing manual parameters to XDP...");
      await writeAllManualParametersForAcquisition();
      setFullAcquisitionStatus("Running Bode 100 sweep...");
      setBodeSweepRunning(true);
      setBodeSweepStatus("Running Bode 100 sweep...");
      const bode = await runBodeSweep(bodeSweepConfig);
      setBodeSweep(bode);
      if (bode.ok === false) {
        setBodeSweepStatus(bode.error ?? "Bode sweep failed");
        throw new Error(bode?.error ?? "Bode sweep failed");
      }
      setBodeSweepStatus(bode.duration_s ? `Done in ${bode.duration_s.toFixed(2)} s` : "Sweep complete");

      setFullAcquisitionStatus(`Applying function generator settings on CH${acquisitionFgConfig.channel}...`);
      setFgRunning(true);
      setFgReadback(await setFunctionGenerator(acquisitionFgConfig));
      setFullAcquisitionStatus("Enabling function generator output...");
      setFgReadback(await setFunctionGenerator({
        channel: acquisitionFgConfig.channel,
        mode: acquisitionFgConfig.mode,
        output_enabled: true
      }));
      afgEnabled = true;
      await new Promise((resolve) => window.setTimeout(resolve, 120));

      setFullAcquisitionStatus("Capturing scope waveform...");
      setScopeRunning(true);
      setScopeCapture(null);
      const scope = await captureScope({
        channels: scopeChannels,
        measurements: scopeMeasurements,
        function_generator_frequency_hz: acquisitionFgConfig.frequency_hz,
        scope_axis_settings: scopeAxisSettings
      });
      setScopeCapture(scope);
      setScopeAcquisitionRunning(false);
      if (scope.ok === false) {
        throw new Error(scope.error ?? "Scope capture failed");
      }
      setFullAcquisitionStatus("");
    } catch (exc) {
      const message = String(exc);
      setError(message);
      setFullAcquisitionStatus(message);
    } finally {
      setBodeSweepRunning(false);
      setScopeRunning(false);
      if (afgEnabled) {
        try {
          setFullAcquisitionStatus((current) => current ? `${current} / Disabling AFG...` : "");
          setFgReadback(await setFunctionGenerator({
            channel: acquisitionFgConfig.channel,
            mode: acquisitionFgConfig.mode,
            output_enabled: false
          }));
          setFullAcquisitionStatus((current) => current.replace(" / Disabling AFG...", ""));
        } catch (exc) {
          setError(`AFG disable failed: ${String(exc)}`);
        }
      }
      setFgRunning(false);
      setFullAcquisitionRunning(false);
      setFullAcquisitionDurationS((performance.now() - started) / 1000);
    }
  };

  const displayTelemetryHistory = smoothTelemetryHistory(telemetryHistory);
  const latestSmoothedTelemetry = displayTelemetryHistory.at(-1);
  const displayVout = latestSmoothedTelemetry && vout
    ? { ...vout, ...latestSmoothedTelemetry }
    : latestSmoothedTelemetry ?? vout;

  return (
    <section className="manual-page">
      <div className="manual-grid">
        <aside className="manual-controls">
          <Panel title="Data Acquisition" icon={<Activity size={17} />}>
            <button className="primary wide-button" type="button" onClick={runFullDataAcquisition} disabled={fullAcquisitionRunning}>
              {fullAcquisitionRunning ? <LoaderCircle className="spin" size={16} /> : <RefreshCw size={16} />}
              Full Data Acquisition
            </button>
            {fullAcquisitionStatus ? (
              <p className={`micro-readback ${fullAcquisitionRunning ? "running-text" : ""}`}>{fullAcquisitionStatus}</p>
            ) : null}
            <p className="micro-readback">Total time: {fullAcquisitionDurationS === null ? "--" : `${fullAcquisitionDurationS.toFixed(2)} s`}</p>
          </Panel>

          <Panel title="XDP Connection" icon={<Activity size={17} />}>
            <div className="connection-summary">
              <span>XDP Address</span>
              <strong>{voutRequest.address}</strong>
              <button type="button" onClick={() => setShowConnectionSettings((open) => !open)}>
                {showConnectionSettings ? "Hide" : "Edit"}
              </button>
            </div>
            {showConnectionSettings ? (
              <div className="connection-settings">
                <TextField label="Address" value={voutRequest.address} onChange={(value) => setVoutRequest({ ...voutRequest, address: value })} />
                <NumberField label="Page" value={voutRequest.page} onChange={(value) => setVoutRequest({ ...voutRequest, page: Math.max(0, Math.round(value)) })} />
                <TextField label="Adapter" value={voutRequest.adapter} onChange={(value) => setVoutRequest({ ...voutRequest, adapter: value })} />
              </div>
            ) : null}
            <LiveRefreshButton enabled={liveRefresh} label={labels.liveReadback} onChange={setLiveRefresh} />
          </Panel>

          <Panel title="Manual Parameters" icon={<SlidersHorizontal size={17} />}>
            <button className="reset-default-button" type="button" onClick={resetManualDefaults}>
              <RotateCcw size={14} /> Reset to Default
            </button>
            <LiveRefreshButton enabled={realTimeWriting} label="Real Time Writing" onChange={setRealTimeWriting} />
            <CheckboxField
              label="Write VOUT_COMMAND"
              checked={writeSelection.vout}
              onChange={(checked) => setWriteSelection({ ...writeSelection, vout: checked })}
            />
            <NumberField
              label="Vout target (V)"
              value={snapVoutToRegister(voutRequest.voltage, voutExponent)}
              step={voutRegisterStep}
              displayDigits={voutDisplayDigits}
              commitOnSpinChange={realTimeWriting}
              commitOnBlur={!realTimeWriting}
              onChange={(value) => {
                const voltage = snapVoutToRegister(value, voutExponent);
                setVoutRequest({ ...voutRequest, voltage });
                queueRealTimeWrite({ voltage });
              }}
            />
            <CheckboxField
              label="Write Output Inductance"
              checked={writeSelection.output_inductance}
              onChange={(checked) => setWriteSelection({ ...writeSelection, output_inductance: checked })}
            />
            <NumberField
              label="Output Inductance (nH)"
              value={inductanceRequest.output_nh}
              commitOnChange={realTimeWriting}
              onChange={(value) => {
                setInductanceRequest({ ...inductanceRequest, output_nh: value });
                queueRealTimeWrite({ output_nh: value });
              }}
            />
            <CheckboxField
              label="Write Effective Lc Inductance"
              checked={writeSelection.effective_lc_inductance}
              onChange={(checked) => setWriteSelection({ ...writeSelection, effective_lc_inductance: checked })}
            />
            <NumberField
              label="Effective Lc Inductance (nH)"
              value={inductanceRequest.effective_lc_nh}
              commitOnChange={realTimeWriting}
              onChange={(value) => {
                setInductanceRequest({ ...inductanceRequest, effective_lc_nh: value });
                queueRealTimeWrite({ effective_lc_nh: value });
              }}
            />
            <CheckboxField
              label="Write mod0 PID register"
              checked={writeSelection.xdp_pid}
              onChange={(checked) => setWriteSelection({ ...writeSelection, xdp_pid: checked })}
            />
            {xdpPidFieldNames.map((name) => (
              <NumberField
                key={name}
                label={xdpPidLimitLabel(name, xdpPid)}
                value={xdpPidRequest[name] ?? 0}
                commitOnChange={realTimeWriting}
                onChange={(value) => {
                  const pid = { ...xdpPidRequest, [name]: Math.max(0, Math.round(value)) };
                  setXdpPidRequest(pid);
                  queueRealTimeWrite({ pid });
                }}
              />
            ))}
            <button className="primary wide-button" onClick={writeManualTuning} disabled={writeRunning}>
              {writeRunning ? <LoaderCircle className="spin" size={16} /> : <Zap size={16} />}
              {writeRunning ? "Writing..." : "Write Selected to XDP"}
            </button>
          </Panel>

          <CollapsiblePanel title="PMBus Output Control" icon={<PowerIcon size={17} />} defaultOpen={false}>
            <div className="control-card">
              <div className="segmented-action-row pmbus-output-row">
                <button
                  type="button"
                  className={formatPmbusOutputState(pmbusOutput) === "On" ? "active" : ""}
                  disabled={pmbusOutputRunning}
                  onClick={() => writeBoardPmbusOutput("on")}
                >
                  Enable
                </button>
                <button
                  type="button"
                  className={formatPmbusOutputState(pmbusOutput) === "Off" ? "active" : ""}
                  disabled={pmbusOutputRunning}
                  onClick={() => writeBoardPmbusOutput("off")}
                >
                  Disable
                </button>
              </div>
              <p className="micro-readback">
                OPERATION {pmbusOutput?.operation_after ?? pmbusOutput?.operation ?? "--"} / STATUS {pmbusOutput?.status_word ?? "--"}
              </p>
            </div>
          </CollapsiblePanel>

          <CollapsiblePanel title="XDP Output Control" icon={<PowerIcon size={17} />} defaultOpen={false}>
            <div className="control-card">
              <div className="segmented-action-row xdp-output-row">
                <button
                  type="button"
                  className={formatXdpOutputState(xdpOutput) === "High" ? "active" : ""}
                  disabled={xdpOutputRunning}
                  onClick={() => writeBoardXdpOutput("enable")}
                >
                  Enable
                </button>
                <button
                  type="button"
                  className={formatXdpOutputState(xdpOutput) === "Low" ? "active" : ""}
                  disabled={xdpOutputRunning}
                  onClick={() => writeBoardXdpOutput("disable")}
                >
                  Disable
                </button>
                <button
                  type="button"
                  className={formatXdpOutputState(xdpOutput) === "Release" ? "active" : ""}
                  disabled={xdpOutputRunning}
                  onClick={() => writeBoardXdpOutput("release")}
                >
                  Release
                </button>
              </div>
              <p className="micro-readback">
                XDP {formatXdpOutputState(xdpOutput)} / byte {xdpOutput?.readback?.byte ?? "--"} / STATUS {xdpOutput?.status_word ?? "--"}
              </p>
            </div>
          </CollapsiblePanel>

        </aside>

        <section className="manual-monitor">
          <Panel title="Live Data" icon={<Activity size={17} />}>
            <div className="telemetry-cards">
              <LiveValue label="Vout" value={displayVout?.read_vout_v} unit="V" />
              <LiveValue label="Iout" value={displayVout?.read_iout_a} unit="A" />
              <LiveValue label="Vout_Command" value={displayVout?.vout_command_v} unit="V" />
              <RateValue
                value={liveRateHz}
                min={liveTelemetryMinHz}
                max={liveTelemetryMaxHz}
                onChange={(value) => setLiveRateHz(clampTelemetryRate(value))}
              />
            </div>
            <div className="chart-shell" onDoubleClick={() => setAxisEditor("telemetry")}>
              <ReactECharts option={telemetryOption(displayTelemetryHistory, telemetryAxisSettings)} className="chart tall" lazyUpdate />
            </div>
          </Panel>

          <Panel title="Scope Capture" icon={<Activity size={17} />}>
            <div className="instrument-controls">
              <div className="inline-options">
                {Array.from({ length: 8 }, (_, index) => `CH${index + 1}`).map((channel) => (
                  <div key={channel} className={`scope-channel-chip ${scopeChannels.includes(channel) ? "selected" : ""}`}>
                    <label>
                      <input
                        type="checkbox"
                        checked={scopeChannels.includes(channel)}
                        onChange={(event) => {
                          setScopeChannels((current) => event.target.checked
                            ? [...new Set([...current, channel])]
                            : current.filter((item) => item !== channel));
                        }}
                      />
                      <span>{channel}</span>
                    </label>
                    <select
                      value={scopeAxisSettings.channelAxes[channel] ?? "left"}
                      onChange={(event) => {
                        const side = event.target.value === "right" ? "right" : "left";
                        setScopeAxisSettings((current) => ({
                          ...current,
                          channelAxes: { ...current.channelAxes, [channel]: side }
                        }));
                      }}
                    >
                      <option value="left">Left</option>
                      <option value="right">Right</option>
                    </select>
                  </div>
                ))}
              </div>
              <div className="inline-options">
                {["MEAN", "PK2PK", "FREQ", "RMS", "MAX", "MIN"].map((measurement) => (
                  <label key={measurement} className="chip-check">
                    <input
                      type="checkbox"
                      checked={scopeMeasurements.includes(measurement)}
                      onChange={(event) => {
                        setScopeMeasurements((current) => event.target.checked
                          ? [...new Set([...current, measurement])]
                          : current.filter((item) => item !== measurement));
                      }}
                    />
                    {measurement}
                  </label>
                ))}
              </div>
              <div className="compact-form-row">
                <ScopeAcquisitionToggle
                  running={scopeAcquisitionRunning}
                  busy={scopeAcquisitionBusy}
                  onChange={setScopeRunState}
                />
                <button className="primary compact-action" type="button" onClick={runScopeCapture} disabled={scopeRunning}>
                  {scopeRunning ? <LoaderCircle className="spin" size={16} /> : <RefreshCw size={16} />}
                  Read Scope
                </button>
              </div>
            </div>
            {scopeCapture?.error ? <p className="message-line bad-text">{scopeCapture.error}</p> : null}
            <p className={`message-line ${scopeRunning ? "running-text" : scopeCapture?.error ? "bad-text" : ""}`}>
              {scopeRunning
                ? "Reading scope..."
                : scopeCapture?.duration_s !== undefined
                  ? `Done in ${scopeCapture.duration_s.toFixed(2)} s`
                  : "Ready"}
            </p>
            {scopeCapture?.identity ? <p className="message-line">{scopeCapture.identity}</p> : null}
            <PlotToggleSection
              title="Scope Web Plot"
              open={scopeWebPlotOpen}
              onToggle={() => setScopeWebPlotOpen((current) => !current)}
            >
              <div className="chart-shell" onDoubleClick={() => setAxisEditor("scope")}>
                <ReactECharts
                  key={scopeChartKey(scopeCapture)}
                  option={scopeWaveformOption(scopeCapture, scopeAxisSettings)}
                  className="chart scope-chart"
                  notMerge
                />
              </div>
            </PlotToggleSection>
            <PlotToggleSection
              title="Scope PNG Plot"
              open={scopePngPlotOpen}
              onToggle={() => setScopePngPlotOpen((current) => !current)}
              summary={scopeCapture?.scope_png ? "full data" : "not available"}
            >
              {scopeCapture?.scope_png ? (
                <div className="scope-png-preview">
                  <img
                    src={`${scopeCapture.scope_png}?v=${encodeURIComponent(scopeCapture.capture_id ?? String(scopeCapture.timestamp ?? Date.now()))}`}
                    alt="Latest full scope capture"
                  />
                </div>
              ) : <p className="message-line">No scope PNG yet.</p>}
            </PlotToggleSection>
            {scopeCapture?.scope_png_error ? <p className="message-line bad-text">{scopeCapture.scope_png_error}</p> : null}
            <ScopeMeasurementTable capture={scopeCapture} />
          </Panel>

          <Panel title="Bode 100 Sweep" icon={<Activity size={17} />}>
            <div className="bode-controls">
              <FrequencyField
                label="Start (Hz)"
                value={bodeSweepConfig.start_hz}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, start_hz: Math.max(1, value) }))}
              />
              <FrequencyField
                label="Stop (Hz)"
                value={bodeSweepConfig.stop_hz}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, stop_hz: Math.max(1, value) }))}
              />
              <NumberField
                label="Points"
                value={bodeSweepConfig.points}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, points: Math.max(2, Math.round(value)) }))}
              />
              <NumberField
                label="RBW (Hz)"
                value={bodeSweepConfig.bandwidth_hz}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, bandwidth_hz: Math.max(1, value) }))}
              />
              <NumberField
                label="Source (dBm)"
                value={bodeSweepConfig.source_dbm ?? 0}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, source_dbm: value }))}
              />
              <button className="primary bode-run-button" onClick={runManualBodeSweep} disabled={bodeSweepRunning}>
                {bodeSweepRunning ? <LoaderCircle className="spin" size={16} /> : <RefreshCw size={16} />}
                {bodeSweepRunning ? "Running..." : "Run Bode 100"}
              </button>
            </div>
            <p className={`message-line bode-status ${bodeSweepRunning ? "running-text" : bodeSweep?.error ? "bad-text" : ""}`}>
              {bodeSweepStatus}
            </p>
            {bodeSweep?.error ? <p className="message-line bad-text">{bodeSweep.error}</p> : null}
            {bodeSweep?.identity ? <p className="message-line">{bodeSweep.identity}</p> : null}
            <BodeMarginReadout sweep={bodeSweep} />
            <PlotToggleSection
              title="Bode Web Plot"
              open={bodeWebPlotOpen}
              onToggle={() => setBodeWebPlotOpen((current) => !current)}
            >
              <div className="chart-shell bode-chart-shell" onDoubleClick={() => setAxisEditor("bode")}>
                <ReactECharts
                  option={bodeSweepOption(bodeSweep, bodeAxisSettings)}
                  className="chart bode-chart"
                  style={{ height: "100%", width: "100%" }}
                />
              </div>
            </PlotToggleSection>
            <PlotToggleSection
              title="Bode PNG Plot"
              open={bodePngPlotOpen}
              onToggle={() => setBodePngPlotOpen((current) => !current)}
              summary={bodeSweep?.bode_png ? "full data" : "not available"}
            >
              {bodeSweep?.bode_png ? (
                <div className="scope-png-preview">
                  <img
                    src={`${bodeSweep.bode_png}?v=${encodeURIComponent(bodeSweep.sweep_id ?? String(bodeSweep.timestamp ?? Date.now()))}`}
                    alt="Latest full Bode 100 sweep"
                  />
                </div>
              ) : <p className="message-line">No Bode PNG yet.</p>}
            </PlotToggleSection>
            {bodeSweep?.bode_png_error ? <p className="message-line bad-text">{bodeSweep.bode_png_error}</p> : null}
          </Panel>

          <Panel
            title="Function Generator"
            icon={<Activity size={17} />}
          >
            <div className="fg-panel-layout">
              <section className="instrument-subsection fg-apply-section">
                <h3>Apply Settings</h3>
                <div className="instrument-form-grid fg-grid">
                  <SelectField label="Channel" value={String(fgConfig.channel)} options={["1", "2"]} onChange={(value) => setFgConfig((current) => ({ ...current, channel: Number(value) }))} />
                  <SelectField label="Waveform" value={fgConfig.mode} options={["square", "pulse", "dc", "sine"]} onChange={(value) => setFgConfig((current) => ({ ...current, mode: value }))} />
                  {fgConfig.mode !== "dc" ? (
                    <FrequencyField label="Frequency" value={fgConfig.frequency_hz} onChange={(value) => setFgConfig((current) => ({ ...current, frequency_hz: value }))} />
                  ) : null}
                  <SelectField label="Voltage Unit" value={fgConfig.voltage_unit} options={["VPP", "VRMS", "DBM"]} onChange={(value) => setFgConfig((current) => ({ ...current, voltage_unit: value }))} />
                  {fgConfig.mode === "square" || fgConfig.mode === "pulse" ? (
                    <>
                      <NumberField label="Low Level" value={fgConfig.low_v} onChange={(value) => setFgConfig((current) => ({ ...current, low_v: value }))} />
                      <NumberField label="High Level" value={fgConfig.high_v} onChange={(value) => setFgConfig((current) => ({ ...current, high_v: value }))} />
                    </>
                  ) : null}
                  {fgConfig.mode === "pulse" ? (
                    <NumberField label="Pulse Width" value={fgConfig.pulse_width_s} onChange={(value) => setFgConfig((current) => ({ ...current, pulse_width_s: value }))} />
                  ) : null}
                  {fgConfig.mode === "dc" ? (
                    <NumberField label="DC Level" value={fgConfig.dc_level_v} onChange={(value) => setFgConfig((current) => ({ ...current, dc_level_v: value }))} />
                  ) : null}
                  {fgConfig.mode === "sine" ? (
                    <>
                      <NumberField label="Amplitude" value={fgConfig.amplitude_vpp} onChange={(value) => setFgConfig((current) => ({ ...current, amplitude_vpp: value }))} />
                      <NumberField label="Offset" value={fgConfig.offset_v} onChange={(value) => setFgConfig((current) => ({ ...current, offset_v: value }))} />
                      <NumberField label="Phase" value={fgConfig.phase_deg} onChange={(value) => setFgConfig((current) => ({ ...current, phase_deg: value }))} />
                    </>
                  ) : null}
                  <button className="fg-inline-button" type="button" onClick={readAfGenerator} disabled={fgRunning}><RefreshCw size={14} /> Read</button>
                  <button className="primary fg-inline-button" type="button" onClick={writeAfGenerator} disabled={fgRunning}>
                    {fgRunning ? <LoaderCircle className="spin" size={14} /> : <Zap size={14} />}
                    Write
                  </button>
                  <span className={`fg-output-indicator ${isOutputEnabled(fgReadback?.output) ? "on" : "off"}`}>
                    {formatOutputState(fgReadback?.output)}
                  </span>
                  <button className="fg-inline-button output-enable-button" type="button" onClick={() => setAfGeneratorOutput(true)} disabled={fgRunning}>
                    <PowerIcon size={14} />
                    Enable
                  </button>
                  <button className="fg-inline-button output-disable-button" type="button" onClick={() => setAfGeneratorOutput(false)} disabled={fgRunning}>
                    <PowerIcon size={14} />
                    Disable
                  </button>
                </div>
              </section>

              <section className="instrument-subsection fg-state-section">
                <h3>Current Instrument State</h3>
                <FunctionGeneratorState readback={fgReadback} />
                {instrumentErrorText(fgReadback) ? (
                  <div className="instrument-error-banner">
                    <AlertTriangle size={16} />
                    {instrumentErrorText(fgReadback)}
                  </div>
                ) : null}
              </section>

            </div>
          </Panel>

          <Panel title="Power Supply" icon={<PowerIcon size={17} />}>
            <div className="supply-layout">
            <div className="telemetry-cards compact-cards supply-cards">
              <LiveValue label="MEAS_VOLT" value={powerSupply?.measured_voltage_v ?? undefined} unit="V" />
              <LiveValue label="MEAS_CURR" value={powerSupply?.measured_current_a ?? undefined} unit="A" />
              <LiveValue label="VOLT_SET" value={powerSupply?.voltage_setpoint_v ?? undefined} unit="V" />
              <LiveValue label="CURR_LIMIT" value={powerSupply?.current_limit_a ?? undefined} unit="A" />
            </div>
            <div className="instrument-form-grid supply-grid">
              <LiveRefreshButton enabled={powerSupplyLive} label="Read Only" onChange={setPowerSupplyLive} />
              <NumberField label="Voltage (V)" value={powerSupplyRequest.voltage_v} onChange={(value) => setPowerSupplyRequest((current) => ({ ...current, voltage_v: value }))} />
              <NumberField label="Current limit (A)" value={powerSupplyRequest.current_limit_a} onChange={(value) => setPowerSupplyRequest((current) => ({ ...current, current_limit_a: value }))} />
              <button type="button" onClick={readSupply} disabled={powerSupplyRunning}><RefreshCw size={14} /> Read</button>
              <button className="primary" type="button" onClick={writeSupply} disabled={powerSupplyRunning}>
                {powerSupplyRunning ? <LoaderCircle className="spin" size={14} /> : <Zap size={14} />}
                Write
              </button>
              <span className={`supply-output-indicator ${powerSupply?.output_enabled ? "on" : "off"}`}>
                {powerSupply?.output_enabled ? "ON" : "OFF"}
              </span>
              <button className="output-enable-button" type="button" onClick={() => setSupplyOutput(true)} disabled={powerSupplyRunning}>
                <PowerIcon size={14} />
                Enable
              </button>
              <button className="output-disable-button" type="button" onClick={() => setSupplyOutput(false)} disabled={powerSupplyRunning}>
                <PowerIcon size={14} />
                Disable
              </button>
            </div>
            </div>
            {powerSupply?.error ? <p className="message-line bad-text">{powerSupply.error}</p> : null}
            {powerSupply?.identity ? <p className="message-line">{powerSupply.identity}</p> : null}
          </Panel>

        </section>
      </div>
      {axisEditor === "telemetry" ? (
        <TelemetryAxisSettingsDialog
          title="Live Data Axis Settings"
          settings={telemetryAxisSettings}
          setSettings={setTelemetryAxisSettings}
          defaults={defaultTelemetryAxisSettings}
          labels={{
            durationSeconds: "Time window (s)",
            yMin: "Vout min (V)",
            yMax: "Vout max (V)",
            y2Min: "Iout min (A)",
            y2Max: "Iout max (A)"
          }}
          onClose={() => setAxisEditor(null)}
        />
      ) : null}
      {axisEditor === "bode" ? (
        <BodeAxisSettingsDialog
          title="Bode 100 Axis Settings"
          settings={bodeAxisSettings}
          setSettings={setBodeAxisSettings}
          defaults={defaultBodeAxisSettings}
          onClose={() => setAxisEditor(null)}
        />
      ) : null}
      {axisEditor === "scope" ? (
        <ScopeAxisSettingsDialog
          title="Scope Axis Settings"
          settings={scopeAxisSettings}
          setSettings={setScopeAxisSettings}
          defaults={defaultScopeAxisSettings}
          channels={scopeChannels}
          onClose={() => setAxisEditor(null)}
        />
      ) : null}
    </section>
  );
}

function ScopeAxisSettingsDialog({
  title,
  settings,
  setSettings,
  defaults,
  channels,
  onClose
}: {
  title: string;
  settings: ScopeAxisSettings;
  setSettings: React.Dispatch<React.SetStateAction<ScopeAxisSettings>>;
  defaults: ScopeAxisSettings;
  channels: string[];
  onClose: () => void;
}) {
  const update = (field: keyof Omit<ScopeAxisSettings, "channelAxes">, value: number) => {
    setSettings((current) => ({ ...current, [field]: value }));
  };
  const setChannelAxis = (channel: string, side: ScopeAxisSide) => {
    setSettings((current) => ({
      ...current,
      channelAxes: { ...current.channelAxes, [channel]: side }
    }));
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section className="axis-dialog scope-axis-dialog" onClick={(event) => event.stopPropagation()}>
        <div className="axis-dialog-title">
          <h2>{title}</h2>
        </div>
        <div className="telemetry-axis-grid">
          <div className="axis-pair-card scope-left-axis-card">
            <strong>Left Voltage Axis</strong>
            <div className="axis-pair-inputs">
              <MiniAxisField label="Min" value={settings.leftMin} onChange={(value) => update("leftMin", value)} />
              <MiniAxisField label="Max" value={settings.leftMax} onChange={(value) => update("leftMax", value)} />
            </div>
          </div>
          <div className="axis-pair-card scope-right-axis-card">
            <strong>Right Voltage Axis</strong>
            <div className="axis-pair-inputs">
              <MiniAxisField label="Min" value={settings.rightMin} onChange={(value) => update("rightMin", value)} />
              <MiniAxisField label="Max" value={settings.rightMax} onChange={(value) => update("rightMax", value)} />
            </div>
          </div>
          <div className="axis-pair-card">
            <strong>Channel Axis</strong>
            <div className="scope-axis-grid">
              {(channels.length > 0 ? channels : Array.from({ length: 8 }, (_, index) => `CH${index + 1}`)).map((channel) => (
                <label key={channel} className="scope-axis-select">
                  <span>{channel}</span>
                  <select
                    value={settings.channelAxes[channel] ?? "left"}
                    onChange={(event) => setChannelAxis(channel, event.target.value === "right" ? "right" : "left")}
                  >
                    <option value="left">Left</option>
                    <option value="right">Right</option>
                  </select>
                </label>
              ))}
            </div>
          </div>
        </div>
        <div className="axis-dialog-actions">
          <button type="button" onClick={() => setSettings({ ...defaults, channelAxes: { ...defaults.channelAxes } })}>
            <RotateCcw size={14} /> Reset
          </button>
          <button type="button" className="primary" onClick={onClose}>Done</button>
        </div>
      </section>
    </div>
  );
}

function BodeAxisSettingsDialog({
  title,
  settings,
  setSettings,
  defaults,
  onClose
}: {
  title: string;
  settings: ChartAxisSettings;
  setSettings: React.Dispatch<React.SetStateAction<ChartAxisSettings>>;
  defaults: ChartAxisSettings;
  onClose: () => void;
}) {
  const update = (field: keyof ChartAxisSettings, value: number) => {
    setSettings((current) => ({ ...current, [field]: value }));
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section className="axis-dialog" onClick={(event) => event.stopPropagation()}>
        <div className="axis-dialog-title">
          <h2>{title}</h2>
        </div>
        <div className="telemetry-axis-grid">
          <div className="axis-pair-card">
            <strong>Frequency (Hz)</strong>
            <div className="axis-pair-inputs">
              <MiniAxisField label="Min" value={settings.xMin} onChange={(value) => update("xMin", value)} />
              <MiniAxisField label="Max" value={settings.xMax} onChange={(value) => update("xMax", value)} />
            </div>
          </div>
          <div className="axis-pair-card">
            <strong>Gain (dB)</strong>
            <div className="axis-pair-inputs">
              <MiniAxisField label="Min" value={settings.yMin} onChange={(value) => update("yMin", value)} />
              <MiniAxisField label="Max" value={settings.yMax} onChange={(value) => update("yMax", value)} />
            </div>
          </div>
          <div className="axis-pair-card">
            <strong>Phase (deg)</strong>
            <div className="axis-pair-inputs">
              <MiniAxisField label="Min" value={settings.y2Min} onChange={(value) => update("y2Min", value)} />
              <MiniAxisField label="Max" value={settings.y2Max} onChange={(value) => update("y2Max", value)} />
            </div>
          </div>
        </div>
        <div className="axis-dialog-actions">
          <button type="button" onClick={() => setSettings({ ...defaults })}>
            <RotateCcw size={14} /> Reset
          </button>
          <button type="button" className="primary" onClick={onClose}>Done</button>
        </div>
      </section>
    </div>
  );
}

function TelemetryAxisSettingsDialog({
  title,
  settings,
  setSettings,
  defaults,
  labels,
  onClose
}: {
  title: string;
  settings: TelemetryAxisSettings;
  setSettings: React.Dispatch<React.SetStateAction<TelemetryAxisSettings>>;
  defaults: TelemetryAxisSettings;
  labels: Record<keyof TelemetryAxisSettings, string>;
  onClose: () => void;
}) {
  const update = (field: keyof TelemetryAxisSettings, value: number) => {
    setSettings((current) => ({
      ...current,
      [field]: field === "durationSeconds" ? clampTelemetryDuration(value) : value
    }));
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section className="axis-dialog" onClick={(event) => event.stopPropagation()}>
        <div className="axis-dialog-title">
          <h2>{title}</h2>
        </div>
        <div className="telemetry-axis-grid">
          <label className="axis-window-field">
            <span>{labels.durationSeconds}</span>
            <input
              type="number"
              min={1}
              max={telemetryHistoryWindowSeconds}
              value={settings.durationSeconds}
              onChange={(event) => update("durationSeconds", Number(event.target.value))}
            />
          </label>
          <div className="axis-pair-card">
            <strong>Vout</strong>
            <div className="axis-pair-inputs">
              <MiniAxisField label="Min" value={settings.yMin} onChange={(value) => update("yMin", value)} />
              <MiniAxisField label="Max" value={settings.yMax} onChange={(value) => update("yMax", value)} />
            </div>
          </div>
          <div className="axis-pair-card">
            <strong>Iout</strong>
            <div className="axis-pair-inputs">
              <MiniAxisField label="Min" value={settings.y2Min} onChange={(value) => update("y2Min", value)} />
              <MiniAxisField label="Max" value={settings.y2Max} onChange={(value) => update("y2Max", value)} />
            </div>
          </div>
        </div>
        <div className="axis-dialog-actions">
          <button type="button" onClick={() => setSettings({ ...defaults })}>
            <RotateCcw size={14} /> Reset
          </button>
          <button type="button" className="primary" onClick={onClose}>Done</button>
        </div>
      </section>
    </div>
  );
}

function MiniAxisField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="mini-axis-field">
      <span>{label}</span>
      <input type="number" value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
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

function Panel({
  title,
  icon,
  headerExtra,
  children
}: {
  title: string;
  icon: React.ReactNode;
  headerExtra?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <section className={`panel ${open ? "open" : "collapsed"}`}>
      <button type="button" className="panel-title panel-toggle" onClick={() => setOpen((current) => !current)}>
        <span className="panel-heading">
          {icon}
          <h2>{title}</h2>
        </span>
        <span className="panel-title-actions">
          {headerExtra}
          <span className="collapse-chevron">{open ? "-" : "+"}</span>
        </span>
      </button>
      {open ? <div className="panel-body">{children}</div> : null}
    </section>
  );
}

function CollapsiblePanel({
  title,
  icon,
  summary,
  defaultOpen = false,
  children
}: {
  title: string;
  icon: React.ReactNode;
  summary?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`panel collapsible-panel ${open ? "open" : ""}`}>
      <button type="button" className="collapsible-title" onClick={() => setOpen((current) => !current)}>
        <span className="collapsible-heading">
          {icon}
          <span>{title}</span>
        </span>
        <span className="collapsible-summary">{summary ?? ""}</span>
        <span className="collapse-chevron">{open ? "−" : "+"}</span>
      </button>
      {open ? <div className="collapsible-body">{children}</div> : null}
    </section>
  );
}

function PlotToggleSection({
  title,
  summary,
  open,
  onToggle,
  children
}: {
  title: string;
  summary?: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <section className={`plot-toggle-section ${open ? "open" : "collapsed"}`}>
      <button type="button" className="plot-toggle-title" onClick={onToggle}>
        <span>{title}</span>
        <span className="plot-toggle-summary">{summary ?? ""}</span>
        <span className="collapse-chevron">{open ? "-" : "+"}</span>
      </button>
      {open ? <div className="plot-toggle-body">{children}</div> : null}
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

function NumberField({
  label,
  value,
  step,
  displayDigits,
  commitOnChange = false,
  commitOnSpinChange = false,
  commitOnBlur = true,
  onChange
}: {
  label: string;
  value: number;
  step?: number;
  displayDigits?: number;
  commitOnChange?: boolean;
  commitOnSpinChange?: boolean;
  commitOnBlur?: boolean;
  onChange: (value: number) => void;
}) {
  const displayValue = Number.isFinite(value)
    ? displayDigits === undefined
      ? value
      : value.toFixed(displayDigits)
    : displayDigits === undefined
      ? 0
      : (0).toFixed(displayDigits);
  const useDraftCommit = displayDigits !== undefined;
  const [draftValue, setDraftValue] = useState(String(displayValue));
  const [isEditing, setIsEditing] = useState(false);
  const spinCommitActive = useRef(false);
  const spinCommitTimer = useRef<number | null>(null);
  const skipNextBlur = useRef(false);

  useEffect(() => {
    if (!isEditing) {
      setDraftValue(String(displayValue));
    }
  }, [displayValue, isEditing]);

  useEffect(() => () => {
    if (spinCommitTimer.current !== null) {
      window.clearTimeout(spinCommitTimer.current);
    }
  }, []);

  const markSpinCommit = () => {
    if (!commitOnSpinChange) return;
    spinCommitActive.current = true;
    if (spinCommitTimer.current !== null) {
      window.clearTimeout(spinCommitTimer.current);
    }
    spinCommitTimer.current = window.setTimeout(() => {
      spinCommitActive.current = false;
    }, 350);
  };

  const commitNumericValue = (rawValue: string) => {
    const parsedValue = Number(rawValue);
    if (Number.isFinite(parsedValue)) {
      onChange(parsedValue);
      return true;
    }
    return false;
  };

  const commitDraftValue = () => {
    if (!commitNumericValue(draftValue)) {
      setDraftValue(String(displayValue));
    }
    setIsEditing(false);
  };

  const cancelDraftValue = () => {
    setDraftValue(String(displayValue));
    setIsEditing(false);
  };

  return (
    <label className="field">
      <span>{label}</span>
      {useDraftCommit ? (
        <input
          type="number"
          step={step}
          value={isEditing ? draftValue : displayValue}
          onFocus={() => {
            setDraftValue(String(displayValue));
            setIsEditing(true);
          }}
          onChange={(event) => {
            const nextValue = event.target.value;
            setDraftValue(nextValue);
            const shouldCommit = (commitOnChange || (commitOnSpinChange && spinCommitActive.current))
              && nextValue !== ""
              && nextValue !== "-"
              && !nextValue.endsWith(".");
            if (shouldCommit) {
              commitNumericValue(nextValue);
            }
          }}
          onPointerDown={(event) => {
            const rect = event.currentTarget.getBoundingClientRect();
            if (event.clientX >= rect.right - 30) {
              markSpinCommit();
            }
          }}
          onBlur={() => {
            if (skipNextBlur.current) {
              skipNextBlur.current = false;
              return;
            }
            if (commitOnBlur) {
              commitDraftValue();
            } else {
              cancelDraftValue();
            }
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              skipNextBlur.current = true;
              commitDraftValue();
              event.currentTarget.blur();
            }
            if (event.key === "ArrowUp" || event.key === "ArrowDown") {
              markSpinCommit();
            }
            if (event.key === "Escape") {
              cancelDraftValue();
              event.currentTarget.blur();
            }
          }}
        />
      ) : (
        <input
          type="number"
          step={step}
          value={displayValue}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      )}
    </label>
  );
}

function FrequencyField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  const displayValue = formatFrequencyInput(value);
  const [draftValue, setDraftValue] = useState(displayValue);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    if (!isEditing) {
      setDraftValue(displayValue);
    }
  }, [displayValue, isEditing]);

  const commitValue = () => {
    const parsed = parseFrequencyText(draftValue);
    if (Number.isFinite(parsed) && parsed > 0) {
      onChange(parsed);
      setDraftValue(formatFrequencyInput(parsed));
    } else {
      setDraftValue(displayValue);
    }
    setIsEditing(false);
  };

  const cancelValue = () => {
    setDraftValue(displayValue);
    setIsEditing(false);
  };

  return (
    <label className="field">
      <span>{label}</span>
      <input
        value={draftValue}
        inputMode="decimal"
        onFocus={() => {
          setDraftValue(displayValue);
          setIsEditing(true);
        }}
        onChange={(event) => setDraftValue(event.target.value)}
        onBlur={commitValue}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            commitValue();
            event.currentTarget.blur();
          }
          if (event.key === "Escape") {
            cancelValue();
            event.currentTarget.blur();
          }
        }}
      />
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

function SelectField({
  label,
  value,
  options,
  onChange
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option key={option} value={option}>{formatSelectOption(option)}</option>)}
      </select>
    </label>
  );
}

function ScopeAcquisitionToggle({
  running,
  busy,
  onChange
}: {
  running: boolean;
  busy: boolean;
  onChange: (running: boolean) => void;
}) {
  return (
    <div className="scope-acquisition-toggle" aria-label="Scope acquisition control">
      <span>Live Scope</span>
      <button
        type="button"
        className={running ? "active" : ""}
        onClick={() => onChange(true)}
        disabled={busy}
      >
        Run
      </button>
      <button
        type="button"
        className={!running ? "active" : ""}
        onClick={() => onChange(false)}
        disabled={busy}
      >
        Stop
      </button>
    </div>
  );
}

function formatSelectOption(option: string) {
  const labels: Record<string, string> = {
    square: "Square",
    pulse: "Pulse",
    dc: "DC",
    sine: "Sine",
    VPP: "Vpp",
    VRMS: "Vrms",
    DBM: "dBm"
  };
  return labels[option] ?? option;
}

function CheckboxField({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="check-field">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function ScopeMeasurementTable({ capture }: { capture: ScopeCaptureReadback | null }) {
  const rows = capture?.measurement_values ?? [];
  if (!rows.length) return null;
  return (
    <div className="mini-table-wrap">
      <table className="mini-table">
        <thead>
          <tr>
            <th>Channel</th>
            <th>Measurement</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.source}-${row.measurement}-${index}`}>
              <td>{row.source}</td>
              <td>{row.measurement}</td>
              <td>{row.ok ? formatMaybeNumber(row.value) : row.error ?? "--"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BodeMarginReadout({ sweep }: { sweep: BodeSweepReadback | null }) {
  const margins = sweep?.margins;
  return (
    <div className="bode-margin-grid">
      <div>
        <span>Phase Margin</span>
        <strong>{formatMaybeNumber(margins?.phase_margin_deg, 2)} deg</strong>
      </div>
      <div>
        <span>Gain Crossover</span>
        <strong>{formatMaybeFrequency(margins?.phase_crossover_hz)}</strong>
      </div>
      <div>
        <span>Gain Margin</span>
        <strong>{formatMaybeNumber(margins?.gain_margin_db, 2)} dB</strong>
      </div>
    </div>
  );
}

function FunctionGeneratorState({ readback }: { readback: FunctionGeneratorReadback | null }) {
  const high = readback?.high_v;
  const low = readback?.low_v;
  const hasLevels = typeof high === "number" && typeof low === "number";
  return (
    <div className="fg-state-grid">
      <ReadbackTile label="Waveform" value={formatFunctionGeneratorFunction(readback?.function)} />
      <ReadbackTile label="Frequency" value={formatFrequencyValue(readback?.frequency_hz)} />
      <ReadbackTile label="Voltage Unit" value={formatVoltageUnit(readback?.voltage_unit)} />
      <ReadbackTile label="Low Level" value={formatVoltageValue(low)} />
      <ReadbackTile label="High Level" value={formatVoltageValue(high)} />
      <ReadbackTile label="Output State" value={formatOutputState(readback?.output)} emphasis={isOutputEnabled(readback?.output) ? "good" : "neutral"} />
      {hasLevels ? <ReadbackTile label="Vpp" value={formatVoltageValue(high - low)} /> : null}
      {hasLevels ? <ReadbackTile label="Offset" value={formatVoltageValue((high + low) / 2)} /> : null}
      <ReadbackTile label="Instrument Error" value={instrumentErrorText(readback) ?? "No error"} emphasis={instrumentErrorText(readback) ? "bad" : "good"} />
    </div>
  );
}

function ReadbackTile({
  label,
  value,
  emphasis = "neutral"
}: {
  label: string;
  value: string;
  emphasis?: "neutral" | "good" | "bad";
}) {
  return (
    <div className={`readback-tile ${emphasis}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function InstrumentReadbackGrid({ items, className = "" }: { items: Record<string, unknown>; className?: string }) {
  return (
    <dl className={`compact-kv readback-grid ${className}`}>
      {Object.entries(items).map(([key, value]) => (
        <React.Fragment key={key}>
          <dt>{key}</dt>
          <dd>{formatMaybeValue(value)}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

function formatMaybeValue(value: unknown) {
  if (typeof value === "number") return Number.isFinite(value) ? value.toPrecision(6) : "--";
  if (value === null || value === undefined || value === "") return "--";
  return String(value);
}

function formatFunctionGeneratorFunction(value: unknown) {
  if (value === null || value === undefined || value === "") return "--";
  const normalized = String(value).trim().replace(/^"|"$/g, "").toUpperCase();
  const labels: Record<string, string> = {
    SQU: "Square",
    SQUARE: "Square",
    SIN: "Sine",
    SINE: "Sine",
    PULS: "Pulse",
    PULSE: "Pulse",
    DC: "DC"
  };
  return labels[normalized] ?? String(value);
}

function formatFrequencyValue(value: unknown) {
  const numeric = coerceNumber(value);
  if (numeric === null) return "--";
  const abs = Math.abs(numeric);
  if (abs >= 1_000_000) return `${trimFixed(numeric / 1_000_000, 3)} MHz`;
  if (abs >= 1_000) return `${trimFixed(numeric / 1_000, 3)} kHz`;
  return `${trimFixed(numeric, 3)} Hz`;
}

function formatVoltageValue(value: unknown) {
  const numeric = coerceNumber(value);
  if (numeric === null) return "--";
  return `${trimFixed(numeric, 3)} V`;
}

function formatVoltageUnit(value: unknown) {
  if (value === null || value === undefined || value === "") return "--";
  const normalized = String(value).trim().replace(/^"|"$/g, "").toUpperCase();
  const labels: Record<string, string> = {
    VPP: "Vpp",
    VRMS: "Vrms",
    DBM: "dBm"
  };
  return labels[normalized] ?? String(value);
}

function formatOutputState(value: unknown) {
  if (value === null || value === undefined || value === "") return "--";
  const normalized = String(value).trim().replace(/^"|"$/g, "").toUpperCase();
  if (normalized === "1" || normalized === "ON") return "ON";
  if (normalized === "0" || normalized === "OFF") return "OFF";
  return String(value);
}

function isOutputEnabled(value: unknown) {
  const normalized = String(value ?? "").trim().replace(/^"|"$/g, "").toUpperCase();
  return normalized === "1" || normalized === "ON";
}

function instrumentErrorText(readback: FunctionGeneratorReadback | null | undefined) {
  const raw = readback?.error ?? readback?.system_error;
  if (raw === null || raw === undefined || raw === "") return null;
  const text = String(raw).trim();
  const normalized = text.toLowerCase();
  if (normalized === "0" || normalized === "+0" || normalized.includes("no error")) return null;
  return text;
}

function coerceNumber(value: unknown) {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function trimFixed(value: number, digits: number) {
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

function formatMaybeNumber(value: number | null | undefined, fixedDigits?: number) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  return fixedDigits === undefined ? value.toPrecision(6) : value.toFixed(fixedDigits);
}

function formatMaybeFrequency(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(3)} MHz`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(3)} kHz`;
  return `${value.toFixed(2)} Hz`;
}

function LiveRefreshButton({ enabled, label, onChange }: { enabled: boolean; label: string; onChange: (enabled: boolean) => void }) {
  return (
    <div className="live-refresh-row">
      <span>{label}</span>
      <div className="live-toggle" role="group" aria-label="Live readback">
        <button
          type="button"
          className={enabled ? "active on" : ""}
          aria-pressed={enabled}
          onClick={() => onChange(true)}
        >
          ON
        </button>
        <button
          type="button"
          className={!enabled ? "active off" : ""}
          aria-pressed={!enabled}
          onClick={() => onChange(false)}
        >
          OFF
        </button>
      </div>
    </div>
  );
}

function LiveValue({
  label,
  value,
  unit,
  digits = 4
}: {
  label: string;
  value: number | undefined;
  unit: string;
  digits?: number;
}) {
  return (
    <div className="live-value">
      <span>{label}</span>
      <strong>{value === undefined ? "--" : value.toFixed(digits)}{unit ? ` ${unit}` : ""}</strong>
    </div>
  );
}

function RateValue({
  value,
  min,
  max,
  onChange
}: {
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="live-value rate-value">
      <span>Rate</span>
      <div className="rate-input-row">
        <input
          type="number"
          min={min}
          max={max}
          step={1}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          onBlur={(event) => onChange(Number(event.target.value))}
        />
        <strong>Hz</strong>
      </div>
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

function InductanceReadout({ field }: { field: InductanceField | null }) {
  if (!field) return <p className="readback">No inductance register readback yet.</p>;
  return (
    <dl className="kv compact-kv">
      <dt>Value</dt>
      <dd>{field.value_nh === null ? "--" : `${field.value_nh.toFixed(3)} nH`}</dd>
      <dt>Raw</dt>
      <dd>{field.raw_hex}</dd>
      <dt>Register</dt>
      <dd>{field.memory_address} {field.bitfield}</dd>
    </dl>
  );
}

function XdpPidTable({ pid }: { pid: XdpPidReadback | null }) {
  const fields = pid?.pid_registers?.fields;
  if (!fields) return <p className="readback">No mod0 PID field readback yet.</p>;
  return (
    <div className="table-wrap compact-table">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Range</th>
            <th>Decimal</th>
            <th>Hex</th>
            <th>Limits</th>
          </tr>
        </thead>
        <tbody>
          {xdpPidFieldNames.map((name) => {
            const field = fields[name];
            return (
              <tr key={name}>
                <td>{name}</td>
                <td>{field?.bitfield ?? "--"}</td>
                <td>{field?.raw ?? "--"}</td>
                <td>{field?.raw_hex ?? "--"}</td>
                <td>{field ? `${field.min}-${field.max}` : "--"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
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

function bodeSweepOption(sweep: BodeSweepReadback | null, axis: ChartAxisSettings = defaultBodeAxisSettings) {
  const darkMode = typeof document !== "undefined" && document.body.classList.contains("theme-dark");
  const gridLineColor = darkMode ? "#303030" : "#d9dee7";
  const axisLineColor = darkMode ? "#3a3a3a" : "#9aa0a6";
  const axisTextColor = darkMode ? "#8f8f8f" : "#5f6368";
  const frequencies = sweep?.frequency_hz ?? [];
  const magnitude = sweep?.magnitude_db ?? [];
  const phase = sweep?.phase_deg ?? [];
  const firstFrequency = frequencies.length > 0 ? frequencies[0] : defaultBodeSweepConfig.start_hz;
  const lastFrequency = frequencies.length > 0 ? frequencies[frequencies.length - 1] : defaultBodeSweepConfig.stop_hz;
  const xMin = axis.xMin > 0 ? axis.xMin : firstFrequency;
  const xMax = axis.xMax > xMin ? axis.xMax : lastFrequency;
  return {
    animation: false,
    tooltip: {
      trigger: "axis",
      valueFormatter: (value: number) => Number.isFinite(value) ? value.toFixed(3) : `${value}`
    },
    legend: {
      top: 0,
      itemWidth: 24,
      itemHeight: 14,
      textStyle: { color: axisTextColor }
    },
    grid: { left: 56, right: 58, top: 36, bottom: 46 },
    xAxis: {
      type: "log",
      name: "Hz",
      logBase: 10,
      min: xMin,
      max: xMax,
      nameTextStyle: { color: axisTextColor },
      axisLabel: {
        color: axisTextColor,
        formatter: (value: number) => formatFrequencyTick(value)
      },
      axisLine: { lineStyle: { color: axisLineColor } },
      axisTick: { lineStyle: { color: axisLineColor } },
      splitLine: { lineStyle: { color: gridLineColor, width: 1 } }
    },
    yAxis: [
      {
        type: "value",
        name: "Gain (dB)",
        min: axis.yMin,
        max: axis.yMax,
        nameTextStyle: { color: axisTextColor },
        axisLabel: { color: axisTextColor },
        axisLine: { lineStyle: { color: axisLineColor } },
        axisTick: { lineStyle: { color: axisLineColor } },
        splitLine: { lineStyle: { color: gridLineColor, width: 1 } }
      },
      {
        type: "value",
        name: "Phase (deg)",
        min: axis.y2Min,
        max: axis.y2Max,
        nameTextStyle: { color: axisTextColor },
        axisLabel: { color: axisTextColor },
        axisLine: { lineStyle: { color: axisLineColor } },
        axisTick: { lineStyle: { color: axisLineColor } },
        splitLine: { show: false }
      }
    ],
    series: [
      {
        name: "Gain",
        type: "line",
        symbol: "circle",
        showSymbol: false,
        data: frequencies.map((frequency, index) => [frequency, magnitude[index] ?? null]),
        itemStyle: { color: "#ea4335" },
        lineStyle: { width: 2, color: "#ea4335" }
      },
      {
        name: "Phase",
        type: "line",
        yAxisIndex: 1,
        symbol: "circle",
        showSymbol: false,
        data: frequencies.map((frequency, index) => [frequency, phase[index] ?? null]),
        itemStyle: { color: "#1a73e8" },
        lineStyle: { width: 2, color: "#1a73e8" }
      }
    ]
  };
}

function scopeWaveformOption(capture: ScopeCaptureReadback | null, axis: ScopeAxisSettings = defaultScopeAxisSettings) {
  const darkMode = typeof document !== "undefined" && document.body.classList.contains("theme-dark");
  const gridLineColor = darkMode ? "#303030" : "#d9dee7";
  const axisLineColor = darkMode ? "#3a3a3a" : "#9aa0a6";
  const axisTextColor = darkMode ? "#8f8f8f" : "#5f6368";
  const leftAxisColor = "#1a73e8";
  const rightAxisColor = "#ea4335";
  const colors = ["#1a73e8", "#ea4335", "#34a853", "#fbbc04", "#8ab4f8", "#f28b82", "#81c995", "#fde293"];
  const waveforms = capture?.waveforms ?? [];
  const timeScale = scopeTimeScale(waveforms);
  const seriesData = waveforms.map((waveform, index) => ({
    name: waveform.source,
    color: colors[index % colors.length],
    axisSide: axis.channelAxes[waveform.source] ?? "left",
    data: scopeSeriesData(waveform, timeScale)
  }));
  return {
    animation: false,
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown) => scopeTooltipFormatter(params, seriesData, timeScale.unit)
    },
    legend: { top: 0, textStyle: { color: axisTextColor } },
    grid: { left: 58, right: 24, top: 34, bottom: 42 },
    xAxis: {
      type: "value",
      name: timeScale.unit,
      min: timeScale.min,
      max: timeScale.max,
      nameTextStyle: { color: axisTextColor },
      axisLabel: {
        color: axisTextColor,
        formatter: (value: number) => formatScopeAxisTick(value, timeScale.unit)
      },
      axisLine: { lineStyle: { color: axisLineColor } },
      axisTick: { lineStyle: { color: axisLineColor } },
      splitLine: { lineStyle: { color: gridLineColor } }
    },
    yAxis: [
      {
        type: "value",
        name: "V",
        min: axis.leftMin,
        max: axis.leftMax,
        nameTextStyle: { color: leftAxisColor },
        axisLabel: { color: leftAxisColor },
        axisLine: { lineStyle: { color: leftAxisColor } },
        axisTick: { lineStyle: { color: leftAxisColor } },
        splitLine: { lineStyle: { color: gridLineColor } }
      },
      {
        type: "value",
        name: "V",
        min: axis.rightMin,
        max: axis.rightMax,
        nameTextStyle: { color: rightAxisColor },
        axisLabel: { color: rightAxisColor },
        axisLine: { lineStyle: { color: rightAxisColor } },
        axisTick: { lineStyle: { color: rightAxisColor } },
        splitLine: { show: false }
      }
    ],
    series: seriesData.map((series) => ({
      name: series.name,
      type: "line",
      yAxisIndex: series.axisSide === "right" ? 1 : 0,
      showSymbol: false,
      symbol: "circle",
      sampling: "lttb",
      progressive: 5000,
      progressiveThreshold: 10000,
      data: series.data,
      itemStyle: { color: series.color },
      lineStyle: { width: 1.5, color: series.color }
    }))
  };
}

type ScopeSeriesDisplay = {
  name: string;
  color: string;
  data: Array<[number, number | null]>;
};

function scopeTooltipFormatter(params: unknown, seriesData: ScopeSeriesDisplay[], unit: string) {
  const items = Array.isArray(params) ? params : [params];
  const first = items[0] as { axisValue?: unknown; data?: unknown } | undefined;
  const dataPoint = Array.isArray(first?.data) ? first.data : null;
  const xValue = Number(first?.axisValue ?? dataPoint?.[0]);
  if (!Number.isFinite(xValue)) return "";
  const rows = seriesData
    .map((series) => {
      const point = nearestScopePoint(series.data, xValue);
      if (!point || point[1] === null || !Number.isFinite(point[1])) return null;
      return `<div style="display:flex;align-items:center;gap:8px;justify-content:space-between;min-width:150px;">
        <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${series.color};margin-right:6px;"></span>${series.name}</span>
        <strong>${point[1].toFixed(3)}</strong>
      </div>`;
    })
    .filter(Boolean)
    .join("");
  return `<div>
    <div style="margin-bottom:6px;">${formatScopeAxisTick(xValue, unit)} ${unit}</div>
    ${rows}
  </div>`;
}

function nearestScopePoint(data: Array<[number, number | null]>, xValue: number) {
  if (data.length === 0) return null;
  let low = 0;
  let high = data.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (data[mid][0] < xValue) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  const left = low > 0 ? data[low - 1] : null;
  const right = data[low] ?? null;
  if (!left) return right;
  if (!right) return left;
  return Math.abs(left[0] - xValue) <= Math.abs(right[0] - xValue) ? left : right;
}

function scopeSeriesData(
  waveform: NonNullable<ScopeCaptureReadback["waveforms"]>[number],
  timeScale: ReturnType<typeof scopeTimeScale>
) {
  const total = Math.min(waveform.x.length, waveform.y.length);
  if (total === 0) return [];
  const targetPoints = Math.min(total, scopeFrontendMaxPoints);
  const data: Array<[number, number | null]> = [];
  let lastIndex = -1;
  for (let idx = 0; idx < targetPoints; idx += 1) {
    const sourceIndex = targetPoints === 1 ? 0 : Math.round(idx * (total - 1) / (targetPoints - 1));
    if (sourceIndex === lastIndex) continue;
    const xValue = waveform.x[sourceIndex];
    data.push([scopeRelativeTime(xValue, timeScale.origin, timeScale.scale), waveform.y[sourceIndex] ?? null]);
    lastIndex = sourceIndex;
  }
  return data;
}

function scopeChartKey(capture: ScopeCaptureReadback | null) {
  const waveforms = capture?.waveforms ?? [];
  const signature = waveforms
    .map((waveform) =>
      `${waveform.source}:${waveform.x.length}:${waveform.time_span_s ?? ""}:${waveform.original_points ?? ""}:${waveform.display_points ?? ""}:${waveform.capture_id ?? ""}`
    )
    .join("|");
  return `${capture?.timestamp ?? "empty"}:${signature}`;
}

function scopeTimeScale(waveforms: NonNullable<ScopeCaptureReadback["waveforms"]>) {
  let minSeconds = Number.POSITIVE_INFINITY;
  let maxSeconds = Number.NEGATIVE_INFINITY;
  let hasTime = false;
  for (const waveform of waveforms) {
    for (const value of waveform.x) {
      if (!Number.isFinite(value)) continue;
      if (value < minSeconds) minSeconds = value;
      if (value > maxSeconds) maxSeconds = value;
      hasTime = true;
    }
  }
  if (!hasTime) {
    minSeconds = 0;
    maxSeconds = 1;
  }
  const spanSeconds = Math.max(0, maxSeconds - minSeconds);
  let unit = "s";
  let scale = 1;
  if (spanSeconds < 1e-3) {
    unit = "us";
    scale = 1e6;
  } else if (spanSeconds < 1) {
    unit = "ms";
    scale = 1e3;
  }
  const max = (maxSeconds - minSeconds) * scale;
  const paddedMax = max > 0 ? max : 1;
  return { unit, scale, origin: minSeconds, min: 0, max: paddedMax };
}

function scopeRelativeTime(value: number, origin: number, scale: number) {
  return (value - origin) * scale;
}

function formatScopeAxisTick(value: number, unit: string) {
  if (unit === "s") return trimTick(value);
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function formatChartTooltipValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(3);
  return `${value}`;
}

function formatFrequencyTick(value: number) {
  if (value >= 1_000_000) return `${trimTick(value / 1_000_000)}M`;
  if (value >= 1_000) return `${trimTick(value / 1_000)}k`;
  return trimTick(value);
}

function trimTick(value: number) {
  if (Math.abs(value - Math.round(value)) < 1e-9) return `${Math.round(value)}`;
  return value.toFixed(2).replace(/\.?0+$/, "");
}

function telemetryOption(history: VoutReadback[], axis: TelemetryAxisSettings = defaultTelemetryAxisSettings) {
  const darkMode = typeof document !== "undefined" && document.body.classList.contains("theme-dark");
  const gridLineColor = darkMode ? "#303030" : "#d9dee7";
  const axisLineColor = darkMode ? "#3a3a3a" : "#9aa0a6";
  const axisTextColor = darkMode ? "#8f8f8f" : "#5f6368";
  const voutColor = "#1a73e8";
  const ioutColor = "#ea4335";
  const commandColor = "#34a853";
  const latestMs = history.length > 0 && history[history.length - 1].timestamp
    ? history[history.length - 1].timestamp * 1000
    : Date.now();
  const durationMs = clampTelemetryDuration(axis.durationSeconds) * 1000;
  const axisMaxMs = latestMs;
  const axisStartMs = axisMaxMs - durationMs;
  const toPoint = (item: VoutReadback, value: number | undefined) => [
    (item.timestamp ?? latestMs / 1000) * 1000,
    value ?? null
  ];
  return {
    animation: false,
    animationDuration: 0,
    tooltip: {
      trigger: "axis",
      valueFormatter: formatChartTooltipValue
    },
    legend: {
      top: 0,
      itemWidth: 24,
      itemHeight: 14,
      textStyle: { color: axisTextColor }
    },
    grid: { left: 56, right: 56, top: 36, bottom: 42 },
    xAxis: {
      type: "value",
      min: axisStartMs,
      max: axisMaxMs,
      interval: telemetryAxisStepMs,
      minInterval: telemetryAxisStepMs,
      maxInterval: telemetryAxisStepMs,
      axisLabel: {
        color: axisTextColor,
        formatter: (value: number) => new Date(value).toLocaleTimeString()
      },
      axisLine: { lineStyle: { color: axisLineColor } },
      axisTick: { lineStyle: { color: axisLineColor } },
      splitLine: { lineStyle: { color: gridLineColor, width: 1 } }
    },
    yAxis: [
      {
        type: "value",
        name: "Vout (V)",
        min: axis.yMin,
        max: axis.yMax,
        nameTextStyle: { color: voutColor },
        axisLabel: { color: voutColor },
        axisLine: { lineStyle: { color: voutColor } },
        axisTick: { lineStyle: { color: voutColor } },
        splitLine: { lineStyle: { color: gridLineColor, width: 1 } }
      },
      {
        type: "value",
        name: "Iout (A)",
        min: axis.y2Min,
        max: axis.y2Max,
        nameTextStyle: { color: ioutColor },
        axisLabel: { color: ioutColor },
        axisLine: { lineStyle: { color: ioutColor } },
        axisTick: { lineStyle: { color: ioutColor } },
        splitLine: { show: false }
      }
    ],
    series: [
      {
        name: "Vout",
        type: "line",
        smooth: true,
        symbol: "circle",
        showSymbol: false,
        connectNulls: true,
        data: history.map((item) => toPoint(item, item.read_vout_v)),
        itemStyle: { color: voutColor },
        lineStyle: { width: 2, color: voutColor }
      },
      {
        name: "Vout_Command",
        type: "line",
        smooth: true,
        symbol: "circle",
        showSymbol: false,
        connectNulls: true,
        data: history.map((item) => toPoint(item, item.vout_command_v)),
        itemStyle: { color: commandColor },
        lineStyle: { width: 1, color: commandColor, type: "dashed" }
      },
      {
        name: "Iout",
        type: "line",
        yAxisIndex: 1,
        smooth: true,
        symbol: "circle",
        showSymbol: false,
        connectNulls: true,
        data: history.map((item) => toPoint(item, item.read_iout_a)),
        itemStyle: { color: ioutColor },
        lineStyle: { width: 2, color: ioutColor }
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

function formatPmbusOutputState(value: PmbusOutputReadback | null) {
  if (!value) return "not read";
  if (!value.ok) return "error";
  const rawText = value.operation_after ?? value.operation ?? value.operation_written;
  if (!rawText) return "Unknown";
  const raw = Number.parseInt(rawText, 16);
  if (!Number.isFinite(raw)) return "Unknown";
  if ((raw & 0x80) !== 0) return "On";
  if (raw === 0) return "Off";
  return `0x${(raw & 0xff).toString(16).toUpperCase().padStart(2, "0")}`;
}

function formatXdpOutputState(value: XdpOutputReadback | null) {
  if (!value) return "not read";
  if (!value.ok) return "error";
  const state = value.state ?? value.readback?.state;
  if (state === "high") return "High";
  if (state === "low") return "Low";
  if (state === "release") return "Release";
  return state ?? "Unknown";
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

