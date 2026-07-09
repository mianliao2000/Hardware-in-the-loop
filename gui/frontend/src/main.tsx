import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactECharts from "echarts-for-react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Gauge,
  HelpCircle,
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
  Maximize2,
  MessageCircle,
  Minimize2,
  Send,
  XCircle,
  X,
  Sun,
  Zap
} from "lucide-react";
import { archiveCurrentTuningRun, captureScope, deleteTuningRun, getTuningRuns, getTuningStatus, loadTuningRun, openTuningAnimationGif, pauseTuning, readFunctionGenerator, readInductance, readPmbusOutput, readPowerSupply, readVout, readXdpOutput, readXdpPid, resetTuning, resumeTuning, runBodeSweep, runSelfTestDevice, saveTuningAnimationGif, sendLlmChat, setFunctionGenerator, setInductance, setPmbusOutput, setPowerSupply, setScopeAcquisition, setVout, setXdpOutput, setXdpPid, startTuning, stepTuning, stopTuning, warmScope } from "./api";
import type { AutotuneExperimentConfig, AutotuneGifResponse, AutotuneRunsResponse, BodeSweepConfig, BodeSweepReadback, FunctionGeneratorReadback, InductanceField, InductanceReadback, InstrumentKey, InstrumentTestResult, IterationRecord, LlmChatMessage, PmbusOutputAction, PmbusOutputReadback, PowerSupplyReadback, ScopeCaptureReadback, SearchParameter, SelfTestResponse, TuningConfig, TuningStatus, VoutReadback, XdpOutputAction, XdpOutputReadback, XdpPidReadback } from "./types";
import "./styles.css";

const searchParameter = (center: number, min: number, max: number, points: number): SearchParameter => ({
  center,
  min,
  max,
  points,
  step: points > 1 ? (max - min) / (points - 1) : max - min
});

const defaultConfig: TuningConfig = {
  plant: {
    vdc: 12,
    inductance_h: 30e-6,
    capacitance_f: 15e-6,
    capacitor_esr_ohm: 7.5e-3,
    inductor_dcr_ohm: 50e-3
  },
  targets: {
    vout_target_v: 0.9296875,
    overshoot_pct: 3,
    undershoot_pct: 3,
    settling_time_s: 1e-6,
    oscillations: 0,
    phase_margin_deg: 45,
    crossover_frequency_hz: 200000,
    gain_margin_db: 6,
    phase_margin_tolerance_deg: 5,
    crossover_tolerance_pct: 20
  },
  search: {
    wc_min_rad_s: 94248,
    wc_max_rad_s: 314159,
    phi_min_deg: 30,
    phi_max_deg: 80,
    initial_wc_rad_s: 157080,
    initial_phi_deg: 60,
    max_iterations: 40,
    max_coarse_iterations: 20,
    max_refined_iterations: 20,
    mod0_kp: searchParameter(165, 100, 255, 9),
    mod0_ki: searchParameter(220, 150, 255, 9),
    mod0_kd: searchParameter(175, 100, 200, 9),
    mod0_kpole1: searchParameter(3, 3, 6, 2),
    mod0_kpole2: searchParameter(3, 3, 6, 2),
    mod0_cm_gain: searchParameter(2, 2, 2, 1),
    output_inductance_nh: searchParameter(100.024, 80.019, 120.029, 5),
    effective_lc_inductance_nh: searchParameter(369.276, 295.421, 443.131, 5)
  }
};

const cloneDefaultConfig = (): TuningConfig => JSON.parse(JSON.stringify(defaultConfig)) as TuningConfig;

const normalizeSearchParameter = (value: Partial<SearchParameter> | undefined, fallback: SearchParameter): SearchParameter => {
  const next = {
    ...fallback,
    ...(value ?? {})
  };
  const points = Math.max(1, Math.round(Number.isFinite(next.points) ? next.points : fallback.points));
  const min = Number.isFinite(next.min) ? next.min : fallback.min;
  const max = Number.isFinite(next.max) ? next.max : fallback.max;
  const center = Number.isFinite(next.center) ? Math.min(Math.max(next.center, Math.min(min, max)), Math.max(min, max)) : fallback.center;
  return {
    ...next,
    center,
    min,
    max,
    points,
    step: points > 1 ? Math.abs((max - min) / (points - 1)) : Math.abs(max - min)
  };
};

const normalizeTuningConfig = (config?: Partial<TuningConfig> | null): TuningConfig => {
  const fallback = cloneDefaultConfig();
  const incoming = config ?? {};
  const incomingSearch = incoming.search ?? {};
  const normalizedSearch = {
    ...fallback.search,
    ...incomingSearch
  };
  for (const field of hardwareSearchFields) {
    normalizedSearch[field.key] = normalizeSearchParameter(
      incomingSearch[field.key] as Partial<SearchParameter> | undefined,
      fallback.search[field.key]
    );
  }
  return {
    ...fallback,
    ...incoming,
    plant: {
      ...fallback.plant,
      ...(incoming.plant ?? {})
    },
    targets: {
      ...fallback.targets,
      ...(incoming.targets ?? {})
    },
    search: normalizedSearch
  };
};

const selfTestOrder: InstrumentKey[] = ["afg", "bode", "power_supply", "scope", "board_i2c"];
const xdpPidFieldNames = ["mod0_kp", "mod0_ki", "mod0_kd", "mod0_kpole1", "mod0_kpole2", "mod0_cm_gain"] as const;
const defaultXdpPidLimits: Record<(typeof xdpPidFieldNames)[number], { min: number; max: number }> = {
  mod0_kp: { min: 0, max: 255 },
  mod0_ki: { min: 0, max: 255 },
  mod0_kd: { min: 0, max: 255 },
  mod0_kpole1: { min: 0, max: 15 },
  mod0_kpole2: { min: 0, max: 15 },
  mod0_cm_gain: { min: 0, max: 127 }
};
type HardwareSearchKey =
  | "mod0_kp"
  | "mod0_ki"
  | "mod0_kd"
  | "mod0_kpole1"
  | "mod0_kpole2"
  | "mod0_cm_gain"
  | "output_inductance_nh"
  | "effective_lc_inductance_nh";
const hardwareSearchFields: Array<{ key: HardwareSearchKey; label: string; limit: string }> = [
  { key: "mod0_kp", label: "mod0_kp", limit: "0-255" },
  { key: "mod0_ki", label: "mod0_ki", limit: "0-255" },
  { key: "mod0_kd", label: "mod0_kd", limit: "0-255" },
  { key: "mod0_kpole1", label: "mod0_kpole1", limit: "0-15" },
  { key: "mod0_kpole2", label: "mod0_kpole2", limit: "0-15" },
  { key: "mod0_cm_gain", label: "Current Mode Gain", limit: "0-127" },
  { key: "output_inductance_nh", label: "Output Inductance (nH)", limit: "14.29-117028.57 nH" },
  { key: "effective_lc_inductance_nh", label: "Effective Lc Inductance (nH)", limit: "229.02-117028.57 nH" }
];
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
  mod0_kpole2: 3,
  mod0_cm_gain: 2
};
const defaultBodeSweepConfig: BodeSweepConfig = {
  host: "127.0.0.1",
  port: 5025,
  start_hz: 1000,
  stop_hz: 1_000_000,
  points: 201,
  bandwidth_hz: 300,
  source_vpp: 0.1,
  timeout_ms: 60000
};
const defaultScopeChannels = ["CH1", "CH3"];
const defaultScopeMeasurements: string[] = [];
const scopeFrontendMaxPoints = 200_000;
const defaultFunctionGeneratorConfig = {
  channel: 1,
  mode: "square",
  voltage_unit: "VPP",
  frequency_hz: 10000,
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
type ManualExperimentSettings = {
  fgConfig: typeof defaultFunctionGeneratorConfig;
  scopeChannels: string[];
  scopeMeasurements: string[];
  scopeAxisSettings: ScopeAxisSettings;
};
const defaultManualExperimentSettings = (): ManualExperimentSettings => ({
  fgConfig: { ...defaultFunctionGeneratorConfig },
  scopeChannels: [...defaultScopeChannels],
  scopeMeasurements: [...defaultScopeMeasurements],
  scopeAxisSettings: {
    ...defaultScopeAxisSettings,
    channelAxes: { ...defaultScopeAxisSettings.channelAxes }
  }
});

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
  const field = xdpPid?.pid_registers?.fields[name] ?? xdpPid?.current_mode_registers?.fields[name];
  const limits = field ? { min: field.min, max: field.max } : defaultXdpPidLimits[name];
  const label = name === "mod0_cm_gain" ? "Current Mode Gain" : name;
  return `${label} (${limits.min}-${limits.max})`;
}

function searchParameterWithCenter(parameter: SearchParameter, center: number): SearchParameter {
  const safeCenter = Number.isFinite(center) ? center : parameter.center;
  return {
    ...parameter,
    center: safeCenter,
    min: Math.min(parameter.min, safeCenter),
    max: Math.max(parameter.max, safeCenter)
  };
}

function withManualSearchCenters(
  config: TuningConfig,
  xdpPidRequest: Record<string, number>,
  inductanceRequest: { output_nh: number; effective_lc_nh: number }
): TuningConfig {
  return {
    ...config,
    search: {
      ...config.search,
      mod0_kp: searchParameterWithCenter(config.search.mod0_kp, Math.round(xdpPidRequest.mod0_kp ?? defaultManualXdpPidRequest.mod0_kp)),
      mod0_ki: searchParameterWithCenter(config.search.mod0_ki, Math.round(xdpPidRequest.mod0_ki ?? defaultManualXdpPidRequest.mod0_ki)),
      mod0_kd: searchParameterWithCenter(config.search.mod0_kd, Math.round(xdpPidRequest.mod0_kd ?? defaultManualXdpPidRequest.mod0_kd)),
      mod0_kpole1: searchParameterWithCenter(config.search.mod0_kpole1, Math.round(xdpPidRequest.mod0_kpole1 ?? defaultManualXdpPidRequest.mod0_kpole1)),
      mod0_kpole2: searchParameterWithCenter(config.search.mod0_kpole2, Math.round(xdpPidRequest.mod0_kpole2 ?? defaultManualXdpPidRequest.mod0_kpole2)),
      mod0_cm_gain: searchParameterWithCenter(config.search.mod0_cm_gain, Math.round(xdpPidRequest.mod0_cm_gain ?? defaultManualXdpPidRequest.mod0_cm_gain)),
      output_inductance_nh: searchParameterWithCenter(config.search.output_inductance_nh, inductanceRequest.output_nh),
      effective_lc_inductance_nh: searchParameterWithCenter(config.search.effective_lc_inductance_nh, inductanceRequest.effective_lc_nh)
    }
  };
}

function buildAutotuneExperiment(
  voutRequest: { address: string; page: number; adapter: string; voltage: number },
  bodeSweepConfig: BodeSweepConfig,
  settings: ManualExperimentSettings,
  analysisSelection: { bode: boolean; transient: boolean }
): AutotuneExperimentConfig {
  return {
    board_address: voutRequest.address,
    board_page: voutRequest.page,
    board_adapter: voutRequest.adapter,
    response_channel: "CH3",
    enable_bode_analysis: analysisSelection.bode,
    enable_transient_analysis: analysisSelection.transient,
    bode_config: { ...bodeSweepConfig },
    function_generator_config: { ...settings.fgConfig },
    scope_config: {
      channels: [...settings.scopeChannels],
      measurements: [...settings.scopeMeasurements],
      scope_axis_settings: {
        ...settings.scopeAxisSettings,
        channelAxes: { ...settings.scopeAxisSettings.channelAxes }
      }
    },
    vout_tolerance_v: 0.15,
    response_abs_limit_v: 0.25,
    ignore_pass_until_max_iterations: true
  };
}

function App() {
  const [config, setConfig] = useState<TuningConfig>(() => cloneDefaultConfig());
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
  const [autotuneAnalysisSelection, setAutotuneAnalysisSelection] = useState({ transient: true, bode: true });
  const [autotuneRuns, setAutotuneRuns] = useState<AutotuneRunsResponse | null>(null);
  const [selectedAutotuneRun, setSelectedAutotuneRun] = useState("");
  const [autotuneArchiveStatus, setAutotuneArchiveStatus] = useState("");
  const [autotuneGif, setAutotuneGif] = useState<AutotuneGifResponse | null>(null);
  const [autotuneGifFrameDurationS, setAutotuneGifFrameDurationS] = useState("0.1");
  const viewingLoadedRun = useRef(false);
  const [manualExperimentSettings, setManualExperimentSettings] = useState<ManualExperimentSettings>(defaultManualExperimentSettings);
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
    if (viewingLoadedRun.current) return;
    try {
      const next = await getTuningStatus();
      if (viewingLoadedRun.current) return;
      setStatus(next);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const refreshAutotuneRuns = async (preferredSelection?: string) => {
    try {
      const next = await getTuningRuns();
      setAutotuneRuns(next);
      if (preferredSelection) {
        setSelectedAutotuneRun(preferredSelection);
      } else if (!selectedAutotuneRun) {
        const first = next.recent[0] ?? next.saved[0];
        if (first) setSelectedAutotuneRun(`${first.kind}:${first.run_id}`);
      }
    } catch (exc) {
      setError(String(exc));
    }
  };

  useEffect(() => {
    refresh();
    refreshAutotuneRuns();
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

  const loadAutotuneHardwareBaseline = async () => {
    let nextVout = vout;
    let nextPid = xdpPid;
    let nextInductance = inductance;
    let pidRequest = { ...xdpPidRequest };
    let inductanceValues = { ...inductanceRequest };

    if (!nextVout?.ok) {
      nextVout = await readVout(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setVoutState(nextVout);
      if (nextVout.ok && nextVout.read_vout_v !== undefined) {
        setTelemetryHistory((current) => appendTelemetrySample(current, nextVout as VoutReadback));
      }
    }

    const pidFieldsFromState = nextPid?.pid_registers?.fields;
    const currentModeFieldsFromState = nextPid?.current_mode_registers?.fields;
    if (!pidFieldsFromState || !currentModeFieldsFromState) {
      nextPid = await readXdpPid(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setXdpPidState(nextPid);
    }
    const pidFields = nextPid?.pid_registers?.fields;
    const currentModeFields = nextPid?.current_mode_registers?.fields;
    if (!pidFields) {
      throw new Error("Auto-Tune safety check failed: could not read current XDP PID fields before writing.");
    }
    pidRequest = {
      ...pidRequest,
      ...Object.fromEntries(xdpPidFieldNames.map((name) => [name, (pidFields[name] ?? currentModeFields?.[name])?.raw ?? pidRequest[name]]))
    };
    setXdpPidRequest(pidRequest);

    if (!nextInductance?.ok) {
      nextInductance = await readInductance(voutRequest.address, voutRequest.page, voutRequest.adapter);
      setInductanceState(nextInductance);
    }
    const outputNh = nextInductance?.output_inductance?.value_nh;
    const effectiveNh = nextInductance?.effective_lc_inductance?.value_nh;
    if (!nextInductance?.ok) {
      throw new Error(`Auto-Tune safety check failed: could not read current inductance fields before writing. ${nextInductance?.error ?? ""}`);
    }
    inductanceValues = {
      output_nh: typeof outputNh === "number" && Number.isFinite(outputNh) ? outputNh : inductanceRequest.output_nh,
      effective_lc_nh: typeof effectiveNh === "number" && Number.isFinite(effectiveNh) ? effectiveNh : inductanceRequest.effective_lc_nh
    };
    setInductanceRequest(inductanceValues);

    return { voutReadback: nextVout, pidRequest, inductanceValues };
  };

  const runAction = async (action: "start" | "pause" | "resume" | "stop" | "step") => {
    try {
      if (action === "pause" || action === "resume" || action === "stop") {
        const resumeRun = action === "resume" && viewingLoadedRun.current ? status?.run : undefined;
        viewingLoadedRun.current = false;
        const next = action === "pause"
          ? await pauseTuning()
          : action === "resume"
            ? await resumeTuning(resumeRun?.run_id, resumeRun?.kind)
            : await stopTuning();
        setStatus(next);
        setConfig(normalizeTuningConfig(next.config));
        refreshAutotuneRuns();
        setError("");
        return;
      }
      if (!autotuneAnalysisSelection.transient && !autotuneAnalysisSelection.bode) {
        throw new Error("Select Transient Analysis, Bode Analysis, or both before starting Auto-Tune.");
      }
      viewingLoadedRun.current = false;
      const baseline = await loadAutotuneHardwareBaseline();
      const targetVout = snapVoutToRegister(config.targets.vout_target_v, voutExponentFromReadback(baseline.voutReadback));
      const runConfig = withManualSearchCenters(
        { ...config, targets: { ...config.targets, vout_target_v: targetVout } },
        baseline.pidRequest,
        baseline.inductanceValues
      );
      const experiment = buildAutotuneExperiment(voutRequest, bodeSweepConfig, manualExperimentSettings, autotuneAnalysisSelection);
      const next =
        action === "start" ? await startTuning(runConfig, experiment) : await stepTuning(runConfig, experiment);
      setStatus(next);
      setConfig(normalizeTuningConfig(next.config));
      refreshAutotuneRuns();
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const resetAutotuneDefaults = async () => {
    try {
      const defaults = cloneDefaultConfig();
      viewingLoadedRun.current = false;
      const next = await resetTuning(defaults);
      setConfig(normalizeTuningConfig(next.config));
      setStatus(next);
      refreshAutotuneRuns();
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
      const outputNh = next.output_inductance?.value_nh;
      const effectiveNh = next.effective_lc_inductance?.value_nh;
      if (typeof outputNh === "number" || typeof effectiveNh === "number") {
        setInductanceRequest((current) => ({
          output_nh: typeof outputNh === "number" && Number.isFinite(outputNh) ? outputNh : current.output_nh,
          effective_lc_nh: typeof effectiveNh === "number" && Number.isFinite(effectiveNh) ? effectiveNh : current.effective_lc_nh
        }));
      }
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
      const currentModeFields = next.current_mode_registers?.fields;
      if (fields || currentModeFields) {
        setXdpPidRequest((current) => ({
          ...current,
          ...Object.fromEntries(xdpPidFieldNames.map((name) => [name, (fields?.[name] ?? currentModeFields?.[name])?.raw ?? current[name]]))
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
      if (!showSelectionError) {
        throw exc;
      }
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

  const archiveAutotuneRun = async () => {
    try {
      setAutotuneArchiveStatus("Saving current result...");
      const runId = status?.run?.run_id;
      const kind = status?.run?.kind ?? "recent";
      const result = await archiveCurrentTuningRun(undefined, runId, kind);
      setAutotuneArchiveStatus(`Saved permanently: ${formatRunLabel(result.saved_run, {}, 0, false)}`);
      await refreshAutotuneRuns(`saved:${result.saved_run.run_id}`);
      setError("");
    } catch (exc) {
      setAutotuneArchiveStatus("");
      setError(String(exc));
    }
  };

  const deleteAutotuneRunResult = async (selection = selectedAutotuneRun) => {
    try {
      const active = selection || selectedAutotuneRun;
      const [kind, ...rest] = active.split(":");
      const runId = rest.join(":");
      if (!kind || !runId) throw new Error("Select an auto-tune result to delete.");
      const label = formatLoadedRunLabel(autotuneRuns, kind, runId);
      const confirmed = window.confirm(`Delete ${label}?`);
      if (!confirmed) return;
      await deleteTuningRun(runId, kind);
      if (status?.run?.run_id === runId && status?.run?.kind === kind) {
        viewingLoadedRun.current = false;
      }
      setSelectedAutotuneRun("");
      setAutotuneArchiveStatus(`Deleted ${label}.`);
      await refreshAutotuneRuns();
      setError("");
    } catch (exc) {
      setAutotuneArchiveStatus("");
      setError(String(exc));
    }
  };

  const loadAutotuneRunResult = async (selection = selectedAutotuneRun) => {
    try {
      const [kind, ...rest] = selection.split(":");
      const runId = rest.join(":");
      if (!kind || !runId) throw new Error("Select an auto-tune result to load.");
      const loaded = await loadTuningRun(runId, kind);
      viewingLoadedRun.current = true;
      setSelectedAutotuneRun(`${kind}:${runId}`);
      setStatus(loaded);
      setConfig(normalizeTuningConfig(loaded.config));
      setAutotuneArchiveStatus(`Loaded ${formatLoadedRunLabel(autotuneRuns, kind, runId)}`);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const saveAutotuneGif = async () => {
    try {
      setAutotuneArchiveStatus("Saving GIF animation...");
      const parsedDurationS = Number.parseFloat(autotuneGifFrameDurationS);
      const safeDurationS = Number.isFinite(parsedDurationS) ? parsedDurationS : 0.1;
      const durationMs = Math.round(Math.max(0.05, Math.min(5, safeDurationS)) * 1000);
      const { targetRunId, targetKind } = selectedAutotuneGifRun();
      const result = await saveTuningAnimationGif(targetRunId, targetKind, durationMs);
      setAutotuneGif(result);
      const savedPath = gifResultDisplayPath(result);
      const savedSelection = `${result.kind}:${result.run_id}`;
      setSelectedAutotuneRun(savedSelection);
      setAutotuneArchiveStatus(savedPath ? `GIF animation saved: ${savedPath}` : "GIF animation saved, but no GIF file was returned.");
      await refreshAutotuneRuns(savedSelection);
      setError("");
    } catch (exc) {
      setAutotuneArchiveStatus("");
      setError(String(exc));
    }
  };

  const openAutotuneGif = async () => {
    try {
      setAutotuneArchiveStatus("Opening GIF animation...");
      const parsedDurationS = Number.parseFloat(autotuneGifFrameDurationS);
      const safeDurationS = Number.isFinite(parsedDurationS) ? parsedDurationS : 0.1;
      const durationMs = Math.round(Math.max(0.05, Math.min(5, safeDurationS)) * 1000);
      const { targetRunId, targetKind } = selectedAutotuneGifRun();
      const result = await openTuningAnimationGif(targetRunId, targetKind, durationMs);
      setAutotuneGif(result);
      const savedSelection = `${result.kind}:${result.run_id}`;
      setSelectedAutotuneRun(savedSelection);
      const openedPath = result.opened_path ? result.opened_path.replace(/\//g, "\\") : gifResultDisplayPath(result);
      setAutotuneArchiveStatus(openedPath ? `GIF animation opened: ${openedPath}` : "GIF animation opened.");
      await refreshAutotuneRuns(savedSelection);
      setError("");
    } catch (exc) {
      setAutotuneArchiveStatus("");
      setError(String(exc));
    }
  };

  const selectedAutotuneGifRun = () => {
    const selectedParts = selectedAutotuneRun ? selectedAutotuneRun.split(":") : [];
    const selectedKind = selectedParts[0] || undefined;
    const selectedRunId = selectedParts.slice(1).join(":") || undefined;
    return {
      targetRunId: status?.run?.run_id || selectedRunId,
      targetKind: status?.run?.kind || selectedKind
    };
  };

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
        bodeSweepConfig={bodeSweepConfig}
        setBodeSweepConfig={setBodeSweepConfig}
        analysisSelection={autotuneAnalysisSelection}
        setAnalysisSelection={setAutotuneAnalysisSelection}
        experimentSettings={manualExperimentSettings}
        setExperimentSettings={setManualExperimentSettings}
        runAction={runAction}
        resetAutotuneDefaults={resetAutotuneDefaults}
        readBoardInductance={readBoardInductance}
        writeOutputInductance={writeOutputInductance}
        writeEffectiveLcInductance={writeEffectiveLcInductance}
        runs={autotuneRuns}
        selectedRun={selectedAutotuneRun}
        setSelectedRun={setSelectedAutotuneRun}
        refreshRuns={refreshAutotuneRuns}
        archiveRun={archiveAutotuneRun}
        loadRun={loadAutotuneRunResult}
        deleteRun={deleteAutotuneRunResult}
        saveGif={saveAutotuneGif}
        openGif={openAutotuneGif}
        gifFrameDurationS={autotuneGifFrameDurationS}
        setGifFrameDurationS={setAutotuneGifFrameDurationS}
        archiveStatus={autotuneArchiveStatus}
        gifResult={autotuneGif}
        isViewingLoadedRun={viewingLoadedRun.current}
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
        experimentSettings={manualExperimentSettings}
        setExperimentSettings={setManualExperimentSettings}
        labels={t}
      /> : <SelfTestingView
        result={selfTest}
        running={selfTestRunning}
        activeKey={activeSelfTestKey}
        onRun={testInstruments}
        onRunDevice={testSingleInstrument}
      />}
      <LlmAssistantWidget
        activeTab={activeTab}
        status={status}
        config={config}
        language={language}
      />
    </main>
  );
}

function LlmAssistantWidget({
  activeTab,
  status,
  config,
  language
}: {
  activeTab: AppTab;
  status: TuningStatus | null;
  config: TuningConfig;
  language: Language;
}) {
  const greeting = language === "zh"
    ? "你好，我可以帮你理解这个 GUI、Auto-Tune 流程、Manual Tuning、Self Testing，以及 Bode 100 / Scope / Function Generator / PMBus 这些模块。"
    : "Hi, I can help explain this GUI, the Auto-Tune flow, Manual Tuning, Self Testing, and the Bode 100 / Scope / Function Generator / PMBus panels.";
  const [open, setOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState<LlmChatMessage[]>([{ role: "assistant", content: greeting }]);
  const messagesRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: "smooth" });
  }, [open, messages, sending]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;
    const nextMessages: LlmChatMessage[] = [...messages, { role: "user", content: text }];
    const context = {
      active_tab: activeTab,
      backend_state: status?.state ?? "unknown",
      backend_message: status?.message ?? "",
      current_iteration: status?.current?.iteration ?? null,
      best_iteration: status?.best?.iteration ?? null,
      best_penalty: status?.best?.metrics?.score ?? null,
      max_coarse_iterations: config.search.max_coarse_iterations,
      max_refined_iterations: config.search.max_refined_iterations,
      targets: {
        overshoot_pct: config.targets.overshoot_pct,
        undershoot_pct: config.targets.undershoot_pct,
        settling_time_us: config.targets.settling_time_s * 1e6,
        phase_margin_deg: config.targets.phase_margin_deg,
        crossover_upper_limit_hz: config.targets.crossover_frequency_hz
      }
    };
    setInput("");
    setMessages(nextMessages);
    setSending(true);
    try {
      const response = await sendLlmChat(nextMessages, context);
      setMessages([...nextMessages, { role: "assistant", content: response.reply ?? "No reply." }]);
    } catch (exc) {
      setMessages([...nextMessages, { role: "assistant", content: `LLM API error: ${String(exc)}` }]);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="llm-chat-root">
      {open && (
        <section className={`llm-chat-panel${fullscreen ? " is-fullscreen" : ""}`} aria-label="LLM assistant chat">
          <div className="llm-chat-header">
            <div>
              <h2>AI Help</h2>
            </div>
            <div className="llm-chat-header-actions">
              <button
                className="llm-chat-icon-button"
                onClick={() => setFullscreen((current) => !current)}
                aria-label={fullscreen ? "Exit fullscreen AI Help" : "Open AI Help fullscreen"}
                title={fullscreen ? "Exit fullscreen" : "Fullscreen"}
              >
                {fullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
              </button>
              <button
                className="llm-chat-icon-button"
                onClick={() => {
                  setOpen(false);
                  setFullscreen(false);
                }}
                aria-label="Close LLM chat"
              >
                <X size={18} />
              </button>
            </div>
          </div>
          <div className="llm-chat-messages" ref={messagesRef}>
            {messages.map((message, index) => (
              <div className={`llm-message ${message.role}`} key={`${message.role}-${index}`}>
                <span>{message.role === "user" ? "You" : "Assistant"}</span>
                <p>{message.content}</p>
              </div>
            ))}
            {sending && (
              <div className="llm-message assistant">
                <span>Assistant</span>
                <p>Thinking...</p>
              </div>
            )}
          </div>
          <div className="llm-chat-input-row">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleSend();
                }
              }}
              placeholder={language === "zh" ? "问一下这个 GUI 怎么用..." : "Ask about this GUI..."}
              rows={2}
            />
            <button className="primary llm-send-button" onClick={handleSend} disabled={!input.trim() || sending}>
              <Send size={16} />
            </button>
          </div>
        </section>
      )}
      <button className="llm-chat-launcher" onClick={() => setOpen((current) => !current)} aria-pressed={open}>
        <MessageCircle size={19} />
        <span>AI Help</span>
      </button>
    </div>
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
  const [showPenaltyExplanation, setShowPenaltyExplanation] = useState(false);

  return (
    <div className="header-toggles" aria-label="Display controls">
      <button
        type="button"
        className="penalty-explanation-button"
        onClick={() => setShowPenaltyExplanation(true)}
      >
        <HelpCircle size={16} />
        Penalty Explanation
      </button>
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
      {showPenaltyExplanation && (
        <PenaltyExplanationDialog onClose={() => setShowPenaltyExplanation(false)} />
      )}
    </div>
  );
}

function PenaltyExplanationDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="explanation-overlay" onMouseDown={onClose}>
      <section className="explanation-dialog" onMouseDown={(event) => event.stopPropagation()}>
        <div className="explanation-title">
          <div>
            <h2>Penalty Explanation</h2>
            <p>Lower penalty is better. A candidate passes only when every enabled target passes.</p>
          </div>
          <button type="button" className="explanation-close" onClick={onClose} aria-label="Close">
            <XCircle size={20} />
          </button>
        </div>

        <div className="explanation-section">
          <h3>Penalty Function</h3>
          <p>
            The tuner starts from transient penalty, then adds Bode penalty. If all enabled targets pass, a small
            reward is subtracted as a tie-breaker. The reward is intentionally small, so meeting the limits is more
            important than over-optimizing one metric.
          </p>
          <div className="formula-box">
            transient penalty = excess OS [%] + excess US [%] + 3 x excess OS settling [us] + 3 x excess US settling [us]
          </div>
          <div className="formula-box">
            Bode penalty = 1.5 x phase-margin shortage [deg] + 0.5 x crossover upper-limit excess [%]
          </div>
          <p>
            Default limits: OS at or below 3%, US at or below 3%, OS/US settling below 1 us, phase margin at or above
            45 deg, and crossover frequency upper limit = 200 kHz. The OS/US terms are percentage points, settling terms
            are microseconds, phase margin is degrees, and crossover upper-limit excess is percent.
          </p>
        </div>

        <div className="explanation-grid">
          <DefinitionCard
            title="Overshoot (OS)"
            text="After CH1 falling edge, CH3 should rise to the low-load steady voltage. OS is the amount that CH3 rises above that low-load steady average. The default limit is 3%."
          />
          <DefinitionCard
            title="Undershoot (US)"
            text="After CH1 rising edge, CH3 should fall to the high-load steady voltage. US is the amount that CH3 drops below that high-load steady average. The default limit is 3%."
          />
          <DefinitionCard
            title="OS / US Settling"
            text="Each load step has its own settling time in microseconds. The code finds the final steady voltage for that segment, then reports the last time CH3 is outside the +/-2% band. The default target is below 1 us for both OS settling and US settling."
          />
          <DefinitionCard
            title="Phase Margin"
            text="Phase margin is evaluated at the gain crossover frequency, where loop gain crosses 0 dB. The default requirement is phase margin greater than or equal to 45 deg."
          />
          <DefinitionCard
            title="Crossover Frequency"
            text="Crossover is the frequency where gain crosses 0 dB. The default upper limit is 200 kHz. Frequencies below 200 kHz receive a small tie-break reward; frequencies above 200 kHz receive penalty based on percent excess."
          />
        </div>
      </section>
    </div>
  );
}

function DefinitionCard({ title, text }: { title: string; text: string }) {
  return (
    <article className="definition-card">
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
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
  bodeSweepConfig,
  setBodeSweepConfig,
  analysisSelection,
  setAnalysisSelection,
  experimentSettings,
  setExperimentSettings,
  runAction,
  resetAutotuneDefaults,
  readBoardInductance,
  writeOutputInductance,
  writeEffectiveLcInductance,
  runs,
  selectedRun,
  setSelectedRun,
  refreshRuns,
  archiveRun,
  loadRun,
  deleteRun,
  saveGif,
  openGif,
  gifFrameDurationS,
  setGifFrameDurationS,
  archiveStatus,
  gifResult,
  isViewingLoadedRun
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
  bodeSweepConfig: BodeSweepConfig;
  setBodeSweepConfig: React.Dispatch<React.SetStateAction<BodeSweepConfig>>;
  analysisSelection: { transient: boolean; bode: boolean };
  setAnalysisSelection: React.Dispatch<React.SetStateAction<{ transient: boolean; bode: boolean }>>;
  experimentSettings: ManualExperimentSettings;
  setExperimentSettings: React.Dispatch<React.SetStateAction<ManualExperimentSettings>>;
  runAction: (action: "start" | "pause" | "resume" | "stop" | "step") => Promise<void>;
  resetAutotuneDefaults: () => Promise<void>;
  readBoardInductance: () => Promise<void>;
  writeOutputInductance: () => Promise<void>;
  writeEffectiveLcInductance: () => Promise<void>;
  runs: AutotuneRunsResponse | null;
  selectedRun: string;
  setSelectedRun: React.Dispatch<React.SetStateAction<string>>;
  refreshRuns: () => Promise<void>;
  archiveRun: () => Promise<void>;
  loadRun: (selection?: string) => Promise<void>;
  deleteRun: (selection?: string) => Promise<void>;
  saveGif: () => Promise<void>;
  openGif: () => Promise<void>;
  gifFrameDurationS: string;
  setGifFrameDurationS: React.Dispatch<React.SetStateAction<string>>;
  archiveStatus: string;
  gifResult: AutotuneGifResponse | null;
  isViewingLoadedRun: boolean;
}) {
  const voutExponent = voutExponentFromReadback(vout);
  const voutRegisterStep = voutRegisterStepFromExponent(voutExponent);
  const fgConfig = experimentSettings.fgConfig;
  const canRunAnalysis = analysisSelection.transient || analysisSelection.bode;
  const [selectedIteration, setSelectedIteration] = useState<number | null>(null);
  const selectedRecord = selectedIteration === null ? null : history.find((item) => item.iteration === selectedIteration) ?? null;
  const visibleRecord = selectedRecord ?? current;
  useEffect(() => {
    if (status?.state === "running") {
      setSelectedIteration(null);
    } else if (selectedIteration !== null && !history.some((item) => item.iteration === selectedIteration)) {
      setSelectedIteration(null);
    }
  }, [status?.state, history, selectedIteration]);
  const setFgConfig = (updater: React.SetStateAction<typeof defaultFunctionGeneratorConfig>) => {
    setExperimentSettings((current) => {
      const nextFgConfig = typeof updater === "function" ? updater(current.fgConfig) : updater;
      return { ...current, fgConfig: nextFgConfig };
    });
  };
  return (
      <div className="workspace">
        <aside className="control-rail">
          <Panel title="Run Control" icon={<Activity size={17} />}>
            <div className="autotune-controls">
              <div className="analysis-check-row">
                <label>
                  <input
                    type="checkbox"
                    checked={analysisSelection.transient}
                    onChange={(event) => setAnalysisSelection((current) => ({ ...current, transient: event.target.checked }))}
                  />
                  <span>Transient Analysis</span>
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={analysisSelection.bode}
                    onChange={(event) => setAnalysisSelection((current) => ({ ...current, bode: event.target.checked }))}
                  />
                  <span>Bode Analysis</span>
                </label>
              </div>
              <button className="autotune-control-button start" onClick={() => runAction("start")} disabled={status?.state === "running" || status?.state === "paused" || !canRunAnalysis}>
                Start Auto-Tune
              </button>
              <button className="autotune-control-button iterate" onClick={() => runAction("step")} disabled={status?.state === "running" || !canRunAnalysis}>
                Run Single Iteration
              </button>
              <div className="autotune-control-grid">
                <button className="autotune-control-button" onClick={() => runAction("pause")} disabled={status?.state !== "running"}>
                  Pause
                </button>
                <button className="autotune-control-button" onClick={() => runAction("resume")} disabled={status?.state !== "paused" && status?.state !== "stopped"}>
                  Resume
                </button>
                <button className="autotune-control-button" onClick={() => runAction("stop")} disabled={status?.state !== "running" && status?.state !== "paused"}>
                  Stop
                </button>
              </div>
              <div className="gif-save-row">
                <button className="autotune-control-button gif-save-button" onClick={saveGif} disabled={status?.state === "running" || history.length < 2}>
                  Save GIF
                </button>
                <button className="autotune-control-button gif-save-button" onClick={openGif} disabled={status?.state === "running" || history.length < 2}>
                  Open
                </button>
                <label className="gif-duration-field">
                  <span>s / frame</span>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={gifFrameDurationS}
                    onChange={(event) => setGifFrameDurationS(event.target.value)}
                    onBlur={() => {
                      const parsed = Number.parseFloat(gifFrameDurationS);
                      if (!Number.isFinite(parsed)) {
                        setGifFrameDurationS("0.1");
                        return;
                      }
                      setGifFrameDurationS(String(Math.max(0.05, Math.min(5, parsed))));
                    }}
                  />
                </label>
              </div>
              <button className="autotune-control-button reset" onClick={resetAutotuneDefaults} disabled={status?.state === "running"}>
                Reset to Defaults
              </button>
            </div>
            <p className="message-line">{status?.message ?? "Connecting to local backend..."}</p>
          </Panel>

          <ResultLibraryPanel
            runs={runs}
            currentRun={status?.run}
            selectedRun={selectedRun}
            setSelectedRun={setSelectedRun}
            refreshRuns={refreshRuns}
            archiveRun={archiveRun}
            loadRun={loadRun}
            deleteRun={deleteRun}
            archiveStatus={archiveStatus}
            gifResult={gifResult}
            disabled={!isViewingLoadedRun && status?.state === "running"}
          />

          <Panel title="Function Generator" icon={<Activity size={17} />}>
            <div className="autotune-run-settings">
              <FunctionGeneratorSettingsRow fgConfig={fgConfig} setFgConfig={setFgConfig} compact />
            </div>
          </Panel>

          <Panel title="Bode Analysis" icon={<Gauge size={17} />}>
            <div className="autotune-run-settings">
              <ReadOnlyField label="Measurement Mode" value="Gain/Phase Log Sweep" />
              <FrequencyField
                label="Start Frequency (Hz)"
                value={bodeSweepConfig.start_hz}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, start_hz: Math.max(1, value) }))}
              />
              <FrequencyField
                label="Stop Frequency (Hz)"
                value={bodeSweepConfig.stop_hz}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, stop_hz: Math.max(1, value) }))}
              />
              <NumberField
                label="Sweep Points"
                value={bodeSweepConfig.points}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, points: Math.max(2, Math.round(value)) }))}
              />
              <NumberField
                label="Receiver Bandwidth (Hz)"
                value={bodeSweepConfig.bandwidth_hz}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, bandwidth_hz: Math.max(1, value) }))}
              />
              <NumberField
                label="Source Level (Vpp)"
                value={bodeSweepConfig.source_vpp ?? 0.1}
                step={0.01}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, source_vpp: Math.max(0.001, value), source_dbm: null }))}
              />
            </div>
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
            <NumberField label="Phase Margin (deg)" value={config.targets.phase_margin_deg} onChange={(value) => updateConfig(setConfig, "targets", "phase_margin_deg", value)} />
            <FrequencyField label="Crossover Frequency Upper Limit (Hz)" value={config.targets.crossover_frequency_hz} onChange={(value) => updateConfig(setConfig, "targets", "crossover_frequency_hz", value)} />
          </Panel>

          <Panel title="Search Space" icon={<RefreshCw size={17} />}>
            <DeferredNumberField
              label="Max Coarse Iterations"
              value={config.search.max_coarse_iterations ?? Math.max(1, Math.round((config.search.max_iterations ?? 40) / 2))}
              integer
              min={1}
              onCommit={(value) => updateSearchIterationBudget(setConfig, "max_coarse_iterations", Math.max(1, Math.round(value)))}
            />
            <DeferredNumberField
              label="Max Refined Iterations"
              value={config.search.max_refined_iterations ?? Math.max(0, (config.search.max_iterations ?? 40) - Math.max(1, Math.round((config.search.max_iterations ?? 40) / 2)))}
              integer
              min={0}
              onCommit={(value) => updateSearchIterationBudget(setConfig, "max_refined_iterations", Math.max(0, Math.round(value)))}
            />
            {hardwareSearchFields.map((field) => (
              <SearchParameterField
                key={field.key}
                label={field.label}
                limit={field.limit}
                value={config.search[field.key] ?? defaultConfig.search[field.key]}
                integer={field.key.startsWith("mod0_")}
                onChange={(value) => {
                  updateSearchParameter(setConfig, field.key, value);
                  if (field.key === "output_inductance_nh") {
                    setInductanceRequest((current) => ({ ...current, output_nh: value.center }));
                  } else if (field.key === "effective_lc_inductance_nh") {
                    setInductanceRequest((current) => ({ ...current, effective_lc_nh: value.center }));
                  } else {
                    setXdpPidRequest((current) => ({ ...current, [field.key]: Math.round(value.center) }));
                  }
                }}
              />
            ))}
          </Panel>
        </aside>

        <section className="plot-deck">
          <div className="autotune-result-grid">
            <Panel title="Transient Response" icon={<Activity size={17} />}>
              <AutotuneResultImage
                record={visibleRecord}
                resultKey="scope_result"
                imageKey="scope_png"
                versionKeys={["capture_id", "duration_s"]}
                alt="Latest transient response with CH1 input signal and CH3 output voltage"
                emptyText="No transient PNG yet. Run one hardware iteration to capture CH1 input and CH3 output voltage."
              />
            </Panel>
            <Panel title="Bode Plot" icon={<Gauge size={17} />}>
              <AutotuneResultImage
                record={visibleRecord}
                resultKey="bode_result"
                imageKey="bode_png"
                versionKeys={["sweep_id", "duration_s"]}
                alt="Latest Bode 100 sweep with gain and phase"
                emptyText="No Bode PNG yet. Run one hardware iteration to capture gain and phase."
              />
            </Panel>
          </div>
          <div className="autotune-result-grid">
            <Panel title="Penalty Trend" icon={<Pause size={17} />}>
              <ReactECharts option={penaltyOption(history)} className="chart" />
            </Panel>
          </div>
          <Panel title="Iteration History" icon={<RefreshCw size={17} />}>
            <IterationTable history={history} selectedIteration={selectedIteration} onSelectIteration={setSelectedIteration} />
          </Panel>
        </section>

        <aside className="metrics-rail">
          <RunCurrentPanel
            title={selectedRecord ? `Iteration #${selectedRecord.iteration}` : "Current Result"}
            candidateTitle={selectedRecord ? "Candidate" : "Current Candidate"}
            metricsTitle="Current Metrics"
            status={status}
            current={current}
            best={best}
            history={history}
            record={visibleRecord}
          />
          <CandidateMetricsPanel
            title="Best Result"
            candidateTitle="Best Candidate"
            metricsTitle="Best Metrics"
            record={best}
            showIteration
          />
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
  experimentSettings,
  setExperimentSettings,
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
  experimentSettings: ManualExperimentSettings;
  setExperimentSettings: React.Dispatch<React.SetStateAction<ManualExperimentSettings>>;
  labels: (typeof copy)[Language];
}) {
  const voutExponent = voutExponentFromReadback(vout);
  const voutRegisterStep = voutRegisterStepFromExponent(voutExponent);
  const [showConnectionSettings, setShowConnectionSettings] = useState(false);
  const [axisEditor, setAxisEditor] = useState<"telemetry" | "scope" | "bode" | null>(null);
  const [telemetryAxisSettings, setTelemetryAxisSettings] = useState<TelemetryAxisSettings>({ ...defaultTelemetryAxisSettings });
  const [scopeAxisSettings, setScopeAxisSettings] = useState<ScopeAxisSettings>({
    ...experimentSettings.scopeAxisSettings,
    channelAxes: { ...experimentSettings.scopeAxisSettings.channelAxes }
  });
  const [bodeAxisSettings, setBodeAxisSettings] = useState<ChartAxisSettings>({ ...defaultBodeAxisSettings });
  const [scopeChannels, setScopeChannels] = useState<string[]>(experimentSettings.scopeChannels);
  const [scopeMeasurements, setScopeMeasurements] = useState<string[]>(experimentSettings.scopeMeasurements);
  const [scopeCapture, setScopeCapture] = useState<ScopeCaptureReadback | null>(null);
  const [scopeWebPlotOpen, setScopeWebPlotOpen] = useState(true);
  const [scopePngPlotOpen, setScopePngPlotOpen] = useState(true);
  const [bodeWebPlotOpen, setBodeWebPlotOpen] = useState(true);
  const [bodePngPlotOpen, setBodePngPlotOpen] = useState(true);
  const [scopeRunning, setScopeRunning] = useState(false);
  const [scopeAcquisitionRunning, setScopeAcquisitionRunning] = useState(false);
  const [scopeAcquisitionBusy, setScopeAcquisitionBusy] = useState(false);
  const [fgConfig, setFgConfig] = useState({ ...experimentSettings.fgConfig });
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
    setExperimentSettings({
      fgConfig: { ...fgConfig },
      scopeChannels: [...scopeChannels],
      scopeMeasurements: [...scopeMeasurements],
      scopeAxisSettings: {
        ...scopeAxisSettings,
        channelAxes: { ...scopeAxisSettings.channelAxes }
      }
    });
  }, [fgConfig, scopeChannels, scopeMeasurements, scopeAxisSettings, setExperimentSettings]);

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
      setFullAcquisitionStatus(`Disabling function generator output on CH${acquisitionFgConfig.channel} before Bode...`);
      setFgRunning(true);
      setFgReadback(await setFunctionGenerator({
        channel: acquisitionFgConfig.channel,
        mode: acquisitionFgConfig.mode,
        output_enabled: false
      }));
      setFgRunning(false);
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
                label="Source (Vpp)"
                value={bodeSweepConfig.source_vpp ?? 0.1}
                step={0.01}
                onChange={(value) => setBodeSweepConfig((current) => ({ ...current, source_vpp: Math.max(0.001, value), source_dbm: null }))}
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
                  <FunctionGeneratorSettingsRow fgConfig={fgConfig} setFgConfig={setFgConfig} />
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

function FunctionGeneratorSettingsRow({
  fgConfig,
  setFgConfig,
  compact = false
}: {
  fgConfig: typeof defaultFunctionGeneratorConfig;
  setFgConfig: React.Dispatch<React.SetStateAction<typeof defaultFunctionGeneratorConfig>>;
  compact?: boolean;
}) {
  return (
    <>
      <SelectField label="Channel #" value={String(fgConfig.channel)} options={["1", "2"]} onChange={(value) => setFgConfig((current) => ({ ...current, channel: Number(value) }))} />
      <SelectField label={compact ? "Wave" : "Waveform"} value={fgConfig.mode} options={["square", "pulse", "dc", "sine"]} onChange={(value) => setFgConfig((current) => ({ ...current, mode: value }))} />
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
    </>
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

function DeferredNumberField({
  label,
  value,
  integer = false,
  min,
  max,
  onCommit
}: {
  label: string;
  value: number;
  integer?: boolean;
  min?: number;
  max?: number;
  onCommit: (value: number) => void;
}) {
  const displayValue = Number.isFinite(value) ? String(value) : "";
  const [draftValue, setDraftValue] = useState(displayValue);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    if (!isEditing) {
      setDraftValue(displayValue);
    }
  }, [displayValue, isEditing]);

  const commitValue = () => {
    const trimmed = draftValue.trim();
    if (trimmed === "" || trimmed === "-" || trimmed === "." || trimmed === "-.") {
      setDraftValue(displayValue);
      setIsEditing(false);
      return;
    }

    let parsedValue = Number(trimmed);
    if (!Number.isFinite(parsedValue)) {
      setDraftValue(displayValue);
      setIsEditing(false);
      return;
    }
    if (integer) {
      parsedValue = Math.round(parsedValue);
    }
    if (min !== undefined) {
      parsedValue = Math.max(min, parsedValue);
    }
    if (max !== undefined) {
      parsedValue = Math.min(max, parsedValue);
    }
    onCommit(parsedValue);
    setDraftValue(String(parsedValue));
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
        type="text"
        inputMode={integer ? "numeric" : "decimal"}
        value={draftValue}
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

function ReadOnlyField({ label, value }: { label: string; value: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input className="readonly-input" value={value} readOnly />
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
    </div>
  );
}

function AutotuneResultImage({
  record,
  resultKey,
  imageKey,
  versionKeys,
  alt,
  emptyText
}: {
  record: IterationRecord | null;
  resultKey: "scope_result" | "bode_result";
  imageKey: "scope_png" | "bode_png";
  versionKeys: string[];
  alt: string;
  emptyText: string;
}) {
  const result = record?.[resultKey] as Record<string, unknown> | null | undefined;
  const imagePath = typeof result?.[imageKey] === "string" ? String(result[imageKey]) : "";
  const errorKey = imageKey === "scope_png" ? "scope_png_error" : "bode_png_error";
  const pendingKey = imageKey === "scope_png" ? "scope_png_pending" : "bode_png_pending";
  const error = typeof result?.[errorKey] === "string" ? String(result[errorKey]) : "";
  const pending = result?.[pendingKey] === true;
  const version = versionKeys.map((key) => result?.[key]).find((value) => value !== undefined && value !== null) ?? record?.iteration ?? Date.now();
  const [retryNonce, setRetryNonce] = useState(0);
  const [loadFailed, setLoadFailed] = useState(false);
  useEffect(() => {
    setRetryNonce(0);
    setLoadFailed(false);
  }, [imagePath, String(version), pending]);

  if (pending) {
    return (
      <div className="autotune-image-empty">
        <p>Generating image...</p>
        {error ? <p className="bad-text">{error}</p> : null}
      </div>
    );
  }
  if (!imagePath) {
    return (
      <div className="autotune-image-empty">
        <p>{emptyText}</p>
        {error ? <p className="bad-text">{error}</p> : null}
      </div>
    );
  }
  return (
    <div className="autotune-image-preview">
      <img
        src={`${imagePath}?v=${encodeURIComponent(`${String(version)}-${retryNonce}`)}`}
        alt={alt}
        onLoad={() => setLoadFailed(false)}
        onError={() => {
          if (retryNonce < 6) {
            window.setTimeout(() => setRetryNonce((value) => value + 1), 500);
          } else {
            setLoadFailed(true);
          }
        }}
      />
      {loadFailed ? <p className="message-line bad-text">Image file is not ready yet. Retrying on the next status update.</p> : null}
      {error ? <p className="message-line bad-text">{error}</p> : null}
    </div>
  );
}

type ResultRunSummary = AutotuneRunsResponse["recent"][number];

function runDateKey(runId: string): string {
  const friendly = String(runId).match(/^(?:Recent|Permanent)_(\d{4}-\d{2}-\d{2})_/);
  if (friendly) return friendly[1];
  const match = String(runId).match(/^(\d{4})(\d{2})(\d{2})/);
  if (!match) return String(runId || "Unknown");
  return `${match[1]}-${match[2]}-${match[3]}`;
}

function countRunDates(runs: Array<{ run_id: string }>): Record<string, number> {
  return runs.reduce<Record<string, number>>((counts, run) => {
    const key = runDateKey(run.run_id);
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});
}

function formatRunLabel(
  run: Pick<ResultRunSummary, "run_id" | "kind" | "display_name">,
  dateCounts: Record<string, number>,
  index: number,
  includeDuplicateIndex = true
): string {
  if (run.display_name && / #\d+$/.test(run.display_name)) return run.display_name;
  const date = runDateKey(run.run_id);
  const kind = run.kind === "saved" ? "Permanent" : "Recent";
  const duplicateSuffix = includeDuplicateIndex && (dateCounts[date] ?? 0) > 1 ? ` #${index + 1}` : "";
  return `${kind} / ${date}${duplicateSuffix}`;
}

function formatLoadedRunLabel(runs: AutotuneRunsResponse | null, kind: string, runId: string): string {
  const collection = kind === "saved" ? runs?.saved ?? [] : runs?.recent ?? [];
  const index = collection.findIndex((run) => run.run_id === runId);
  const run = collection[index] ?? { run_id: runId, kind };
  return formatRunLabel(run, countRunDates(collection), Math.max(index, 0));
}

function versionedAssetUrl(path: string | null | undefined, version: number | string | null | undefined): string {
  if (!path) {
    return "";
  }
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}v=${encodeURIComponent(String(version ?? Date.now()))}`;
}

function gifFolderDisplayPath(path?: string | null): string {
  if (!path) return "";
  const clean = path.split("?")[0].replace(/^\/+/, "");
  const slashIndex = clean.lastIndexOf("/");
  const folder = slashIndex >= 0 ? clean.slice(0, slashIndex) : clean;
  return folder.replace(/\//g, "\\");
}

function gifResultDisplayPath(result: AutotuneGifResponse): string {
  const gifPath = result.combined_gif ?? result.transient_gif ?? result.bode_gif;
  if (gifPath) {
    return gifPath.replace(/^\/results\//, "results/").replace(/\//g, "\\");
  }
  return result.animation_dir ? result.animation_dir.replace(/\//g, "\\") : "";
}

function ResultLibraryPanel({
  runs,
  currentRun,
  selectedRun,
  setSelectedRun,
  refreshRuns,
  archiveRun,
  loadRun,
  deleteRun,
  archiveStatus,
  gifResult,
  disabled
}: {
  runs: AutotuneRunsResponse | null;
  currentRun: TuningStatus["run"];
  selectedRun: string;
  setSelectedRun: React.Dispatch<React.SetStateAction<string>>;
  refreshRuns: () => Promise<void>;
  archiveRun: () => Promise<void>;
  loadRun: (selection?: string) => Promise<void>;
  deleteRun: (selection?: string) => Promise<void>;
  archiveStatus: string;
  gifResult: AutotuneGifResponse | null;
  disabled?: boolean;
}) {
  const recent = runs?.recent ?? [];
  const saved = runs?.saved ?? [];
  const options = [
    ...recent.map((run) => ({ ...run, optionKey: `recent:${run.run_id}` })),
    ...saved.map((run) => ({ ...run, optionKey: `saved:${run.run_id}` }))
  ];
  const activeSelection = selectedRun || options[0]?.optionKey || "";
  const recentLabelCounts = countRunDates(recent);
  const savedLabelCounts = countRunDates(saved);

  return (
    <Panel title="Result Library" icon={<RefreshCw size={17} />}>
      <div className="result-library">
        <dl className="kv compact-kv">
          <dt>Current run</dt>
          <dd>
            {currentRun?.run_id
              ? formatRunLabel(
                  { run_id: currentRun.run_id, kind: currentRun.kind, display_name: currentRun.display_name },
                  currentRun.kind === "saved" ? savedLabelCounts : recentLabelCounts,
                  0,
                  false
                )
              : "--"}
          </dd>
          <dt>Recent limit</dt>
          <dd>{currentRun?.recent_limit ?? 5}</dd>
        </dl>
        <button className="autotune-control-button" onClick={archiveRun} disabled={disabled || !currentRun}>
          Save Current Permanently
        </button>
        <div className="result-load-row">
          <select
            value={activeSelection}
            onChange={(event) => setSelectedRun(event.target.value)}
            disabled={options.length === 0}
          >
            {options.length === 0 ? <option value="">No saved result</option> : null}
            {recent.length ? <option disabled>Recent results</option> : null}
            {recent.map((run, index) => (
              <option key={`recent:${run.run_id}`} value={`recent:${run.run_id}`}>
                {formatRunLabel(run, recentLabelCounts, index)} / {run.iteration_count ?? 0} iter
              </option>
            ))}
            {saved.length ? <option disabled>Permanent results</option> : null}
            {saved.map((run, index) => (
              <option key={`saved:${run.run_id}`} value={`saved:${run.run_id}`}>
                {formatRunLabel(run, savedLabelCounts, index)} / {run.iteration_count ?? 0} iter
              </option>
            ))}
          </select>
          <button onClick={() => loadRun(activeSelection)} disabled={disabled || !activeSelection}>
            Load
          </button>
          <button onClick={refreshRuns} disabled={disabled}>
            Refresh
          </button>
          <button className="result-delete-button" onClick={() => deleteRun(activeSelection)} disabled={disabled || !activeSelection}>
            Delete
          </button>
        </div>
        {archiveStatus ? (
          <p className="message-line result-save-message">
            {archiveStatus.startsWith("GIF animation saved:") || archiveStatus.startsWith("GIF animation opened:") ? (
              <>
                <span>{archiveStatus.startsWith("GIF animation opened:") ? "GIF animation opened:" : "GIF animation saved:"}</span>
                <code>{archiveStatus.replace(/^GIF animation (?:saved|opened):/, "").trim()}</code>
              </>
            ) : (
              archiveStatus
            )}
          </p>
        ) : null}
        {gifResult?.combined_gif || gifResult?.transient_gif || gifResult?.bode_gif ? (
          <div className="result-links">
            {gifResult.combined_gif ? (
              <a href={versionedAssetUrl(gifResult.combined_gif, gifResult.generated_at)} target="_blank" rel="noreferrer">Animation GIF</a>
            ) : null}
            {gifResult.transient_gif ? (
              <a href={versionedAssetUrl(gifResult.transient_gif, gifResult.generated_at)} target="_blank" rel="noreferrer">Transient GIF</a>
            ) : null}
            {gifResult.bode_gif ? (
              <a href={versionedAssetUrl(gifResult.bode_gif, gifResult.generated_at)} target="_blank" rel="noreferrer">Bode GIF</a>
            ) : null}
          </div>
        ) : null}
      </div>
    </Panel>
  );
}

function BodeIterationMetrics({ record }: { record: IterationRecord | null }) {
  const metrics = record?.metrics;
  const bodeResult = record?.bode_result as Record<string, unknown> | null | undefined;
  const margins = bodeResult?.margins as Record<string, unknown> | null | undefined;
  const phaseMargin = asDisplayNumber(metrics?.phase_margin_deg ?? margins?.phase_margin_deg);
  const crossover = asDisplayNumber(metrics?.crossover_frequency_hz ?? margins?.phase_crossover_hz);
  const gainMargin = asDisplayNumber(metrics?.gain_margin_db ?? margins?.gain_margin_db);
  return (
    <dl className="kv">
      <dt>Phase Margin</dt>
      <dd>{phaseMargin === null ? "--" : `${phaseMargin.toFixed(2)} deg`}</dd>
      <dt>Crossover</dt>
      <dd>{crossover === null ? "--" : formatMaybeFrequency(crossover)}</dd>
      <dt>Gain Margin</dt>
      <dd>{gainMargin === null ? "--" : `${gainMargin.toFixed(2)} dB`}</dd>
      <dt>Bode data</dt>
      <dd>{typeof bodeResult?.data_file === "string" ? "saved" : "--"}</dd>
    </dl>
  );
}

function asDisplayNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
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
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} MHz`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(2)} kHz`;
  return `${value.toFixed(2)} Hz`;
}

function formatSettlingUs(value: number | null | undefined, fallback: number | null | undefined) {
  const selected = value === null || value === undefined ? fallback : value;
  if (selected === null || selected === undefined || !Number.isFinite(selected)) return "--";
  return `${(selected * 1e6).toFixed(1)} us`;
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

function SearchParameterField({
  label,
  limit,
  value,
  integer = false,
  onChange
}: {
  label: string;
  limit: string;
  value: SearchParameter;
  integer?: boolean;
  onChange: (value: SearchParameter) => void;
}) {
  const update = (field: "min" | "max" | "points", nextValue: number) => {
    const normalized = field === "points" || integer ? Math.round(nextValue) : nextValue;
    const next = { ...value, [field]: normalized };
    if (field === "min" && next.min > next.max) {
      next.max = next.min;
    }
    if (field === "max" && next.max < next.min) {
      next.min = next.max;
    }
    next.points = Math.max(1, Math.min(101, Math.round(next.points || 1)));
    next.step = next.points > 1 ? Math.abs((next.max - next.min) / (next.points - 1)) : Math.abs(next.max - next.min);
    if (integer) {
      next.step = Math.max(1, Math.round(next.step));
    }
    next.center = Math.min(Math.max(next.center, next.min), next.max);
    onChange(next);
  };

  return (
    <div className="search-parameter">
      <div className="search-parameter-title">
        <span>{label}</span>
        <small>{limit}</small>
      </div>
      <div className="search-parameter-grid">
        <DeferredSearchInput label="min" value={value.min} integer={integer} onCommit={(nextValue) => update("min", nextValue)} />
        <DeferredSearchInput label="max" value={value.max} integer={integer} onCommit={(nextValue) => update("max", nextValue)} />
        <DeferredSearchInput label="points" value={value.points ?? 1} integer min={1} max={101} onCommit={(nextValue) => update("points", nextValue)} />
      </div>
    </div>
  );
}

function DeferredSearchInput({
  label,
  value,
  integer = false,
  min,
  max,
  onCommit
}: {
  label: string;
  value: number;
  integer?: boolean;
  min?: number;
  max?: number;
  onCommit: (value: number) => void;
}) {
  const displayValue = Number.isFinite(value) ? String(value) : "";
  const [draftValue, setDraftValue] = useState(displayValue);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    if (!isEditing) {
      setDraftValue(displayValue);
    }
  }, [displayValue, isEditing]);

  const commitValue = () => {
    const trimmed = draftValue.trim();
    if (trimmed === "" || trimmed === "-" || trimmed === "." || trimmed === "-.") {
      setDraftValue(displayValue);
      setIsEditing(false);
      return;
    }
    let parsedValue = Number(trimmed);
    if (!Number.isFinite(parsedValue)) {
      setDraftValue(displayValue);
      setIsEditing(false);
      return;
    }
    if (integer) {
      parsedValue = Math.round(parsedValue);
    }
    if (min !== undefined) {
      parsedValue = Math.max(min, parsedValue);
    }
    if (max !== undefined) {
      parsedValue = Math.min(max, parsedValue);
    }
    onCommit(parsedValue);
    setDraftValue(String(parsedValue));
    setIsEditing(false);
  };

  const cancelValue = () => {
    setDraftValue(displayValue);
    setIsEditing(false);
  };

  return (
    <label>
      <span>{label}</span>
      <input
        type="text"
        inputMode={integer ? "numeric" : "decimal"}
        value={draftValue}
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

function PidPanel({ title, record, showIteration = false }: { title: string; record: IterationRecord | null; showIteration?: boolean }) {
  return (
    <Panel title={title} icon={<Gauge size={17} />}>
      <PidReadout record={record} showIteration={showIteration} />
    </Panel>
  );
}

function CandidateMetricsPanel({
  title,
  candidateTitle,
  metricsTitle,
  record,
  showIteration = false
}: {
  title: string;
  candidateTitle: string;
  metricsTitle: string;
  record: IterationRecord | null;
  showIteration?: boolean;
}) {
  return (
    <Panel title={title} icon={<Gauge size={17} />}>
      <div className="combined-result-section">
        <h4>{candidateTitle}</h4>
        <PidReadout record={record} showIteration={showIteration} />
      </div>
      <div className="combined-result-divider" />
      <div className="combined-result-section">
        <h4>{metricsTitle}</h4>
        <Metrics record={record} />
      </div>
    </Panel>
  );
}

function RunCurrentPanel({
  title,
  candidateTitle,
  metricsTitle,
  status,
  current,
  best,
  history,
  record
}: {
  title: string;
  candidateTitle: string;
  metricsTitle: string;
  status: TuningStatus | null;
  current: IterationRecord | null;
  best: IterationRecord | null;
  history: IterationRecord[];
  record: IterationRecord | null;
}) {
  return (
    <Panel title={title} icon={<Activity size={17} />}>
      <div className="combined-result-section">
        <h4>Run Status</h4>
        <RunStatusReadout status={status} current={current} best={best} history={history} />
      </div>
      <div className="combined-result-divider" />
      <div className="combined-result-section">
        <h4>{candidateTitle}</h4>
        <PidReadout record={record} />
      </div>
      <div className="combined-result-divider" />
      <div className="combined-result-section">
        <h4>{metricsTitle}</h4>
        <Metrics record={record} />
      </div>
    </Panel>
  );
}

function PidReadout({ record, showIteration = false }: { record: IterationRecord | null; showIteration?: boolean }) {
  const candidate = record?.candidate;
  return (
    <dl className="kv mono">
      {showIteration ? (
        <>
          <dt>Iteration</dt>
          <dd>{record?.iteration ? `#${record.iteration}` : "--"}</dd>
        </>
      ) : null}
      <dt>mod0_kp</dt>
      <dd>{candidate ? candidate.mod0_kp : "--"}</dd>
      <dt>mod0_ki</dt>
      <dd>{candidate ? candidate.mod0_ki : "--"}</dd>
      <dt>mod0_kd</dt>
      <dd>{candidate ? candidate.mod0_kd : "--"}</dd>
      <dt>mod0_kpole1</dt>
      <dd>{candidate ? candidate.mod0_kpole1 : "--"}</dd>
      <dt>mod0_kpole2</dt>
      <dd>{candidate ? candidate.mod0_kpole2 : "--"}</dd>
      <dt>cm gain</dt>
      <dd>{candidate ? candidate.mod0_cm_gain : "--"}</dd>
      <dt>Output L</dt>
      <dd>{candidate ? `${candidate.output_inductance_nh.toFixed(3)} nH` : "--"}</dd>
      <dt>Effective Lc</dt>
      <dd>{candidate ? `${candidate.effective_lc_inductance_nh.toFixed(3)} nH` : "--"}</dd>
    </dl>
  );
}

function RunStatusPanel({
  status,
  current,
  best,
  history
}: {
  status: TuningStatus | null;
  current: IterationRecord | null;
  best: IterationRecord | null;
  history: IterationRecord[];
}) {
  return (
    <Panel title="Run Status" icon={<Activity size={17} />}>
      <RunStatusReadout status={status} current={current} best={best} history={history} />
    </Panel>
  );
}

function RunStatusReadout({
  status,
  current,
  best,
  history
}: {
  status: TuningStatus | null;
  current: IterationRecord | null;
  best: IterationRecord | null;
  history: IterationRecord[];
}) {
  const iteration = history.length;
  const search = normalizeTuningConfig(status?.config).search;
  const maxCoarse = searchCoarseBudget(search);
  const maxRefined = searchRefinedBudget(search);
  const safeMaxIterations = Math.max(1, maxCoarse + maxRefined);
  const coarseDone = Math.min(maxCoarse, countCoarseIterations(history));
  const refinedDone = Math.min(maxRefined, Math.max(0, iteration - coarseDone));
  const progressPct = Math.max(0, Math.min(100, (iteration / safeMaxIterations) * 100));
  const currentLabel = current ? `#${current.iteration} ${current.phase}` : "-";
  const bestPenalty = best ? displayPenaltyScore(best) : null;
  const bestLabel = best ? `#${best.iteration} penalty ${formatMaybeNumber(bestPenalty, 3)}` : "-";
  const resultsSaved = Boolean(
    current?.scope_result && typeof current.scope_result === "object" && "scope_png" in current.scope_result
  ) || Boolean(
    current?.bode_result && typeof current.bode_result === "object" && "bode_png" in current.bode_result
  );

  return (
    <>
      <dl className="kv run-status-kv">
        <dt>Backend</dt>
        <dd>{status ? status.state : "connecting"}</dd>
        <dt>Mode</dt>
        <dd>{status?.state === "running" ? "Running" : "Ready"}</dd>
        <dt>Iter Coarse</dt>
        <dd>{coarseDone} / {maxCoarse}</dd>
        <dt>Iter Refined</dt>
        <dd>{refinedDone} / {maxRefined}</dd>
        <dt>Phase</dt>
        <dd>{current?.phase ?? "-"}</dd>
        <dt>Current</dt>
        <dd>{currentLabel}</dd>
        <dt>Best</dt>
        <dd>{bestLabel}</dd>
        <dt>Results</dt>
        <dd>{resultsSaved ? "saved" : "-"}</dd>
      </dl>
      <div className="run-progress" aria-label="Auto-tune progress">
        <div style={{ width: `${progressPct}%` }} />
      </div>
      <div className="run-progress-label">{progressPct.toFixed(0)}%</div>
    </>
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
      <dt>OS settling</dt>
      <dd>{formatSettlingUs(record.metrics.overshoot_settling_time_s, record.metrics.settling_time_s)}</dd>
      <dt>US settling</dt>
      <dd>{formatSettlingUs(record.metrics.undershoot_settling_time_s, record.metrics.settling_time_s)}</dd>
      <dt>Low-load Vout</dt>
      <dd>{record.metrics.low_load_steady_v === null || record.metrics.low_load_steady_v === undefined ? "--" : `${record.metrics.low_load_steady_v.toFixed(3)} V`}</dd>
      <dt>High-load Vout</dt>
      <dd>{record.metrics.high_load_steady_v === null || record.metrics.high_load_steady_v === undefined ? "--" : `${record.metrics.high_load_steady_v.toFixed(3)} V`}</dd>
      <dt>Phase Margin</dt>
      <dd>{record.metrics.phase_margin_deg === null || record.metrics.phase_margin_deg === undefined ? "--" : `${record.metrics.phase_margin_deg.toFixed(2)} deg`}</dd>
      <dt>Crossover</dt>
      <dd>{formatMaybeFrequency(record.metrics.crossover_frequency_hz)}</dd>
      <dt>Gain Margin</dt>
      <dd>{record.metrics.gain_margin_db === null || record.metrics.gain_margin_db === undefined ? "--" : `${record.metrics.gain_margin_db.toFixed(2)} dB`}</dd>
      <dt>Penalty</dt>
      <dd>{formatMaybeNumber(displayPenaltyScore(record), 3)}</dd>
      <dt>Pass</dt>
      <dd>{record.metrics.passed ? "yes" : "no"}</dd>
      <dt>Reason</dt>
      <dd>{record.metrics.pass_reasons?.join(", ") || "--"}</dd>
      <dt>Duration</dt>
      <dd>{record.duration_s === undefined ? "--" : `${record.duration_s.toFixed(2)} s`}</dd>
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
  const pidFields = pid?.pid_registers?.fields;
  const currentModeFields = pid?.current_mode_registers?.fields;
  if (!pidFields && !currentModeFields) return <p className="readback">No mod0 PID field readback yet.</p>;
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
            const field = pidFields?.[name] ?? currentModeFields?.[name];
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

function IterationTable({
  history,
  selectedIteration,
  onSelectIteration
}: {
  history: IterationRecord[];
  selectedIteration: number | null;
  onSelectIteration: (iteration: number | null) => void;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Phase</th>
            <th>kp</th>
            <th>ki</th>
            <th>kd</th>
            <th>p1</th>
            <th>p2</th>
            <th>cm gain</th>
            <th><span className="table-head-main">Out L</span><span className="table-head-unit">nH</span></th>
            <th><span className="table-head-main">Eff Lc</span><span className="table-head-unit">nH</span></th>
            <th><span className="table-head-main">PM</span><span className="table-head-unit">deg</span></th>
            <th><span className="table-head-main">fc</span><span className="table-head-unit">kHz</span></th>
            <th><span className="table-head-main">GM</span><span className="table-head-unit">dB</span></th>
            <th><span className="table-head-main">OS</span><span className="table-head-unit">%</span></th>
            <th><span className="table-head-main">US</span><span className="table-head-unit">%</span></th>
            <th><span className="table-head-main">OS Ts</span><span className="table-head-unit">us</span></th>
            <th><span className="table-head-main">US Ts</span><span className="table-head-unit">us</span></th>
            <th>Penalty</th>
          </tr>
        </thead>
        <tbody>
          {history.slice().reverse().map((item) => (
            <tr
              key={item.iteration}
              className={`${item.metrics.passed ? "passed" : ""} ${selectedIteration === item.iteration ? "selected-row" : ""}`}
              onClick={() => onSelectIteration(selectedIteration === item.iteration ? null : item.iteration)}
              title="Show this iteration's transient and Bode result"
            >
              <td>{item.iteration}</td>
              <td>{item.phase}</td>
              <td>{item.candidate?.mod0_kp ?? "--"}</td>
              <td>{item.candidate?.mod0_ki ?? "--"}</td>
              <td>{item.candidate?.mod0_kd ?? "--"}</td>
              <td>{item.candidate?.mod0_kpole1 ?? "--"}</td>
              <td>{item.candidate?.mod0_kpole2 ?? "--"}</td>
              <td>{item.candidate?.mod0_cm_gain ?? "--"}</td>
              <td>{item.candidate ? item.candidate.output_inductance_nh.toFixed(1) : "--"}</td>
              <td>{item.candidate ? item.candidate.effective_lc_inductance_nh.toFixed(1) : "--"}</td>
              <td>{formatMaybeNumber(item.metrics.phase_margin_deg, 1)}</td>
              <td>{formatMaybeNumber(item.metrics.crossover_frequency_hz === null || item.metrics.crossover_frequency_hz === undefined ? null : item.metrics.crossover_frequency_hz / 1000, 2)}</td>
              <td>{formatMaybeNumber(item.metrics.gain_margin_db, 2)}</td>
              <td>{item.metrics.overshoot_pct.toFixed(2)}</td>
              <td>{item.metrics.undershoot_pct.toFixed(2)}</td>
              <td>{formatSettlingUs(item.metrics.overshoot_settling_time_s, item.metrics.settling_time_s).replace(" us", "")}</td>
              <td>{formatSettlingUs(item.metrics.undershoot_settling_time_s, item.metrics.settling_time_s).replace(" us", "")}</td>
              <td>{formatMaybeNumber(displayPenaltyScore(item), 3)}</td>
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

function penaltyOption(history: IterationRecord[]) {
  const plotted = history.map((item) => displayPenaltyScore(item));
  return {
    animation: false,
    tooltip: { trigger: "axis" },
    grid: { left: 48, right: 18, top: 20, bottom: 38 },
    xAxis: { type: "category", data: history.map((item) => item.iteration) },
    yAxis: { type: "value", name: "penalty" },
    series: [{ name: "Penalty", type: "line", smooth: true, data: plotted, areaStyle: {}, lineStyle: { color: "#ea4335" } }]
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
        data: points.map((point, index) => {
          if (!last) return null;
          const crossover = last.metrics.crossover_frequency_hz ?? 100000;
          return 24 - index * 7 + Math.log10(Math.max(crossover, 1) / Math.max(point, 1)) * 2;
        }),
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

function updateSearchParameter(
  setConfig: React.Dispatch<React.SetStateAction<TuningConfig>>,
  field: HardwareSearchKey,
  value: SearchParameter
) {
  setConfig((current) => ({
    ...current,
    search: {
      ...current.search,
      [field]: value
    }
  }));
}

function updateSearchIterationBudget(
  setConfig: React.Dispatch<React.SetStateAction<TuningConfig>>,
  field: "max_coarse_iterations" | "max_refined_iterations",
  value: number
) {
  setConfig((current) => {
    const nextSearch = {
      ...current.search,
      [field]: value
    };
    const maxCoarse = Math.max(1, Math.round(nextSearch.max_coarse_iterations ?? 1));
    const maxRefined = Math.max(0, Math.round(nextSearch.max_refined_iterations ?? 0));
    return {
      ...current,
      search: {
        ...nextSearch,
        max_coarse_iterations: maxCoarse,
        max_refined_iterations: maxRefined,
        max_iterations: Math.max(1, maxCoarse + maxRefined)
      }
    };
  });
}

function searchIterationBudget(search: TuningConfig["search"]) {
  const maxCoarse = searchCoarseBudget(search);
  const maxRefined = searchRefinedBudget(search);
  return Math.max(1, maxCoarse + maxRefined);
}

function searchCoarseBudget(search: TuningConfig["search"]) {
  return Math.max(1, Math.round(search.max_coarse_iterations ?? Math.max(1, Math.round((search.max_iterations ?? 40) / 2))));
}

function searchRefinedBudget(search: TuningConfig["search"]) {
  const maxCoarse = searchCoarseBudget(search);
  return Math.max(0, Math.round(search.max_refined_iterations ?? Math.max(0, (search.max_iterations ?? 40) - maxCoarse)));
}

function countCoarseIterations(history: IterationRecord[]) {
  return history.filter((item) => isCoarsePhase(item.phase)).length;
}

function isCoarsePhase(phase: string | undefined) {
  const value = (phase ?? "").toLowerCase();
  return value === "baseline" || value === "coordinate" || value.includes("coarse");
}

function displayPenaltyScore(record: IterationRecord) {
  const reasons = (record.metrics.pass_reasons ?? []).join(" ").toLowerCase();
  if (reasons.includes("invalid bode") || reasons.includes("duplicate 0 db crossover") || reasons.includes("second 0 db crossover")) {
    return 250;
  }
  if (reasons.includes("protection skipped") || reasons.includes("transient protection skipped")) {
    return 300;
  }
  const scope = record.scope_result ?? {};
  const scopeError = String(scope.error ?? "").toLowerCase();
  if (scope.skipped || scopeError.includes("protection") || scopeError.includes("trip")) {
    return 300;
  }
  const score = Number(record.metrics.score);
  if (Number.isFinite(score) && score >= 1e6) {
    return 300;
  }
  return Number.isFinite(score) ? score : null;
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

