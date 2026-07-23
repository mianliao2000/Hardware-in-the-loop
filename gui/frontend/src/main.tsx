import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactECharts from "echarts-for-react";
import type { ECharts } from "echarts";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Database,
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
  ShieldCheck,
  Trash2,
  XCircle,
  X,
  Sun,
  Zap
} from "lucide-react";
import { archiveCurrentTuningRun, captureScope, deleteTuningRun, getDrlWorkflowStatus, getTuningRuns, getTuningStatus, loadTuningRun, pauseTuning, readFunctionGenerator, readInductance, readPmbusOutput, readPowerSupply, readVout, readXdpOutput, readXdpPid, resetTuning, resumeTuning, runBodeSweep, runDrlWorkflowAction, runSelfTestDevice, saveTuningAnimationGif, sendLlmChat, setFunctionGenerator, setInductance, setPmbusOutput, setPowerSupply, setScopeAcquisition, setVout, setXdpOutput, setXdpPid, startTuning, stepTuning, stopDrlWorkflow, warmScope } from "./api";
import type { AutotuneExperimentConfig, AutotuneGifResponse, AutotuneRunsResponse, BodeSweepConfig, BodeSweepReadback, DrlWorkflowStatus, FunctionGeneratorReadback, InductanceField, InductanceReadback, InstrumentKey, InstrumentTestResult, IterationRecord, LlmChatMessage, PmbusOutputAction, PmbusOutputReadback, PowerSupplyReadback, ScopeCaptureReadback, SearchParameter, SelfTestResponse, TuningConfig, TuningStatus, VoutReadback, XdpOutputAction, XdpOutputReadback, XdpPidReadback } from "./types";
import "./styles.css";

const searchParameter = (center: number, min: number, max: number, points: number): SearchParameter => ({
  center,
  min,
  max,
  points,
  step: points > 1 ? (max - min) / (points - 1) : max - min
});

const AI_COPILOT_HISTORY_KEY = "power-auto-tuner.ai-copilot.messages";
const AI_COPILOT_MODEL_KEY = "power-auto-tuner.ai-copilot.model";
const PANEL_OPEN_STATE_KEY = "power-auto-tuner.panel-open-state.v1";

function readPersistedPanelOpen(key: string, defaultOpen: boolean): boolean {
  try {
    const raw = localStorage.getItem(PANEL_OPEN_STATE_KEY);
    const saved = raw ? JSON.parse(raw) : null;
    return saved && typeof saved[key] === "boolean" ? saved[key] : defaultOpen;
  } catch {
    return defaultOpen;
  }
}

function writePersistedPanelOpen(key: string, open: boolean): void {
  try {
    const raw = localStorage.getItem(PANEL_OPEN_STATE_KEY);
    const saved = raw ? JSON.parse(raw) : {};
    const next = saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {};
    localStorage.setItem(PANEL_OPEN_STATE_KEY, JSON.stringify({ ...next, [key]: open }));
  } catch {
    // Storage may be disabled or full; panel toggles still work for this session.
  }
}

function usePersistentPanelOpen(key: string, defaultOpen: boolean) {
  const [open, setOpen] = useState(() => readPersistedPanelOpen(key, defaultOpen));
  useEffect(() => writePersistedPanelOpen(key, open), [key, open]);
  return [open, setOpen] as const;
}

const LLM_MODEL_OPTIONS = [
  { value: "minimax-m3", label: "Minimax 3" },
  { value: "gemini-3.5-flash", label: "Gemini 3.5 Flash" }
];

const OPTIMIZATION_ALGORITHM_OPTIONS = [
  { value: "heuristic", label: "Grid Search + Heuristic Algorithm" },
  { value: "deep-reinforcement", label: "Deep Reinforcement Learning" }
];

const BASIN_QUALITY_POOL_MULTIPLIER = 6;

const optimizationAlgorithmLabel = (value: string | null | undefined) =>
  OPTIMIZATION_ALGORITHM_OPTIONS.find((option) => option.value === value)?.label
  ?? value
  ?? "Grid Search + Heuristic Algorithm";

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
    settling_time_s: 2e-6,
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
    mod0_kpole1: searchParameter(3, 2, 6, 5),
    mod0_kpole2: searchParameter(3, 2, 6, 5),
    mod0_cm_gain: searchParameter(2, 0, 9, 10),
    mod0_ll_bw: searchParameter(66, 47, 79, 33),
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
  const incomingSearch: Partial<TuningConfig["search"]> = incoming.search ?? {};
  const normalizedSearch = {
    ...fallback.search,
    ...incomingSearch
  } as TuningConfig["search"];
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
  mod0_cm_gain: { min: 0, max: 9 }
};
type HardwareSearchKey =
  | "mod0_kp"
  | "mod0_ki"
  | "mod0_kd"
  | "mod0_kpole1"
  | "mod0_kpole2"
  | "mod0_cm_gain"
  | "mod0_ll_bw"
  | "output_inductance_nh"
  | "effective_lc_inductance_nh";
const hardwareSearchFields: Array<{ key: HardwareSearchKey; label: string; limit: string }> = [
  { key: "mod0_kp", label: "mod0_kp", limit: "0-255" },
  { key: "mod0_ki", label: "mod0_ki", limit: "0-255" },
  { key: "mod0_kd", label: "mod0_kd", limit: "0-255" },
  { key: "mod0_kpole1", label: "mod0_kpole1", limit: "2-6" },
  { key: "mod0_kpole2", label: "mod0_kpole2", limit: "2-6" },
  { key: "mod0_cm_gain", label: "Current Mode Gain", limit: "0-9" },
  { key: "mod0_ll_bw", label: "mod0_ll_ls_bw = mod0_ll_lr_bw", limit: "47-79" },
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
const telemetryDisplayMaxPoints = 6000;
const telemetryUiRefreshIntervalMs = 50;
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
  low_v: 0.1,
  high_v: 1.1,
  pulse_width_s: 5e-6,
  dc_level_v: 0,
  amplitude_vpp: 1,
  offset_v: 0.6,
  phase_deg: 0
};
type AppTab = "autotune" | "manual" | "selftest";
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

function getInitialTheme(): ThemeMode {
  const stored = window.localStorage.getItem(themeStorageKey);
  return stored === "light" || stored === "dark" ? stored : "light";
}

const copy = {
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
};

function appendTelemetrySample(current: VoutReadback[], next: VoutReadback) {
  const timestamp = next.timestamp ?? Date.now() / 1000;
  const sample = { ...next, timestamp };
  const cutoff = timestamp - telemetryHistoryWindowSeconds;
  current.push(sample);
  const oldestTimestamp = current[0]?.timestamp ?? timestamp;
  // Trim in one-second batches instead of shifting a large array for every
  // high-rate sample. The chart axis still clips exactly to the chosen window.
  if (oldestTimestamp < cutoff - 1) {
    let expired = 0;
    while (expired < current.length && (current[expired].timestamp ?? timestamp) < cutoff) expired += 1;
    if (expired > 0) current.splice(0, expired);
  }
  return current;
}

function smoothTelemetryHistory(history: VoutReadback[], windowSeconds = telemetryMovingAverageSeconds) {
  if (history.length === 0) return [];
  let windowStart = 0;
  let voutSum = 0;
  let voutCount = 0;
  let ioutSum = 0;
  let ioutCount = 0;
  let commandSum = 0;
  let commandCount = 0;

  const accumulate = (sample: VoutReadback, direction: 1 | -1) => {
    if (typeof sample.read_vout_v === "number" && Number.isFinite(sample.read_vout_v)) {
      voutSum += direction * sample.read_vout_v;
      voutCount += direction;
    }
    if (typeof sample.read_iout_a === "number" && Number.isFinite(sample.read_iout_a)) {
      ioutSum += direction * sample.read_iout_a;
      ioutCount += direction;
    }
    if (typeof sample.vout_command_v === "number" && Number.isFinite(sample.vout_command_v)) {
      commandSum += direction * sample.vout_command_v;
      commandCount += direction;
    }
  };

  return history.map((sample, index) => {
    const timestamp = sample.timestamp ?? Date.now() / 1000;
    accumulate(sample, 1);
    const cutoff = timestamp - windowSeconds;
    while (windowStart < index && (history[windowStart].timestamp ?? timestamp) < cutoff) {
      accumulate(history[windowStart], -1);
      windowStart += 1;
    }
    return {
      ...sample,
      read_vout_v: voutCount > 0 ? voutSum / voutCount : undefined,
      read_iout_a: ioutCount > 0 ? ioutSum / ioutCount : undefined,
      vout_command_v: commandCount > 0 ? commandSum / commandCount : undefined
    };
  });
}

function decimateTelemetryHistory(history: VoutReadback[], maxPoints = telemetryDisplayMaxPoints) {
  if (history.length <= maxPoints || maxPoints < 2) return history;
  const output: VoutReadback[] = [];
  const step = (history.length - 1) / (maxPoints - 1);
  let previousIndex = -1;
  for (let index = 0; index < maxPoints; index += 1) {
    const sourceIndex = Math.min(history.length - 1, Math.round(index * step));
    if (sourceIndex !== previousIndex) output.push(history[sourceIndex]);
    previousIndex = sourceIndex;
  }
  return output;
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
  const limits = name === "mod0_cm_gain"
    ? defaultXdpPidLimits[name]
    : field ? { min: field.min, max: field.max } : defaultXdpPidLimits[name];
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
      mod0_cm_gain: searchParameterWithCenter(
        config.search.mod0_cm_gain,
        Math.min(9, Math.max(0, Math.round(xdpPidRequest.mod0_cm_gain ?? defaultManualXdpPidRequest.mod0_cm_gain)))
      ),
      // This search-only value intentionally has no independent LS/LR manual controls.
      mod0_ll_bw: config.search.mod0_ll_bw,
      output_inductance_nh: searchParameterWithCenter(config.search.output_inductance_nh, inductanceRequest.output_nh),
      effective_lc_inductance_nh: searchParameterWithCenter(config.search.effective_lc_inductance_nh, inductanceRequest.effective_lc_nh)
    }
  };
}

function buildAutotuneExperiment(
  voutRequest: { address: string; page: number; adapter: string; voltage: number },
  bodeSweepConfig: BodeSweepConfig,
  settings: ManualExperimentSettings,
  analysisSelection: { bode: boolean; transient: boolean },
  optimizationAlgorithm: string,
  drlModelId = "",
  drlEpisodeBudget = 15
): AutotuneExperimentConfig {
  return {
    board_address: voutRequest.address,
    board_page: voutRequest.page,
    board_adapter: voutRequest.adapter,
    response_channel: "CH3",
    enable_bode_analysis: analysisSelection.bode,
    enable_transient_analysis: analysisSelection.transient,
    optimization_algorithm: optimizationAlgorithm,
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
    ignore_pass_until_max_iterations: true,
    drl_model_id: drlModelId,
    drl_episode_budget: Math.max(1, Math.round(drlEpisodeBudget)),
    drl_hardware_protection_mode: true
  };
}

function lastHistoryIteration(history: IterationRecord[]) {
  return history.reduce((latest, record) => Math.max(latest, Number(record.iteration) || 0), 0);
}

function recordRenderSignature(record: IterationRecord | null | undefined) {
  if (!record) return null;
  const scope = record.scope_result as Record<string, unknown> | null | undefined;
  const bode = record.bode_result as Record<string, unknown> | null | undefined;
  const compactAssetSignature = (value: unknown) => {
    if (typeof value !== "string") return value;
    return `${value.length}:${value.slice(0, 24)}:${value.slice(-24)}`;
  };
  return [
    record.iteration,
    record.phase,
    displayPenaltyScore(record),
    displayObjectiveScore(record),
    record.bandwidth_bonus,
    record.metrics.passed,
    record.duration_s,
    scope?.capture_id,
    compactAssetSignature(scope?.scope_png),
    bode?.sweep_id,
    compactAssetSignature(bode?.bode_png)
  ];
}

function tuningStatusRenderSignature(status: TuningStatus) {
  return JSON.stringify({
    state: status.state,
    message: status.message,
    historyToken: status.history_token,
    historyTotal: status.history_total ?? status.history.length,
    historyLast: status.history_last_iteration ?? lastHistoryIteration(status.history),
    current: recordRenderSignature(status.current),
    best: recordRenderSignature(status.best),
    recommendations: (status.recommendations ?? []).map(recordRenderSignature),
    config: status.config,
    experiment: status.experiment,
    pidProgramming: status.pid_programming,
    run: status.run
  });
}

function drlStatusRenderSignature(status: DrlWorkflowStatus) {
  return JSON.stringify(status);
}

function mergeIncrementalTuningStatus(previous: TuningStatus | null, next: TuningStatus): TuningStatus | null {
  if (!next.history_delta) return next;
  if (!previous || !next.history_token || previous.history_token !== next.history_token) return null;
  const previousLast = lastHistoryIteration(previous.history ?? []);
  if (next.history_after_iteration !== previousLast) return null;

  if ((next.history ?? []).length === 0) {
    const merged = { ...next, history: previous.history, history_delta: false };
    return tuningStatusRenderSignature(previous) === tuningStatusRenderSignature(merged)
      ? previous
      : merged;
  }

  const byIteration = new Map<number, IterationRecord>();
  for (const record of previous.history ?? []) byIteration.set(record.iteration, record);
  for (const record of next.history ?? []) byIteration.set(record.iteration, record);
  const history = [...byIteration.values()].sort((left, right) => left.iteration - right.iteration);
  const expectedTotal = next.history_total ?? history.length;
  const expectedLast = next.history_last_iteration ?? lastHistoryIteration(history);
  if (history.length !== expectedTotal || lastHistoryIteration(history) !== expectedLast) return null;
  return { ...next, history, history_delta: false };
}

function App() {
  const [config, setConfig] = useState<TuningConfig>(() => cloneDefaultConfig());
  const [status, setStatus] = useState<TuningStatus | null>(null);
  const statusRef = useRef<TuningStatus | null>(null);
  const [activeTab, setActiveTab] = useState<AppTab>(getInitialTab);
  const [themeMode, setThemeMode] = useState<ThemeMode>(getInitialTheme);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const statusRefreshFailures = useRef(0);
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
  const [optimizationAlgorithm, setOptimizationAlgorithm] = useState("heuristic");
  const [drlStatus, setDrlStatus] = useState<DrlWorkflowStatus | null>(null);
  const drlStatusRef = useRef<DrlWorkflowStatus | null>(null);
  const [autotuneRuns, setAutotuneRuns] = useState<AutotuneRunsResponse | null>(null);
  const [selectedAutotuneRun, setSelectedAutotuneRun] = useState("");
  const [autotuneArchiveStatus, setAutotuneArchiveStatus] = useState("");
  const [autotuneGif, setAutotuneGif] = useState<AutotuneGifResponse | null>(null);
  const [autotuneGifFrameDurationS, setAutotuneGifFrameDurationS] = useState("0.1");
  const viewingLoadedRun = useRef(false);
  const [manualExperimentSettings, setManualExperimentSettings] = useState<ManualExperimentSettings>(defaultManualExperimentSettings);
  const [telemetryHistory, setTelemetryHistory] = useState<VoutReadback[]>([]);
  const telemetryHistoryRef = useRef<VoutReadback[]>([]);
  const telemetryPendingReadback = useRef<VoutReadback | null>(null);
  const telemetryPublishTimer = useRef<number | null>(null);
  const telemetryLastPublishMs = useRef(0);
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
  const t = copy;

  const commitStatus = (next: TuningStatus) => {
    if (statusRef.current === next) return;
    statusRef.current = next;
    setStatus(next);
  };

  const commitDrlStatus = (next: DrlWorkflowStatus) => {
    const previous = drlStatusRef.current;
    if (previous && drlStatusRenderSignature(previous) === drlStatusRenderSignature(next)) return;
    drlStatusRef.current = next;
    setDrlStatus(next);
  };

  const flushTelemetryDisplay = () => {
    telemetryPublishTimer.current = null;
    const pending = telemetryPendingReadback.current;
    telemetryPendingReadback.current = null;
    if (pending) setVoutState(pending);
    setTelemetryHistory([...telemetryHistoryRef.current]);
    telemetryLastPublishMs.current = performance.now();
  };

  const publishTelemetryReadback = (next: VoutReadback, deferRender = false) => {
    if (next.ok && next.read_vout_v !== undefined) {
      appendTelemetrySample(telemetryHistoryRef.current, next);
    }
    telemetryPendingReadback.current = next;
    if (!deferRender) {
      if (telemetryPublishTimer.current !== null) {
        window.clearTimeout(telemetryPublishTimer.current);
        telemetryPublishTimer.current = null;
      }
      flushTelemetryDisplay();
      return;
    }
    if (telemetryPublishTimer.current !== null) return;
    const elapsed = performance.now() - telemetryLastPublishMs.current;
    const delay = Math.max(0, telemetryUiRefreshIntervalMs - elapsed);
    telemetryPublishTimer.current = window.setTimeout(flushTelemetryDisplay, delay);
  };

  const refresh = async () => {
    if (viewingLoadedRun.current) return;
    try {
      const current = statusRef.current;
      const afterIteration = current ? lastHistoryIteration(current.history ?? []) : undefined;
      let next = await getTuningStatus(afterIteration, current?.history_token);
      if (viewingLoadedRun.current) return;
      let merged = mergeIncrementalTuningStatus(current, next);
      if (merged === null) {
        next = await getTuningStatus();
        if (viewingLoadedRun.current) return;
        merged = next;
      }
      if (merged === current) {
        statusRefreshFailures.current = 0;
        setConnectionError("");
        return;
      }
      commitStatus(merged);
      if (["drl-collection", "deep-reinforcement", "safe-sac"].includes(merged.experiment?.optimization_algorithm ?? "")) {
        setConfig(normalizeTuningConfig(merged.config));
      }
      statusRefreshFailures.current = 0;
      setConnectionError("");
    } catch (exc) {
      statusRefreshFailures.current += 1;
      if (statusRefreshFailures.current >= 3) {
        setConnectionError(String(exc));
      }
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
      console.warn("Could not refresh the result library:", exc);
    }
  };

  const refreshDrlStatus = async () => {
    try {
      const next = await getDrlWorkflowStatus();
      commitDrlStatus(next);
    } catch (exc) {
      console.warn("Could not refresh the DRL workflow status:", exc);
    }
  };

  useEffect(() => {
    let cancelled = false;
    let statusTimer: number | null = null;
    const pollStatus = async () => {
      await refresh();
      if (cancelled) return;
      const delayMs = statusRef.current?.state === "running" ? 4000 : 5000;
      statusTimer = window.setTimeout(pollStatus, delayMs);
    };
    pollStatus();
    refreshAutotuneRuns();
    return () => {
      cancelled = true;
      if (statusTimer !== null) window.clearTimeout(statusTimer);
      if (telemetryPublishTimer.current !== null) {
        window.clearTimeout(telemetryPublishTimer.current);
        telemetryPublishTimer.current = null;
      }
    };
  }, []);

  const drlPollingActive = Boolean(drlStatus?.busy)
    || ["collecting", "training", "validating"].includes(drlStatus?.state ?? "");

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    const poll = async () => {
      await refreshDrlStatus();
      if (!cancelled) timer = window.setTimeout(poll, drlPollingActive ? 4000 : 10000);
    };
    poll();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [drlPollingActive]);

  useEffect(() => {
    if (["collecting", "collection_complete", "validating", "hardware_ready", "validation_failed"].includes(drlStatus?.state ?? "")) {
      refreshAutotuneRuns();
    }
  }, [drlStatus?.state]);

  useEffect(() => {
    window.localStorage.setItem(tabStorageKey, activeTab);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", activeTab);
    window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
  }, [activeTab]);

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
      publishTelemetryReadback(nextVout);
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

  const runAction = async (action: "start" | "pause" | "resume" | "step") => {
    setError("");
    setConnectionError("");
    try {
      if (action === "pause" || action === "resume") {
        const resumeRun = action === "resume" && viewingLoadedRun.current ? status?.run : undefined;
        viewingLoadedRun.current = false;
        const next = action === "pause"
          ? await pauseTuning()
          : await resumeTuning(resumeRun?.run_id, resumeRun?.kind);
        commitStatus(next);
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
      const experiment = buildAutotuneExperiment(
        voutRequest,
        bodeSweepConfig,
        manualExperimentSettings,
        autotuneAnalysisSelection,
        optimizationAlgorithm,
        drlStatus?.model_id ?? "",
        searchIterationBudget(runConfig.search)
      );
      const next =
        action === "start" ? await startTuning(runConfig, experiment) : await stepTuning(runConfig, experiment);
      commitStatus(next);
      setConfig(normalizeTuningConfig(next.config));
      refreshAutotuneRuns();
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const runDrlAction = async (action: "collect" | "train" | "validate" | "stop") => {
    try {
      if (action === "stop") {
        commitDrlStatus(await stopDrlWorkflow());
        await refresh();
        await refreshAutotuneRuns();
        setError("");
        return;
      }
      if (!autotuneAnalysisSelection.transient || !autotuneAnalysisSelection.bode) {
        throw new Error("Safe SAC collection and validation require both Transient Analysis and Bode Analysis.");
      }
      let runConfig = config;
      if (action !== "train") {
        viewingLoadedRun.current = false;
        const baseline = await loadAutotuneHardwareBaseline();
        const targetVout = snapVoutToRegister(config.targets.vout_target_v, voutExponentFromReadback(baseline.voutReadback));
        runConfig = withManualSearchCenters(
          { ...config, targets: { ...config.targets, vout_target_v: targetVout } },
          baseline.pidRequest,
          baseline.inductanceValues
        );
        setConfig(runConfig);
      }
      const experiment = buildAutotuneExperiment(
        voutRequest,
        bodeSweepConfig,
        manualExperimentSettings,
        { transient: true, bode: true },
        "deep-reinforcement",
        drlStatus?.model_id ?? "",
        searchIterationBudget(runConfig.search)
      );
      const next = await runDrlWorkflowAction(action, runConfig, experiment);
      commitDrlStatus(next);
      setError("");
      await refreshAutotuneRuns();
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
      commitStatus(next);
      refreshAutotuneRuns();
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const readBoardVout = async (deferRender = false) => {
    try {
      const next = await readVout(voutRequest.address, voutRequest.page, voutRequest.adapter);
      publishTelemetryReadback(next, deferRender);
      setError("");
    } catch (exc) {
      setError(String(exc));
    }
  };

  const writeBoardVout = async () => {
    try {
      const requestedVoltage = snapVoutToRegister(voutRequest.voltage, voutExponentFromReadback(vout));
      const next = await setVout(voutRequest.address, voutRequest.page, voutRequest.adapter, requestedVoltage);
      publishTelemetryReadback(next);
      setVoutRequest((current) => ({ ...current, voltage: requestedVoltage }));
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
      publishTelemetryReadback(refreshed);
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
      publishTelemetryReadback(refreshed);
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
        await readBoardVout(true);
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
        publishTelemetryReadback(nextVout);
        setVoutRequest((current) => ({ ...current, voltage: snapshot.voltage }));
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
      publishTelemetryReadback(refreshed);
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
      commitStatus(loaded);
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
          <img className="google-mark" src="/google-cloud-mark.png" alt="Google Cloud logo" />
          <div>
            <h1 className="google-title">
              <span className="google-cloud-wordmark" aria-label="Google Cloud">
                <span className="google-wordmark" aria-hidden="true">
                  <span className="google-blue">G</span>
                  <span className="google-red">o</span>
                  <span className="google-yellow">o</span>
                  <span className="google-blue">g</span>
                  <span className="google-green">l</span>
                  <span className="google-red">e</span>
                </span>
                <span className="cloud-title" aria-hidden="true">Cloud</span>
              </span>{" "}
              <span className="product-title">Power AI Auto Tuner (V1.0)</span>
            </h1>
            <p>{t.platform} ({t.copyright})</p>
            <p>{t.author}</p>
          </div>
        </div>
        <div className="topbar-actions">
          <HeaderToggles
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

      {(error || connectionError) && (
        <div className="banner error">
          <AlertTriangle size={18} />
          <span>{error || connectionError}</span>
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
        optimizationAlgorithm={optimizationAlgorithm}
        setOptimizationAlgorithm={setOptimizationAlgorithm}
        drlStatus={drlStatus}
        runDrlAction={runDrlAction}
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
        optimizationAlgorithm={optimizationAlgorithm}
      />
    </main>
  );
}

function LlmAssistantWidget({
  activeTab,
  status,
  config,
  optimizationAlgorithm
}: {
  activeTab: AppTab;
  status: TuningStatus | null;
  config: TuningConfig;
  optimizationAlgorithm: string;
}) {
  const greeting = "Hi, I can help explain this GUI, the Auto-Tune flow, Manual Tuning, Self Testing, and the Bode 100 / Scope / Function Generator / PMBus panels.";
  const initialMessages = () => {
    try {
      const stored = localStorage.getItem(AI_COPILOT_HISTORY_KEY);
      const parsed = stored ? JSON.parse(stored) : null;
      if (Array.isArray(parsed) && parsed.length) {
        const clean = parsed
          .filter((message) => message && ["user", "assistant", "system"].includes(message.role) && typeof message.content === "string")
          .slice(-40)
          .map((message) => ({ role: message.role, content: message.content }));
        if (clean.length) return clean as LlmChatMessage[];
      }
    } catch {
      localStorage.removeItem(AI_COPILOT_HISTORY_KEY);
    }
    return [{ role: "assistant", content: greeting }] as LlmChatMessage[];
  };
  const [open, setOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [input, setInput] = useState("");
  const [panelSize, setPanelSize] = useState({ width: 460, height: 540 });
  const [panelCursor, setPanelCursor] = useState<string | undefined>();
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState<LlmChatMessage[]>(initialMessages);
  const [modelChoice, setModelChoice] = useState(() => {
    try {
      const stored = localStorage.getItem(AI_COPILOT_MODEL_KEY);
      if (stored && LLM_MODEL_OPTIONS.some((option) => option.value === stored)) return stored;
    } catch {
      // Ignore storage failures; the default model still works.
    }
    return "gemini-3.5-flash";
  });
  const messagesRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: "smooth" });
  }, [open, messages, sending]);

  useEffect(() => {
    try {
      localStorage.setItem(AI_COPILOT_HISTORY_KEY, JSON.stringify(messages.slice(-40)));
    } catch {
      // Ignore storage quota/private-mode failures; chat still works for this session.
    }
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem(AI_COPILOT_MODEL_KEY, modelChoice);
    } catch {
      // Ignore storage failures; the selected model still works for this session.
    }
  }, [modelChoice]);

  const clearHistory = () => {
    const resetMessages = [{ role: "assistant", content: greeting }] as LlmChatMessage[];
    setMessages(resetMessages);
    localStorage.removeItem(AI_COPILOT_HISTORY_KEY);
  };

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
      optimization_algorithm: optimizationAlgorithm,
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
      const response = await sendLlmChat(nextMessages, context, modelChoice);
      if (!response.completion?.complete) {
        throw new Error(
          "The backend returned an unverified response. Restart gui/server.py to enable the complete-response protocol."
        );
      }
      setMessages([...nextMessages, { role: "assistant", content: response.reply ?? "No reply." }]);
    } catch (exc) {
      setMessages([...nextMessages, { role: "assistant", content: `LLM API error: ${String(exc)}` }]);
    } finally {
      setSending(false);
    }
  };

  const startPanelResize = (direction: "left" | "top" | "top-left", event: React.MouseEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const startY = event.clientY;
    const startWidth = panelSize.width;
    const startHeight = panelSize.height;
    const maxWidth = Math.max(320, window.innerWidth - 44);
    const maxHeight = Math.max(360, window.innerHeight - 120);

    const handlePointerMove = (moveEvent: MouseEvent) => {
      const nextWidth = direction.includes("left")
        ? Math.min(maxWidth, Math.max(320, startWidth + startX - moveEvent.clientX))
        : startWidth;
      const nextHeight = direction.includes("top")
        ? Math.min(maxHeight, Math.max(360, startHeight + startY - moveEvent.clientY))
        : startHeight;
      setPanelSize({ width: nextWidth, height: nextHeight });
    };

    const stopResize = () => {
      window.removeEventListener("mousemove", handlePointerMove);
      window.removeEventListener("mouseup", stopResize);
    };

    window.addEventListener("mousemove", handlePointerMove);
    window.addEventListener("mouseup", stopResize);
  };

  const getPanelResizeDirection = (event: React.MouseEvent<HTMLElement>) => {
    if (fullscreen) return null;
    const bounds = event.currentTarget.getBoundingClientRect();
    const nearLeft = event.clientX - bounds.left <= 12;
    const nearTop = event.clientY - bounds.top <= 12;
    if (nearLeft && nearTop) return "top-left";
    if (nearLeft) return "left";
    if (nearTop) return "top";
    return null;
  };

  const handlePanelMouseMove = (event: React.MouseEvent<HTMLElement>) => {
    const direction = getPanelResizeDirection(event);
    setPanelCursor(direction === "left" ? "ew-resize" : direction === "top" ? "ns-resize" : direction ? "nwse-resize" : undefined);
  };

  const handlePanelMouseDown = (event: React.MouseEvent<HTMLElement>) => {
    const direction = getPanelResizeDirection(event);
    if (direction) startPanelResize(direction, event);
  };

  return (
    <div className="llm-chat-root">
      {open && (
        <section
          className={`llm-chat-panel${fullscreen ? " is-fullscreen" : ""}`}
          style={fullscreen ? undefined : { width: `${panelSize.width}px`, height: `${panelSize.height}px`, cursor: panelCursor }}
          aria-label="AI Copilot chat"
          onMouseMove={handlePanelMouseMove}
          onMouseLeave={() => setPanelCursor(undefined)}
          onMouseDown={handlePanelMouseDown}
        >
          {!fullscreen && (
            <>
              <button
                className="llm-panel-resize-handle is-left"
                aria-label="Resize AI Copilot wider from the left"
                type="button"
                onMouseDown={(event) => startPanelResize("left", event)}
              />
              <button
                className="llm-panel-resize-handle is-top"
                aria-label="Resize AI Copilot taller from the top"
                type="button"
                onMouseDown={(event) => startPanelResize("top", event)}
              />
              <button
                className="llm-panel-resize-handle is-top-left"
                aria-label="Resize AI Copilot from the top left"
                type="button"
                onMouseDown={(event) => startPanelResize("top-left", event)}
              />
            </>
          )}
          <div className="llm-chat-header">
            <div>
              <h2>AI Copilot</h2>
            </div>
            <div className="llm-chat-header-actions">
              <select
                className="llm-model-select"
                value={modelChoice}
                onChange={(event) => setModelChoice(event.target.value)}
                aria-label="AI Copilot model"
                disabled={sending}
              >
                {LLM_MODEL_OPTIONS.map((option) => (
                  <option value={option.value} key={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <button
                className="llm-chat-icon-button llm-clear-button"
                onClick={clearHistory}
                aria-label="Clear AI Copilot history"
                title="Clear history"
                type="button"
                disabled={sending}
              >
                <Trash2 size={16} />
              </button>
              <button
                className="llm-chat-icon-button"
                onClick={() => setFullscreen((current) => !current)}
                aria-label={fullscreen ? "Exit fullscreen AI Copilot" : "Open AI Copilot fullscreen"}
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
                aria-label="Close AI Copilot"
              >
                <X size={18} />
              </button>
            </div>
          </div>
          <div className="llm-chat-messages" ref={messagesRef}>
            {messages.map((message, index) => (
              <div className={`llm-message ${message.role}`} key={`${message.role}-${index}`}>
                <span>{message.role === "user" ? "You" : "Assistant"}</span>
                <MarkdownMessage content={message.content} />
              </div>
            ))}
            {sending && (
              <div className="llm-message assistant">
                <span>Assistant</span>
                <MarkdownMessage content="Thinking..." />
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
              placeholder="Ask and Revise the Project"
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
        <span>AI Copilot</span>
      </button>
    </div>
  );
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="llm-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          )
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function HeaderToggles({
  themeMode,
  setThemeMode
}: {
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
            transient penalty = excess OS [%] + excess US [%] + 10 x excess OS settling [us] + 10 x excess US settling [us]
          </div>
          <div className="formula-box">
            passing settling reward = 10 x OS settling headroom [us] + 10 x US settling headroom [us]
          </div>
          <div className="formula-box">
            Bode penalty = 1.5 x phase-margin shortage [deg] + 0.5 x crossover upper-limit excess [%]
          </div>
          <p>
            Default limits: OS at or below 3%, US at or below 3%, OS/US settling at or below 2 us, phase margin at or above
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
            text="Each load step has its own settling time in microseconds. The analyzer uses the filtered CH3 response, a bounded settling band, and a stable dwell check. The default target is 2 us for both OS settling and US settling."
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

function DrlControl({
  status,
  tuningState,
  onAction
}: {
  status: DrlWorkflowStatus | null;
  tuningState?: TuningStatus["state"];
  onAction: (action: "collect" | "train" | "validate" | "stop") => Promise<void>;
}) {
  const busy = Boolean(status?.busy) || tuningState === "running";
  const collectionReady = Boolean(
    status
    && status.collection_finished
    && status.collection_total > 0
    && status.collection_completed >= status.collection_total
    && !status.resume_available
    && !["preparing_collection", "collecting", "paused"].includes(status.state)
  );
  const validationReady = status?.model_status === "ready_for_validation" || status?.model_status === "hardware_ready";
  const progress = Math.round(Math.max(0, Math.min(1, status?.progress ?? 0)) * 100);
  const dependencyReady = Boolean(status?.dependency?.ok);
  return (
    <div className="drl-control" aria-live="polite">
      <div className="drl-control-heading">
        <BrainCircuit size={16} />
        <span>DRL Control</span>
        <span className={`drl-state ${status?.state ?? "idle"}`}>{status?.state?.replaceAll("_", " ") ?? "loading"}</span>
      </div>
      <div className="drl-facts">
        <span>Dataset</span>
        <strong title="Source records / usable six-action samples">
          {status?.dataset_source_count ?? status?.dataset_count ?? 0} / {status?.dataset_count ?? 0}
        </strong>
        <span>Collect</span>
        <strong>{status?.collection_completed ?? 0} / {status?.collection_total ?? 240}</strong>
        <span>Model</span>
        <strong title={status?.model_id ?? "No model"}>{status?.model_id ? status.model_id.replace("safe_sac_", "") : "Missing"}</strong>
        <span>Compatible</span>
        <strong>{status?.model_id ? (status.model_compatible ? "Yes" : "No") : "-"}</strong>
        <span>Validate</span>
        <strong>{status?.validation_completed ?? 0} / {status?.validation_total ?? 60}</strong>
      </div>
      <div className="drl-progress" aria-label={`DRL workflow progress ${progress}%`}>
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="drl-actions">
        <button type="button" onClick={() => onAction("collect")} disabled={busy || !dependencyReady} title="Collect the guarded 240-point hardware dataset">
          <Database size={14} /> Collect
        </button>
        <button type="button" onClick={() => onAction("train")} disabled={busy || !collectionReady || !dependencyReady} title="Train the surrogate ensemble and Safe SAC policy">
          <BrainCircuit size={14} /> Train
        </button>
        <button type="button" onClick={() => onAction("validate")} disabled={busy || !validationReady || !status?.model_compatible} title="Run four guarded hardware validation episodes">
          <ShieldCheck size={14} /> Validate
        </button>
        <button type="button" onClick={() => onAction("stop")} disabled={!status?.busy && !status?.resume_available} title="Stop the DRL workflow">
          <StopCircle size={14} /> Stop
        </button>
      </div>
      <p className={`drl-message${status?.error ? " error" : ""}`}>
        {!dependencyReady
          ? status?.dependency?.error || "The optional CPU ML environment is not ready."
          : !status?.model_id && status?.state === "idle"
            ? "No usable model. Complete Collect and Train before Start Auto-Tune can use Safe SAC."
            : status?.message || "DRL workflow is idle."}
      </p>
      {status?.validation_result && (
        <p className={status.validation_result.accepted ? "drl-result accepted" : "drl-result rejected"}>
          Hardware episodes: {status.validation_result.episodes_succeeded}/{status.validation_result.episodes_completed} successful
        </p>
      )}
    </div>
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
  optimizationAlgorithm,
  setOptimizationAlgorithm,
  drlStatus,
  runDrlAction,
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
  optimizationAlgorithm: string;
  setOptimizationAlgorithm: React.Dispatch<React.SetStateAction<string>>;
  drlStatus: DrlWorkflowStatus | null;
  runDrlAction: (action: "collect" | "train" | "validate" | "stop") => Promise<void>;
  experimentSettings: ManualExperimentSettings;
  setExperimentSettings: React.Dispatch<React.SetStateAction<ManualExperimentSettings>>;
  runAction: (action: "start" | "pause" | "resume" | "step") => Promise<void>;
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
  const [livePlayback, setLivePlayback] = useState(false);
  const historyPreviewTimerRef = useRef<number | null>(null);
  const selectedRecord = selectedIteration === null ? null : history.find((item) => item.iteration === selectedIteration) ?? null;
  const visibleRecord = selectedRecord ?? current;
  const recommendedRecords = useMemo(
    () => selectRecommendedRecords(status?.recommendations, history, 5),
    [history, status?.recommendations]
  );
  const selectIterationForPreview = useCallback((iteration: number | null) => {
    if (historyPreviewTimerRef.current !== null) {
      window.clearTimeout(historyPreviewTimerRef.current);
      historyPreviewTimerRef.current = null;
    }
    setSelectedIteration(iteration);
    if (iteration !== null && status?.state === "running") {
      historyPreviewTimerRef.current = window.setTimeout(() => {
        setSelectedIteration((currentIteration) => currentIteration === iteration ? null : currentIteration);
        historyPreviewTimerRef.current = null;
      }, 5000);
    }
  }, [status?.state]);
  useEffect(() => {
    if (selectedIteration !== null && !history.some((item) => item.iteration === selectedIteration)) {
      setSelectedIteration(null);
    }
  }, [history, selectedIteration]);
  useEffect(() => {
    if (status?.state === "running" || historyPreviewTimerRef.current === null) return;
    window.clearTimeout(historyPreviewTimerRef.current);
    historyPreviewTimerRef.current = null;
  }, [status?.state]);
  useEffect(() => () => {
    if (historyPreviewTimerRef.current !== null) {
      window.clearTimeout(historyPreviewTimerRef.current);
    }
  }, []);
  useEffect(() => {
    if (selectedIteration === null || history.length === 0) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, button, [contenteditable='true']")) return;
      const ordered = history.slice().reverse();
      const currentIndex = ordered.findIndex((item) => item.iteration === selectedIteration);
      if (currentIndex < 0) return;
      event.preventDefault();
      const direction = event.key === "ArrowUp" ? -1 : 1;
      const nextIndex = Math.max(0, Math.min(ordered.length - 1, currentIndex + direction));
      if (nextIndex === currentIndex) return;
      selectIterationForPreview(ordered[nextIndex].iteration);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [history, selectIterationForPreview, selectedIteration]);
  useEffect(() => {
    if (status?.state === "running" || history.length < 1) {
      setLivePlayback(false);
    }
  }, [history.length, status?.state]);
  useEffect(() => {
    if (!livePlayback || history.length < 1) return;
    if (history.length === 1) {
      setSelectedIteration(history[0].iteration);
      return;
    }
    const parsedDurationS = Number.parseFloat(gifFrameDurationS);
    const safeDurationS = Number.isFinite(parsedDurationS) ? parsedDurationS : 0.1;
    const frameMs = Math.round(Math.max(0.05, Math.min(5, safeDurationS)) * 1000);
    const intervalId = window.setInterval(() => {
      setSelectedIteration((currentIteration) => {
        const currentIndex = currentIteration === null
          ? -1
          : history.findIndex((item) => item.iteration === currentIteration);
        const nextIndex = currentIndex < 0 ? 0 : (currentIndex + 1) % history.length;
        return history[nextIndex].iteration;
      });
    }, frameMs);
    return () => window.clearInterval(intervalId);
  }, [gifFrameDurationS, history, livePlayback]);
  const toggleLivePlayback = () => {
    if (livePlayback) {
      setLivePlayback(false);
      return;
    }
    if (history.length === 0) return;
    const displayedIteration = visibleRecord?.iteration;
    const playbackStart = displayedIteration !== undefined
      && history.some((item) => item.iteration === displayedIteration)
      ? displayedIteration
      : history[history.length - 1].iteration;
    setSelectedIteration(playbackStart);
    setLivePlayback(true);
  };
  const setFgConfig = (updater: React.SetStateAction<typeof defaultFunctionGeneratorConfig>) => {
    setExperimentSettings((current) => {
      const nextFgConfig = typeof updater === "function" ? updater(current.fgConfig) : updater;
      return { ...current, fgConfig: nextFgConfig };
    });
  };
  return (
      <div className="workspace">
        <aside className="control-rail">
          <Panel title="Run Control" persistenceKey="autotune-run-control" icon={<Activity size={17} />}>
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
              <div className="autotune-algorithm-field">
                <select
                  className="autotune-algorithm-select"
                  aria-label="Optimization algorithm"
                  title="Optimization algorithm"
                  value={optimizationAlgorithm}
                  onChange={(event) => setOptimizationAlgorithm(event.target.value)}
                  disabled={status?.state === "running"}
                >
                  {OPTIMIZATION_ALGORITHM_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              {optimizationAlgorithm === "deep-reinforcement" && (
                <DrlControl status={drlStatus} tuningState={status?.state} onAction={runDrlAction} />
              )}
              {optimizationAlgorithm !== "deep-reinforcement" && (
                <div className="autotune-grid-budget-fields">
                  <DeferredNumberField
                    label="Corase"
                    value={config.search.max_coarse_iterations ?? Math.max(1, Math.round((config.search.max_iterations ?? 40) / 2))}
                    integer
                    min={1}
                    onCommit={(value) => updateSearchIterationBudget(setConfig, "max_coarse_iterations", Math.max(1, Math.round(value)))}
                  />
                  <DeferredNumberField
                    label="Refined"
                    value={config.search.max_refined_iterations ?? Math.max(0, (config.search.max_iterations ?? 40) - Math.max(1, Math.round((config.search.max_iterations ?? 40) / 2)))}
                    integer
                    min={0}
                    onCommit={(value) => updateSearchIterationBudget(setConfig, "max_refined_iterations", Math.max(0, Math.round(value)))}
                  />
                </div>
              )}
              <button className="autotune-control-button start" onClick={() => runAction("start")} disabled={status?.state === "running" || !canRunAnalysis}>
                Start Auto-Tune
              </button>
              <button className="autotune-control-button iterate" onClick={() => runAction("step")} disabled={status?.state === "running" || !canRunAnalysis}>
                Run Single Iteration
              </button>
              <div className="autotune-control-grid">
                <button className="autotune-control-button" onClick={() => runAction("resume")} disabled={status?.state !== "paused" && status?.state !== "stopped"}>
                  Resume
                </button>
                <button className="autotune-control-button" onClick={() => runAction("pause")} disabled={status?.state !== "running"}>
                  Stop
                </button>
              </div>
              <div className="gif-save-row">
                <button className="autotune-control-button gif-save-button" onClick={saveGif} disabled={status?.state === "running" || history.length < 1}>
                  Save GIF
                </button>
                <button
                  className={`autotune-control-button gif-save-button live-playback-button${livePlayback ? " active" : ""}`}
                  onClick={toggleLivePlayback}
                  disabled={status?.state === "running" || history.length < 1}
                  aria-pressed={livePlayback}
                  title={livePlayback ? "Stop live iteration playback" : "Start live iteration playback"}
                >
                  Live
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

          <Panel title="Function Generator" persistenceKey="autotune-function-generator" icon={<Activity size={17} />}>
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
            {hardwareSearchFields.map((field) => (
              <SearchParameterField
                key={field.key}
                label={field.label}
                limit={field.limit}
                value={config.search[field.key] ?? defaultConfig.search[field.key]}
                integer={field.key.startsWith("mod0_")}
                singleLineTitle={field.key === "mod0_ll_bw"}
                onChange={(value) => {
                  updateSearchParameter(setConfig, field.key, value);
                  if (field.key === "output_inductance_nh") {
                    setInductanceRequest((current) => ({ ...current, output_nh: value.center }));
                  } else if (field.key === "effective_lc_inductance_nh") {
                    setInductanceRequest((current) => ({ ...current, effective_lc_nh: value.center }));
                  } else if (field.key === "mod0_ll_bw") {
                    // Search runner writes this shared register; it is not a manual PID field.
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
            <PenaltyTrendPanel
              history={history}
              selectedIteration={selectedIteration}
              onSelectIteration={selectIterationForPreview}
              running={status?.state === "running"}
            />
          </div>
          <IterationHistoryPanel
            history={history}
            selectedIteration={selectedIteration}
            onSelectIteration={selectIterationForPreview}
          />
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
            selectedAlgorithm={optimizationAlgorithm}
          />
          <RecommendedResultsPanel
            records={recommendedRecords}
            selectedIteration={selectedIteration}
            onPreview={selectIterationForPreview}
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
  setRealTimeWriting: (enabled: boolean) => void;
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
  labels: typeof copy;
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
  const [scopeWebPlotOpen, setScopeWebPlotOpen] = usePersistentPanelOpen("manual-scope-web-plot", true);
  const [scopePngPlotOpen, setScopePngPlotOpen] = usePersistentPanelOpen("manual-scope-png-plot", true);
  const [bodeWebPlotOpen, setBodeWebPlotOpen] = usePersistentPanelOpen("manual-bode-web-plot", true);
  const [bodePngPlotOpen, setBodePngPlotOpen] = usePersistentPanelOpen("manual-bode-png-plot", true);
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

  const displayTelemetryHistory = useMemo(
    () => decimateTelemetryHistory(smoothTelemetryHistory(telemetryHistory)),
    [telemetryHistory]
  );
  const telemetryDarkMode = typeof document !== "undefined" && document.body.classList.contains("theme-dark");
  const telemetryChartOption = useMemo(
    () => telemetryOption(displayTelemetryHistory, telemetryAxisSettings),
    [displayTelemetryHistory, telemetryAxisSettings, telemetryDarkMode]
  );
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
              <ReactECharts option={telemetryChartOption} className="chart tall" lazyUpdate />
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
            persistenceKey="manual-function-generator"
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
              <button type="button" onClick={() => readSupply()} disabled={powerSupplyRunning}><RefreshCw size={14} /> Read</button>
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
  persistenceKey,
  children
}: {
  title: string;
  icon: React.ReactNode;
  headerExtra?: React.ReactNode;
  persistenceKey?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = usePersistentPanelOpen(`panel:${persistenceKey ?? title}`, true);
  return (
    <section className={`panel ${open ? "open" : "collapsed"}`}>
      <button type="button" className="panel-title panel-toggle" aria-expanded={open} onClick={() => setOpen((current) => !current)}>
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
  persistenceKey,
  children
}: {
  title: string;
  icon: React.ReactNode;
  summary?: string;
  defaultOpen?: boolean;
  persistenceKey?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = usePersistentPanelOpen(`collapsible:${persistenceKey ?? title}`, defaultOpen);
  return (
    <section className={`panel collapsible-panel ${open ? "open" : ""}`}>
      <button type="button" className="collapsible-title" aria-expanded={open} onClick={() => setOpen((current) => !current)}>
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
  // Recomputed artifacts keep the original capture/sweep id. Include the
  // analysis label and derived metrics in the cache key so rebuilding a
  // stored Scope image (for example after a Ts algorithm change) is visible
  // immediately instead of reusing the browser's old PNG.
  const versionParts = [
    ...versionKeys.map((key) => result?.[key]),
    record?.metrics.settling_analysis_version,
    record?.metrics.undershoot_settling_time_s,
    record?.metrics.overshoot_settling_time_s,
    record?.metrics.score
  ].filter((value) => value !== undefined && value !== null);
  const version = versionParts.length > 0 ? versionParts.join("-") : record?.iteration ?? Date.now();
  const [retryNonce, setRetryNonce] = useState(0);
  const [loadFailed, setLoadFailed] = useState(false);
  useEffect(() => {
    setRetryNonce(0);
    setLoadFailed(false);
  }, [imagePath, String(version), pending]);

  if (pending && !imagePath) {
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
      {pending && loadFailed ? <p className="message-line">Generating image...</p> : null}
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
  run: Pick<ResultRunSummary, "run_id" | "kind" | "display_name" | "algorithm">,
  dateCounts: Record<string, number>,
  index: number,
  includeDuplicateIndex = true
): string {
  let baseLabel = run.display_name && / #\d+$/.test(run.display_name) ? run.display_name : "";
  if (!baseLabel) {
    const date = runDateKey(run.run_id);
    const kind = run.kind === "saved" ? "Permanent" : "Recent";
    const duplicateSuffix = includeDuplicateIndex && (dateCounts[date] ?? 0) > 1 ? ` #${index + 1}` : "";
    baseLabel = `${kind} / ${date}${duplicateSuffix}`;
  }
  const algorithmLabel = run.algorithm === "DRL" ? "DRL" : run.algorithm === "Grid" ? "Grid" : "";
  return algorithmLabel ? `${baseLabel} [${algorithmLabel}]` : baseLabel;
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
          <button onClick={archiveRun} disabled={disabled || !currentRun}>
            Save
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
  const gainRebound = asDisplayNumber(metrics?.bode_gain_rebound_db ?? margins?.gain_rebound_db);
  const flatSpan = asDisplayNumber(metrics?.bode_gain_flat_span_decades ?? margins?.gain_flat_span_decades);
  const shapePenalty = asDisplayNumber(metrics?.bode_gain_shape_penalty ?? margins?.gain_shape_penalty);
  return (
    <dl className="kv">
      <dt>Phase Margin</dt>
      <dd>{phaseMargin === null ? "--" : `${phaseMargin.toFixed(2)} deg`}</dd>
      <dt>Crossover</dt>
      <dd>{crossover === null ? "--" : formatMaybeFrequency(crossover)}</dd>
      <dt>Gain Margin</dt>
      <dd>{gainMargin === null ? "--" : `${gainMargin.toFixed(2)} dB`}</dd>
      <dt>Gain Rebound</dt>
      <dd>{gainRebound === null ? "--" : `${gainRebound.toFixed(2)} dB`}</dd>
      <dt>Flat Span</dt>
      <dd>{flatSpan === null ? "--" : `${flatSpan.toFixed(2)} dec`}</dd>
      <dt>Shape Penalty</dt>
      <dd>{shapePenalty === null ? "--" : shapePenalty.toFixed(2)}</dd>
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
      <span>Refresh Rate</span>
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
  singleLineTitle = false,
  onChange
}: {
  label: string;
  limit: string;
  value: SearchParameter;
  integer?: boolean;
  singleLineTitle?: boolean;
  onChange: (value: SearchParameter) => void;
}) {
  const update = (field: "min" | "max", nextValue: number) => {
    const normalized = integer ? Math.round(nextValue) : nextValue;
    const next = { ...value, [field]: normalized };
    if (field === "min" && next.min > next.max) {
      next.max = next.min;
    }
    if (field === "max" && next.max < next.min) {
      next.min = next.max;
    }
    next.center = Math.min(Math.max(next.center, next.min), next.max);
    onChange(next);
  };

  return (
    <div className={`search-parameter${singleLineTitle ? " search-parameter-single-line-title" : ""}`}>
      <div className="search-parameter-title">
        <span>{label}</span>
        <small>{limit}</small>
      </div>
      <div className="search-parameter-grid">
        <DeferredSearchInput label="min" value={value.min} integer={integer} onCommit={(nextValue) => update("min", nextValue)} />
        <DeferredSearchInput label="max" value={value.max} integer={integer} onCommit={(nextValue) => update("max", nextValue)} />
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

const RunCurrentPanel = React.memo(function RunCurrentPanel({
  title,
  candidateTitle,
  metricsTitle,
  status,
  current,
  best,
  history,
  record,
  selectedAlgorithm
}: {
  title: string;
  candidateTitle: string;
  metricsTitle: string;
  status: TuningStatus | null;
  current: IterationRecord | null;
  best: IterationRecord | null;
  history: IterationRecord[];
  record: IterationRecord | null;
  selectedAlgorithm?: string;
}) {
  return (
    <Panel title={title} persistenceKey="autotune-current-result" icon={<Activity size={17} />}>
      <div className="combined-result-section">
        <h4>Run Status</h4>
        <RunStatusReadout
          status={status}
          current={current}
          best={best}
          history={history}
          selectedAlgorithm={selectedAlgorithm}
        />
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
});

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
      <dt>LS/LR BW</dt>
      <dd>{candidate ? candidate.mod0_ll_bw : "--"}</dd>
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
  history,
  selectedAlgorithm
}: {
  status: TuningStatus | null;
  current: IterationRecord | null;
  best: IterationRecord | null;
  history: IterationRecord[];
  selectedAlgorithm?: string;
}) {
  const iteration = history.length;
  const search = normalizeTuningConfig(status?.config).search;
  const maxCoarse = searchCoarseBudget(search);
  const maxRefined = searchRefinedBudget(search);
  const algorithmValue = status?.state === "running"
    ? status.experiment?.optimization_algorithm
    : selectedAlgorithm ?? status?.experiment?.optimization_algorithm;
  const isDrl = algorithmValue === "deep-reinforcement";
  const safeMaxIterations = Math.max(1, maxCoarse + maxRefined);
  // DRL candidates use phases such as drl_start/drl_policy instead of the
  // heuristic baseline/coordinate/local_refine phases. Their configured
  // budget is stored in max_coarse_iterations, so count the DRL history
  // against that budget instead of leaving the readout stuck at zero.
  const coarseDone = isDrl
    ? Math.min(maxCoarse, iteration)
    : Math.min(maxCoarse, countCoarseIterations(history));
  const refinedDone = Math.min(maxRefined, Math.max(0, iteration - coarseDone));
  const progressPct = Math.max(0, Math.min(100, (iteration / safeMaxIterations) * 100));
  const currentLabel = current ? `#${current.iteration} ${current.phase}` : "-";
  const bestPenalty = best ? displayPenaltyScore(best) : null;
  const bestObjective = best ? displayObjectiveScore(best) : null;
  const bestLabel = best
    ? `#${best.iteration} obj ${formatMaybeNumber(bestObjective, 3)} / pen ${formatMaybeNumber(bestPenalty, 3)}`
    : "-";
  const algorithm = isDrl
    ? "DRL"
    : algorithmValue === "heuristic" || !algorithmValue
      ? "Grid+Heuristic"
      : optimizationAlgorithmLabel(algorithmValue);
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
        <dt>Algorithm</dt>
        <dd className="run-status-algorithm" title={algorithm}>{algorithm}</dd>
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

function hasInvalidTransient(record: IterationRecord) {
  return (record.metrics.pass_reasons ?? []).some((reason) =>
    String(reason).toLowerCase().includes("invalid transient waveform")
  );
}

function hasInvalidSettling(record: IterationRecord, kind: "overshoot" | "undershoot") {
  const explicit = kind === "overshoot"
    ? record.metrics.overshoot_settling_valid
    : record.metrics.undershoot_settling_valid;
  return explicit === false || (explicit === undefined && hasInvalidTransient(record));
}

function RecommendedResultsPanel({
  records,
  selectedIteration,
  onPreview
}: {
  records: IterationRecord[];
  selectedIteration: number | null;
  onPreview: (iteration: number | null) => void;
}) {
  return (
    <Panel title="Top 5 Quality Basins" icon={<Gauge size={17} />}>
      {records.length > 0 ? (
        <div className="recommended-results">
          {records.map((record, index) => {
            const candidate = record.candidate;
            const selected = selectedIteration === record.iteration;
            return (
              <button
                key={record.iteration}
                type="button"
                className={`recommended-result${selected ? " selected" : ""}`}
                onClick={() => onPreview(selected ? null : record.iteration)}
                title={`Preview iteration ${record.iteration}`}
              >
                <span className="recommended-result-rank">{index + 1}</span>
                <span className="recommended-result-copy">
                  <strong>Iteration #{record.iteration}</strong>
                  <small>
                    {candidate
                      ? `kp ${candidate.mod0_kp}  ki ${candidate.mod0_ki}  kd ${candidate.mod0_kd}  p ${candidate.mod0_kpole1}/${candidate.mod0_kpole2}  BW ${candidate.mod0_ll_bw}`
                      : record.phase}
                  </small>
                </span>
                <span className="recommended-result-penalty">
                  <small>Objective / Penalty</small>
                  <strong>{formatMaybeNumber(displayObjectiveScore(record), 3)} / {formatMaybeNumber(displayPenaltyScore(record), 3)}</strong>
                </span>
              </button>
            );
          })}
        </div>
      ) : (
        <p className="muted">No valid result yet.</p>
      )}
    </Panel>
  );
}

function Metrics({ record }: { record: IterationRecord | null }) {
  if (!record) return <p className="muted">No iteration has run yet.</p>;
  const osSettlingInvalid = hasInvalidSettling(record, "overshoot");
  const usSettlingInvalid = hasInvalidSettling(record, "undershoot");
  const osDiagnostics = record.metrics.settling_diagnostics?.overshoot;
  const usDiagnostics = record.metrics.settling_diagnostics?.undershoot;
  return (
    <dl className="kv">
      <dt>Overshoot</dt>
      <dd>{record.metrics.overshoot_pct.toFixed(2)}%</dd>
      <dt>Undershoot</dt>
      <dd>{record.metrics.undershoot_pct.toFixed(2)}%</dd>
      <dt>OS settling</dt>
      <dd>{osSettlingInvalid ? "--" : formatSettlingUs(record.metrics.overshoot_settling_time_s, record.metrics.settling_time_s)}</dd>
      <dt>US settling</dt>
      <dd>{usSettlingInvalid ? "--" : formatSettlingUs(record.metrics.undershoot_settling_time_s, record.metrics.settling_time_s)}</dd>
      {(record.metrics.settling_analysis_version ?? 0) >= 2 ? (
        <>
          <dt>Ts rebound count</dt>
          <dd>OS {osDiagnostics?.secondary_excursion_count ?? 0} / US {usDiagnostics?.secondary_excursion_count ?? 0}</dd>
        </>
      ) : null}
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
      <dt>Gain Rebound</dt>
      <dd>{record.metrics.bode_gain_rebound_db === null || record.metrics.bode_gain_rebound_db === undefined ? "--" : `${record.metrics.bode_gain_rebound_db.toFixed(2)} dB`}</dd>
      <dt>Flat Span</dt>
      <dd>{record.metrics.bode_gain_flat_span_decades === null || record.metrics.bode_gain_flat_span_decades === undefined ? "--" : `${record.metrics.bode_gain_flat_span_decades.toFixed(2)} dec`}</dd>
      <dt>Shape Penalty</dt>
      <dd>{formatMaybeNumber(record.metrics.bode_gain_shape_penalty, 2)}</dd>
      <dt>Penalty</dt>
      <dd>{formatMaybeNumber(displayPenaltyScore(record), 3)}</dd>
      <dt>Objective</dt>
      <dd>{formatMaybeNumber(displayObjectiveScore(record), 3)}</dd>
      <dt>BW bonus</dt>
      <dd>{formatMaybeNumber(record.bandwidth_bonus, 3)}</dd>
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

const penaltyChartRefreshIntervalMs = 5000;
const iterationRowHeightPx = 34;
const iterationHeaderHeightPx = 38;
const iterationVirtualOverscan = 6;

function nearestHistoryIteration(history: IterationRecord[], target: number) {
  if (history.length === 0) return null;
  let low = 0;
  let high = history.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const iteration = history[middle].iteration;
    if (iteration === target) return iteration;
    if (iteration < target) low = middle + 1;
    else high = middle - 1;
  }
  const lower = history[Math.max(0, high)];
  const upper = history[Math.min(history.length - 1, low)];
  return Math.abs(lower.iteration - target) <= Math.abs(upper.iteration - target)
    ? lower.iteration
    : upper.iteration;
}

const PenaltyTrendPanel = React.memo(function PenaltyTrendPanel({
  history,
  selectedIteration,
  onSelectIteration,
  running
}: {
  history: IterationRecord[];
  selectedIteration: number | null;
  onSelectIteration: (iteration: number | null) => void;
  running: boolean;
}) {
  const latestHistoryRef = useRef(history);
  const selectedIterationRef = useRef(selectedIteration);
  const onSelectIterationRef = useRef(onSelectIteration);
  const chartCleanupRef = useRef<(() => void) | null>(null);
  const dragPublishTimerRef = useRef<number | null>(null);
  const pendingDragIterationRef = useRef<number | null>(null);
  const lastDragPublishRef = useRef(0);
  const lastPublishRef = useRef(performance.now());
  const publishTimerRef = useRef<number | null>(null);
  const [displayHistory, setDisplayHistory] = useState(history);

  latestHistoryRef.current = history;
  selectedIterationRef.current = selectedIteration;
  onSelectIterationRef.current = onSelectIteration;

  const handleChartReady = useCallback((chart: ECharts) => {
    chartCleanupRef.current?.();
    const renderer = chart.getZr();
    let dragging = false;
    let dragMoved = false;
    let suppressNextClick = false;

    const nearestIteration = (offsetX: number, offsetY: number) => {
      if (!chart.containPixel({ gridIndex: 0 }, [offsetX, offsetY])) return null;
      const converted = chart.convertFromPixel({ seriesIndex: 0 }, [offsetX, offsetY]);
      const iterationValue = Number(Array.isArray(converted) ? converted[0] : converted);
      if (!Number.isFinite(iterationValue)) return null;
      return nearestHistoryIteration(latestHistoryRef.current, iterationValue);
    };
    const selectedLineX = () => {
      const iteration = selectedIterationRef.current;
      if (iteration === null) return null;
      const converted = chart.convertToPixel({ xAxisIndex: 0 }, iteration);
      const pixel = Number(Array.isArray(converted) ? converted[0] : converted);
      return Number.isFinite(pixel) ? pixel : null;
    };
    const publishDragSelection = (iteration: number, immediate = false) => {
      pendingDragIterationRef.current = iteration;
      const publish = () => {
        dragPublishTimerRef.current = null;
        const pending = pendingDragIterationRef.current;
        pendingDragIterationRef.current = null;
        if (pending === null || pending === selectedIterationRef.current) return;
        lastDragPublishRef.current = performance.now();
        onSelectIterationRef.current(pending);
      };
      const elapsed = performance.now() - lastDragPublishRef.current;
      if (immediate || elapsed >= 60) {
        if (dragPublishTimerRef.current !== null) window.clearTimeout(dragPublishTimerRef.current);
        publish();
      } else if (dragPublishTimerRef.current === null) {
        dragPublishTimerRef.current = window.setTimeout(publish, 60 - elapsed);
      }
    };
    const handleMouseDown = (event: { offsetX: number; offsetY: number }) => {
      const lineX = selectedLineX();
      dragging = lineX !== null && Math.abs(event.offsetX - lineX) <= 12;
      dragMoved = false;
      if (dragging) renderer.setCursorStyle("ew-resize");
    };
    const handleMouseMove = (event: { offsetX: number; offsetY: number }) => {
      if (!dragging) {
        const lineX = selectedLineX();
        renderer.setCursorStyle(lineX !== null && Math.abs(event.offsetX - lineX) <= 12 ? "ew-resize" : "default");
        return;
      }
      const iteration = nearestIteration(event.offsetX, event.offsetY);
      if (iteration === null) return;
      dragMoved = true;
      publishDragSelection(iteration);
    };
    const handleMouseUp = (event: { offsetX: number; offsetY: number }) => {
      if (!dragging) return;
      const iteration = nearestIteration(event.offsetX, event.offsetY);
      if (iteration !== null) publishDragSelection(iteration, true);
      dragging = false;
      suppressNextClick = dragMoved;
      renderer.setCursorStyle("default");
    };
    const handleClick = (event: { offsetX: number; offsetY: number }) => {
      if (suppressNextClick) {
        suppressNextClick = false;
        return;
      }
      const iteration = nearestIteration(event.offsetX, event.offsetY);
      if (iteration !== null) onSelectIterationRef.current(iteration);
    };
    const handleGlobalOut = () => {
      dragging = false;
      renderer.setCursorStyle("default");
    };

    renderer.on("mousedown", handleMouseDown);
    renderer.on("mousemove", handleMouseMove);
    renderer.on("mouseup", handleMouseUp);
    renderer.on("click", handleClick);
    renderer.on("globalout", handleGlobalOut);
    chartCleanupRef.current = () => {
      renderer.off("mousedown", handleMouseDown);
      renderer.off("mousemove", handleMouseMove);
      renderer.off("mouseup", handleMouseUp);
      renderer.off("click", handleClick);
      renderer.off("globalout", handleGlobalOut);
    };
  }, []);

  useEffect(() => {
    latestHistoryRef.current = history;
    const publish = () => {
      publishTimerRef.current = null;
      lastPublishRef.current = performance.now();
      setDisplayHistory((current) => current === latestHistoryRef.current ? current : latestHistoryRef.current);
    };
    if (!running) {
      if (publishTimerRef.current !== null) {
        window.clearTimeout(publishTimerRef.current);
        publishTimerRef.current = null;
      }
      publish();
      return;
    }
    const elapsed = performance.now() - lastPublishRef.current;
    if (elapsed >= penaltyChartRefreshIntervalMs) {
      publish();
    } else if (publishTimerRef.current === null) {
      publishTimerRef.current = window.setTimeout(publish, penaltyChartRefreshIntervalMs - elapsed);
    }
  }, [history, running]);

  useEffect(() => () => {
    if (publishTimerRef.current !== null) window.clearTimeout(publishTimerRef.current);
    if (dragPublishTimerRef.current !== null) window.clearTimeout(dragPublishTimerRef.current);
    chartCleanupRef.current?.();
  }, []);

  const option = useMemo(
    () => penaltyOption(displayHistory, selectedIteration),
    [displayHistory, selectedIteration]
  );
  return (
    <Panel title="Penalty Trend" icon={<Pause size={17} />}>
      <div title="Click to select an iteration. Drag the red selection line left or right to scrub through iterations.">
        <ReactECharts option={option} className="chart" lazyUpdate onChartReady={handleChartReady} />
      </div>
    </Panel>
  );
});

const IterationHistoryPanel = React.memo(function IterationHistoryPanel({
  history,
  selectedIteration,
  onSelectIteration
}: {
  history: IterationRecord[];
  selectedIteration: number | null;
  onSelectIteration: (iteration: number | null) => void;
}) {
  return (
    <Panel title="Iteration History" icon={<RefreshCw size={17} />}>
      <IterationTable
        history={history}
        selectedIteration={selectedIteration}
        onSelectIteration={onSelectIteration}
      />
    </Panel>
  );
});

function IterationTable({
  history,
  selectedIteration,
  onSelectIteration
}: {
  history: IterationRecord[];
  selectedIteration: number | null;
  onSelectIteration: (iteration: number | null) => void;
}) {
  const displayHistory = useMemo(() => history.slice().reverse(), [history]);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const pendingScrollTopRef = useRef(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(310);
  const visibleStart = Math.max(
    0,
    Math.floor(Math.max(0, scrollTop - iterationHeaderHeightPx) / iterationRowHeightPx) - iterationVirtualOverscan
  );
  const visibleCount = Math.ceil(viewportHeight / iterationRowHeightPx) + iterationVirtualOverscan * 2;
  const visibleEnd = Math.min(displayHistory.length, visibleStart + visibleCount);
  const visibleHistory = displayHistory.slice(visibleStart, visibleEnd);
  const topSpacerHeight = visibleStart * iterationRowHeightPx;
  const bottomSpacerHeight = Math.max(0, (displayHistory.length - visibleEnd) * iterationRowHeightPx);

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    pendingScrollTopRef.current = event.currentTarget.scrollTop;
    if (scrollFrameRef.current !== null) return;
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      scrollFrameRef.current = null;
      setScrollTop(pendingScrollTopRef.current);
    });
  }, []);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const updateHeight = () => setViewportHeight(element.clientHeight || 310);
    updateHeight();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(updateHeight);
    observer?.observe(element);
    return () => observer?.disconnect();
  }, []);

  useEffect(() => () => {
    if (scrollFrameRef.current !== null) window.cancelAnimationFrame(scrollFrameRef.current);
  }, []);

  useEffect(() => {
    if (selectedIteration === null) return;
    const selectedIndex = displayHistory.findIndex((item) => item.iteration === selectedIteration);
    const element = containerRef.current;
    if (selectedIndex < 0 || !element) return;
    const rowTop = iterationHeaderHeightPx + selectedIndex * iterationRowHeightPx;
    const rowBottom = rowTop + iterationRowHeightPx;
    if (rowTop < element.scrollTop + iterationHeaderHeightPx) {
      element.scrollTo({ top: Math.max(0, rowTop - iterationHeaderHeightPx) });
    } else if (rowBottom > element.scrollTop + element.clientHeight) {
      element.scrollTo({ top: rowBottom - element.clientHeight });
    }
  }, [displayHistory, selectedIteration]);

  return (
    <div
      ref={containerRef}
      className="table-wrap iteration-history-wrap"
      aria-label="Iteration history"
      onScroll={handleScroll}
    >
      <table className="iteration-history-table">
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
            <th>LS/LR BW</th>
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
            <th>Objective</th>
          </tr>
        </thead>
        <tbody>
          {topSpacerHeight > 0 ? (
            <tr className="virtual-spacer" aria-hidden="true">
              <td colSpan={20} style={{ height: topSpacerHeight }} />
            </tr>
          ) : null}
          {visibleHistory.map((item) => (
            <tr
              key={item.iteration}
              data-iteration={item.iteration}
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
              <td>{item.candidate?.mod0_ll_bw ?? "--"}</td>
              <td>{item.candidate ? item.candidate.output_inductance_nh.toFixed(1) : "--"}</td>
              <td>{item.candidate ? item.candidate.effective_lc_inductance_nh.toFixed(1) : "--"}</td>
              <td>{formatMaybeNumber(item.metrics.phase_margin_deg, 1)}</td>
              <td>{formatMaybeNumber(item.metrics.crossover_frequency_hz === null || item.metrics.crossover_frequency_hz === undefined ? null : item.metrics.crossover_frequency_hz / 1000, 2)}</td>
              <td>{formatMaybeNumber(item.metrics.gain_margin_db, 2)}</td>
              <td>{item.metrics.overshoot_pct.toFixed(2)}</td>
              <td>{item.metrics.undershoot_pct.toFixed(2)}</td>
              <td>{hasInvalidSettling(item, "overshoot") ? "--" : formatSettlingUs(item.metrics.overshoot_settling_time_s, item.metrics.settling_time_s).replace(" us", "")}</td>
              <td>{hasInvalidSettling(item, "undershoot") ? "--" : formatSettlingUs(item.metrics.undershoot_settling_time_s, item.metrics.settling_time_s).replace(" us", "")}</td>
              <td>{formatMaybeNumber(displayPenaltyScore(item), 3)}</td>
              <td>{formatMaybeNumber(displayObjectiveScore(item), 3)}</td>
            </tr>
          ))}
          {bottomSpacerHeight > 0 ? (
            <tr className="virtual-spacer" aria-hidden="true">
              <td colSpan={20} style={{ height: bottomSpacerHeight }} />
            </tr>
          ) : null}
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

function downsamplePenaltyHistory(
  history: IterationRecord[],
  selectedIteration: number | null,
  maxPoints = 500
) {
  if (history.length <= maxPoints) return history;
  const selectedIndex = selectedIteration === null
    ? -1
    : history.findIndex((item) => item.iteration === selectedIteration);
  const indices = new Set<number>([0, history.length - 1]);
  if (selectedIndex >= 0) indices.add(selectedIndex);
  const bucketCount = Math.max(1, Math.floor((maxPoints - 3) / 2));
  const interiorLength = history.length - 2;

  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = 1 + Math.floor((bucket * interiorLength) / bucketCount);
    const end = Math.min(history.length - 1, 1 + Math.floor(((bucket + 1) * interiorLength) / bucketCount));
    let minimumIndex = start;
    let maximumIndex = start;
    let minimumValue = Number.POSITIVE_INFINITY;
    let maximumValue = Number.NEGATIVE_INFINITY;
    for (let index = start; index < end; index += 1) {
      const value = displayPenaltyScore(history[index]);
      if (value !== null && value < minimumValue) {
        minimumValue = value;
        minimumIndex = index;
      }
      if (value !== null && value > maximumValue) {
        maximumValue = value;
        maximumIndex = index;
      }
    }
    indices.add(minimumIndex);
    indices.add(maximumIndex);
  }

  return [...indices]
    .sort((left, right) => left - right)
    .slice(0, maxPoints)
    .map((index) => history[index]);
}

function penaltyOption(history: IterationRecord[], selectedIteration: number | null = null) {
  const sampledHistory = downsamplePenaltyHistory(history, selectedIteration);
  const plotted = sampledHistory.map((item) => [item.iteration, displayPenaltyScore(item)]);
  const objectives = sampledHistory.map((item) => [item.iteration, displayObjectiveScore(item)]);
  const selectedIndex = selectedIteration === null
    ? -1
    : sampledHistory.findIndex((item) => item.iteration === selectedIteration);
  const selectedPoint = selectedIndex >= 0
    ? [[selectedIteration, displayPenaltyScore(sampledHistory[selectedIndex])]]
    : [];
  const selectedMarkLine = selectedIndex >= 0
    ? [{ xAxis: selectedIteration }]
    : [];
  return {
    animation: false,
    tooltip: { trigger: "axis" },
    grid: { left: 48, right: 18, top: 48, bottom: 42 },
    xAxis: {
      type: "value",
      name: "iteration",
      min: sampledHistory[0]?.iteration,
      max: sampledHistory.at(-1)?.iteration,
      minInterval: 1
    },
    legend: {
      data: ["Penalty", "Objective"],
      top: 2,
      left: "center"
    },
    yAxis: { type: "value", name: "score" },
    series: [
      {
        name: "Penalty",
        type: "line",
        smooth: true,
        data: plotted,
        areaStyle: {},
        lineStyle: { color: "#ea4335" },
        markLine: {
          silent: true,
          symbol: "none",
          data: selectedMarkLine,
          label: { show: false },
          lineStyle: { color: "#ea4335", width: 2.5, type: "solid", opacity: 0.95 },
          z: 9
        }
      },
      {
        name: "Objective",
        type: "line",
        smooth: true,
        connectNulls: false,
        data: objectives,
        lineStyle: { color: "#1a73e8", width: 2 },
        itemStyle: { color: "#1a73e8" }
      },
      {
        name: "Selected iteration",
        type: "scatter",
        data: selectedPoint,
        symbolSize: 22,
        itemStyle: {
          color: "#ea4335",
          borderColor: "#ffffff",
          borderWidth: 3,
          shadowBlur: 12,
          shadowColor: "rgba(234, 67, 53, 0.55)"
        },
        z: 12,
        tooltip: { show: false }
      }
    ]
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
  const latestSample = history[history.length - 1];
  const latestMs = latestSample?.timestamp
    ? latestSample.timestamp * 1000
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
  if (reasons.includes("invalid transient waveform")) {
    return 300;
  }
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

function displayObjectiveScore(record: IterationRecord) {
  const value = Number(record.objective_score);
  return record.objective_score !== null
    && record.objective_score !== undefined
    && Number.isFinite(value)
    ? value
    : null;
}

function optimizationRecordCompare(left: IterationRecord, right: IterationRecord) {
  if (left.metrics.passed !== right.metrics.passed) return left.metrics.passed ? -1 : 1;
  const objectiveDelta = (displayObjectiveScore(left) ?? Number.POSITIVE_INFINITY)
    - (displayObjectiveScore(right) ?? Number.POSITIVE_INFINITY);
  if (objectiveDelta !== 0) return objectiveDelta;
  const bandwidthDelta = (right.candidate?.mod0_ll_bw ?? 0) - (left.candidate?.mod0_ll_bw ?? 0);
  if (bandwidthDelta !== 0) return bandwidthDelta;
  return (displayPenaltyScore(left) ?? Number.POSITIVE_INFINITY)
    - (displayPenaltyScore(right) ?? Number.POSITIVE_INFINITY);
}

function selectRecommendedRecords(
  preferred: IterationRecord[] | undefined,
  history: IterationRecord[],
  limit: number
) {
  const safeLimit = Math.max(0, Math.floor(limit));
  if (safeLimit === 0) return [];
  const historyByIteration = new Map(history.map((record) => [record.iteration, record]));
  const preferredRecords = (preferred ?? [])
    .map((record) => historyByIteration.get(record.iteration) ?? record)
    .filter((record) => record.candidate && isRecommendationRecordValid(record));
  if (preferredRecords.length > 0) {
    return uniqueIterationRecords(preferredRecords).slice(0, safeLimit);
  }

  const eligibleHistory = history.some((record) => displayObjectiveScore(record) !== null)
    ? history.filter((record) => displayObjectiveScore(record) !== null)
    : history;
  const ranked = uniqueIterationRecords(eligibleHistory)
    .filter((record) => record.candidate && isRecommendationRecordValid(record))
    .sort(optimizationRecordCompare);
  const qualityPool = ranked.slice(0, Math.max(safeLimit, safeLimit * BASIN_QUALITY_POOL_MULTIPLIER));
  const selected: IterationRecord[] = [];
  for (const minimumDistance of [0.18, 0.10, 0]) {
    for (const record of qualityPool) {
      if (selected.length >= safeLimit) {
        return selected.sort(optimizationRecordCompare);
      }
      if (selected.some((item) => item.iteration === record.iteration)) continue;
      if (
        minimumDistance > 0
        && selected.some((item) => candidateBasinDistance(item, record) < minimumDistance)
      ) {
        continue;
      }
      selected.push(record);
    }
  }
  return selected.sort(optimizationRecordCompare);
}

function uniqueIterationRecords(records: IterationRecord[]) {
  const seenIterations = new Set<number>();
  const seenCandidates = new Set<string>();
  return records.filter((record) => {
    const candidate = record.candidate;
    if (!candidate || seenIterations.has(record.iteration)) return false;
    const key = [
      candidate.mod0_kp,
      candidate.mod0_ki,
      candidate.mod0_kd,
      candidate.mod0_kpole1,
      candidate.mod0_kpole2,
      candidate.mod0_cm_gain,
      candidate.mod0_ll_bw,
      candidate.output_inductance_nh.toFixed(6),
      candidate.effective_lc_inductance_nh.toFixed(6)
    ].join(":");
    if (seenCandidates.has(key)) return false;
    seenIterations.add(record.iteration);
    seenCandidates.add(key);
    return true;
  });
}

function isRecommendationRecordValid(record: IterationRecord) {
  const penalty = displayPenaltyScore(record);
  return penalty !== null && Number.isFinite(penalty) && penalty < 250;
}

function candidateBasinDistance(left: IterationRecord, right: IterationRecord) {
  const leftCandidate = left.candidate;
  const rightCandidate = right.candidate;
  if (!leftCandidate || !rightCandidate) return Number.POSITIVE_INFINITY;
  const leftVector = [
    leftCandidate.mod0_kp / 255,
    leftCandidate.mod0_ki / 255,
    leftCandidate.mod0_kd / 255,
    (leftCandidate.mod0_kpole1 - 2) / 4,
    (leftCandidate.mod0_kpole2 - 2) / 4,
    leftCandidate.mod0_cm_gain / 9,
    (leftCandidate.mod0_ll_bw - 47) / 32,
    leftCandidate.output_inductance_nh / 40,
    leftCandidate.effective_lc_inductance_nh / 150
  ];
  const rightVector = [
    rightCandidate.mod0_kp / 255,
    rightCandidate.mod0_ki / 255,
    rightCandidate.mod0_kd / 255,
    (rightCandidate.mod0_kpole1 - 2) / 4,
    (rightCandidate.mod0_kpole2 - 2) / 4,
    rightCandidate.mod0_cm_gain / 9,
    (rightCandidate.mod0_ll_bw - 47) / 32,
    rightCandidate.output_inductance_nh / 40,
    rightCandidate.effective_lc_inductance_nh / 150
  ];
  return Math.sqrt(leftVector.reduce((total, value, index) => {
    const delta = value - rightVector[index];
    return total + delta * delta;
  }, 0));
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
