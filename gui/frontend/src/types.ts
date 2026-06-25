export type PidParameters = {
  kp: number;
  ki: number;
  kd: number;
  kf: number;
};

export type Waveform = {
  time_s: number[];
  vout_v: number[];
};

export type ResponseMetrics = {
  overshoot_pct: number;
  undershoot_pct: number;
  settling_time_s: number;
  oscillations: number;
  score: number;
  passed: boolean;
};

export type IterationRecord = {
  iteration: number;
  phase: string;
  wc_rad_s: number;
  phi_deg: number;
  pid: PidParameters;
  metrics: ResponseMetrics;
  waveform: Waveform;
  timestamp: number;
};

export type TuningConfig = {
  plant: {
    vdc: number;
    inductance_h: number;
    capacitance_f: number;
    capacitor_esr_ohm: number;
    inductor_dcr_ohm: number;
  };
  targets: {
    vout_target_v: number;
    overshoot_pct: number;
    undershoot_pct: number;
    settling_time_s: number;
    oscillations: number;
  };
  search: {
    wc_min_rad_s: number;
    wc_max_rad_s: number;
    phi_min_deg: number;
    phi_max_deg: number;
    initial_wc_rad_s: number;
    initial_phi_deg: number;
    max_iterations: number;
  };
};

export type TuningStatus = {
  ok?: boolean;
  state: "idle" | "running" | "stopped" | "complete" | "error";
  message: string;
  config: TuningConfig;
  current: IterationRecord | null;
  best: IterationRecord | null;
  history: IterationRecord[];
  pid_programming: {
    available: boolean;
    mode: string;
    disabled: boolean;
    message: string;
    write_attempts: number;
  };
};

export type VoutReadback = {
  ok: boolean;
  error?: string;
  address?: string;
  page?: number;
  loop?: string;
  vout_command_v?: number;
  read_vout_v?: number;
  vout_mode?: string;
  timestamp?: number;
};

export type InstrumentTestResult = {
  key: string;
  label: string;
  status: "passed" | "failed";
  resource: string;
  resource_present: boolean;
  identity: string;
  details: Record<string, string>;
  actions: Array<{ name: string; status: string; value: string }>;
  restored: boolean;
  error: string;
  duration_s: number;
};

export type InstrumentKey = "afg" | "bode" | "power_supply" | "scope" | "board_i2c";

export type SelfTestResponse = {
  ok: boolean;
  timestamp: number;
  duration_s: number;
  visa_resources: string[];
  visa_resource_error: string | null;
  tests: InstrumentTestResult[];
  all_passed: boolean;
};
