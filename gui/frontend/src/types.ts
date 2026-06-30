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
    phase_margin_deg: number;
    crossover_frequency_hz: number;
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
  operation?: string | null;
  status_word?: string | null;
  vout_command_v?: number;
  read_vout_v?: number;
  exponent?: number;
  vout_mode?: string;
  timestamp?: number;
  read_iout_a?: number;
};

export type InductanceField = {
  name: string;
  memory_address: string;
  bitfield: string;
  word: string;
  raw: number;
  raw_hex: string;
  value_nh: number | null;
  requested_nh?: number;
  actual_nh?: number | null;
  word_before?: string;
  word_after?: string;
  changed?: boolean;
};

export type InductanceReadback = {
  ok: boolean;
  error?: string;
  address?: string;
  page?: number;
  loop?: string;
  output_inductance?: InductanceField;
  effective_lc_inductance?: InductanceField;
  writes?: Record<string, InductanceField>;
  timestamp?: number;
};

export type XdpPidField = {
  name: string;
  memory_address: string;
  bitfield: string;
  word: string;
  raw: number;
  raw_hex: string;
  min: number;
  max: number;
  step: number;
};

export type XdpPidRegisterBlock = {
  name: string;
  memory_address: string;
  word: string;
  fields: Record<string, XdpPidField>;
};

export type XdpPidReadback = {
  ok: boolean;
  error?: string;
  address?: string;
  page?: number;
  loop?: string;
  pid_registers?: XdpPidRegisterBlock;
  write?: {
    name: string;
    memory_address: string;
    word_before: string;
    word_after: string;
    changed: boolean;
    writes: Record<string, XdpPidField>;
    readback: XdpPidRegisterBlock;
  };
  timestamp?: number;
};

export type PmbusOutputAction = "on" | "off";
export type XdpOutputAction = "enable" | "disable" | "release";

export type PmbusOutputReadback = {
  ok: boolean;
  error?: string;
  address?: string;
  page?: number;
  loop?: string;
  operation?: string | null;
  operation_before?: string | null;
  operation_after?: string | null;
  operation_written?: string | null;
  operation_bit7_before?: number;
  operation_bit7_after?: number;
  preserved_bits_6_0?: string | null;
  on_off_config?: string | null;
  status_word?: string | null;
  requested?: string;
  standard_commands?: Record<string, string>;
  note?: string;
  timestamp?: number;
};

export type XdpOutputReadback = {
  ok: boolean;
  error?: string;
  address?: string;
  page?: number;
  loop?: string;
  method?: string;
  requested?: string;
  state?: string;
  state_written?: string;
  operation?: string | null;
  status_word?: string | null;
  readback?: {
    name?: string;
    memory_address?: string;
    bitfield?: string;
    page?: number;
    word?: string;
    byte?: string;
    raw?: number;
    raw_binary?: string;
    state?: string;
    bit5_sw_enable_pin_value?: number;
    bit4_enable_sw_enable_pin?: number;
  };
  write?: Record<string, unknown>;
  timestamp?: number;
};

export type BodeSweepConfig = {
  host?: string;
  port?: number;
  start_hz: number;
  stop_hz: number;
  points: number;
  bandwidth_hz: number;
  source_dbm?: number | null;
  timeout_ms?: number;
};

export type BodeSweepReadback = {
  ok: boolean;
  error?: string;
  identity?: string;
  host?: string;
  port?: number;
  config?: BodeSweepConfig;
  frequency_hz?: number[];
  magnitude_db?: number[];
  phase_deg?: number[];
  sweep_id?: string;
  data_file?: string;
  original_points?: number;
  display_points?: number;
  bode_png?: string | null;
  bode_png_error?: string | null;
  margins?: {
    phase_margin_deg?: number | null;
    phase_crossover_hz?: number | null;
    gain_margin_db?: number | null;
    gain_crossover_hz?: number | null;
  };
  system_error?: string;
  duration_s?: number;
  timestamp?: number;
};

export type ScopeWaveform = {
  source: string;
  x: number[];
  y: number[];
  x_unit: string;
  y_unit: string;
  time_span_s?: number | null;
  original_points?: number | null;
  plotted_points?: number | null;
  display_points?: number | null;
  display_strategy?: string | null;
  capture_id?: string | null;
  data_file?: string | null;
  transfer_encoding?: string | null;
};

export type ScopeMeasurementValue = {
  source: string;
  measurement: string;
  value: number | null;
  ok: boolean;
  error?: string;
};

export type ScopeCaptureReadback = {
  ok: boolean;
  error?: string;
  resource?: string;
  identity?: string;
  channels?: string[];
  measurements?: string[];
  waveforms?: ScopeWaveform[];
  measurement_values?: ScopeMeasurementValue[];
  capture_id?: string;
  scope_png?: string | null;
  scope_png_error?: string | null;
  function_generator_frequency_hz?: number | null;
  scope_window_s?: number | null;
  scope_actual_window_s?: number | null;
  scope_scale_s_per_div?: number | null;
  scope_trigger_source?: string | null;
  scope_trigger_slope?: string | null;
  scope_trigger_offset_from_left_s?: number | null;
  scope_trigger_position_percent?: number | null;
  duration_s?: number;
  timestamp?: number;
};

export type PowerSupplyReadback = {
  ok: boolean;
  error?: string | null;
  resource?: string;
  identity?: string;
  output_enabled?: boolean | null;
  voltage_setpoint_v?: number | null;
  current_limit_a?: number | null;
  measured_voltage_v?: number | null;
  measured_current_a?: number | null;
  timestamp?: number;
};

export type FunctionGeneratorReadback = {
  ok: boolean;
  error?: string;
  resource?: string;
  channel?: number;
  identity?: string;
  function?: string | null;
  frequency_hz?: number | null;
  voltage_unit?: string | null;
  amplitude_vpp?: number | null;
  offset_v?: number | null;
  high_v?: number | null;
  low_v?: number | null;
  phase_deg?: number | null;
  duty_percent?: number | null;
  pulse_width_s?: number | null;
  output?: string | null;
  system_error?: string | null;
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
