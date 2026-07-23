import type { AutotuneArchiveResponse, AutotuneExperimentConfig, AutotuneGifResponse, AutotuneRunsResponse, BodeSweepConfig, BodeSweepReadback, DrlWorkflowStatus, FunctionGeneratorReadback, InductanceReadback, InstrumentKey, LlmChatMessage, LlmChatResponse, PmbusOutputAction, PmbusOutputReadback, PowerSupplyReadback, ScopeCaptureReadback, SelfTestResponse, TuningConfig, TuningStatus, VoutReadback, XdpOutputAction, XdpOutputReadback, XdpPidReadback } from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });
  const text = await response.text();
  let payload: any;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    const preview = text.slice(0, 80).replace(/\s+/g, " ");
    throw new Error(`Expected JSON from ${url}, got ${response.status}: ${preview}`);
  }
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload as T;
}

export function getTuningStatus(afterIteration?: number, historyToken?: string): Promise<TuningStatus> {
  const params = new URLSearchParams();
  if (afterIteration !== undefined && historyToken) {
    params.set("after_iteration", String(Math.max(0, Math.round(afterIteration))));
    params.set("history_token", historyToken);
  }
  const query = params.toString();
  return requestJson<TuningStatus>(`/api/tuning/status${query ? `?${query}` : ""}`);
}

export function getDrlWorkflowStatus(): Promise<DrlWorkflowStatus> {
  return requestJson<DrlWorkflowStatus>("/api/tuning/drl/status");
}

export function runDrlWorkflowAction(
  action: "collect" | "train" | "validate",
  config: TuningConfig,
  experiment: AutotuneExperimentConfig
): Promise<DrlWorkflowStatus> {
  return requestJson<DrlWorkflowStatus>(`/api/tuning/drl/${action}`, {
    method: "POST",
    body: JSON.stringify({ config, experiment })
  });
}

export function stopDrlWorkflow(): Promise<DrlWorkflowStatus> {
  return requestJson<DrlWorkflowStatus>("/api/tuning/drl/stop", { method: "POST", body: "{}" });
}

export function startTuning(config: TuningConfig, experiment?: AutotuneExperimentConfig): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/start", {
    method: "POST",
    body: JSON.stringify({ config, experiment })
  });
}

export function pauseTuning(): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/pause", { method: "POST", body: "{}" });
}

export function resumeTuning(run_id?: string, kind?: string): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/resume", {
    method: "POST",
    body: JSON.stringify({ run_id, kind })
  });
}

export function resetTuning(config?: TuningConfig, experiment?: AutotuneExperimentConfig): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/reset", {
    method: "POST",
    body: JSON.stringify({ config, experiment })
  });
}

export function stepTuning(config: TuningConfig, experiment?: AutotuneExperimentConfig): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/step", {
    method: "POST",
    body: JSON.stringify({ config, experiment })
  });
}

export function getTuningRuns(): Promise<AutotuneRunsResponse> {
  return requestJson<AutotuneRunsResponse>("/api/tuning/runs");
}

export function archiveCurrentTuningRun(name?: string, run_id?: string, kind?: string): Promise<AutotuneArchiveResponse> {
  return requestJson<AutotuneArchiveResponse>("/api/tuning/archive", {
    method: "POST",
    body: JSON.stringify({ name, run_id, kind })
  });
}

export function deleteTuningRun(run_id: string, kind: string): Promise<{ ok: boolean; deleted_run_id: string; kind: string }> {
  return requestJson<{ ok: boolean; deleted_run_id: string; kind: string }>("/api/tuning/delete", {
    method: "POST",
    body: JSON.stringify({ run_id, kind })
  });
}

export function loadTuningRun(run_id: string, kind: string): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/load", {
    method: "POST",
    body: JSON.stringify({ run_id, kind })
  });
}

export function saveTuningAnimationGif(run_id?: string, kind?: string, duration_ms?: number): Promise<AutotuneGifResponse> {
  return requestJson<AutotuneGifResponse>("/api/tuning/gif", {
    method: "POST",
    body: JSON.stringify({ run_id, kind, duration_ms })
  });
}

export function openTuningAnimationGif(run_id?: string, kind?: string, duration_ms?: number): Promise<AutotuneGifResponse> {
  return requestJson<AutotuneGifResponse>("/api/tuning/gif/open", {
    method: "POST",
    body: JSON.stringify({ run_id, kind, duration_ms })
  });
}

export function sendLlmChat(messages: LlmChatMessage[], context?: Record<string, unknown>, modelChoice?: string): Promise<LlmChatResponse> {
  return requestJson<LlmChatResponse>("/api/llm/chat", {
    method: "POST",
    body: JSON.stringify({ messages, context, model_choice: modelChoice })
  });
}

export function readVout(address: string, page: number, adapter: string): Promise<VoutReadback> {
  const params = new URLSearchParams({ address, page: String(page), adapter });
  return requestJson<VoutReadback>(`/api/read?${params.toString()}`);
}

export function setVout(address: string, page: number, adapter: string, voltage: number): Promise<VoutReadback> {
  return requestJson<VoutReadback>("/api/vout", {
    method: "POST",
    body: JSON.stringify({ address, page, adapter, voltage })
  });
}

export function readInductance(address: string, page: number, adapter: string): Promise<InductanceReadback> {
  const params = new URLSearchParams({ address, page: String(page), adapter });
  return requestJson<InductanceReadback>(`/api/inductance?${params.toString()}`);
}

export function setInductance(
  address: string,
  page: number,
  adapter: string,
  values: { output_inductance_nh?: number; effective_lc_inductance_nh?: number }
): Promise<InductanceReadback> {
  return requestJson<InductanceReadback>("/api/inductance", {
    method: "POST",
    body: JSON.stringify({ address, page, adapter, ...values })
  });
}

export function readXdpPid(address: string, page: number, adapter: string): Promise<XdpPidReadback> {
  const params = new URLSearchParams({ address, page: String(page), adapter });
  return requestJson<XdpPidReadback>(`/api/xdp-pid?${params.toString()}`);
}

export function setXdpPid(
  address: string,
  page: number,
  adapter: string,
  values: Record<string, number>
): Promise<XdpPidReadback> {
  return requestJson<XdpPidReadback>("/api/xdp-pid", {
    method: "POST",
    body: JSON.stringify({ address, page, adapter, values })
  });
}

export function readPmbusOutput(address: string, page: number, adapter: string): Promise<PmbusOutputReadback> {
  const params = new URLSearchParams({ address, page: String(page), adapter });
  return requestJson<PmbusOutputReadback>(`/api/pmbus-output?${params.toString()}`);
}

export function setPmbusOutput(
  address: string,
  page: number,
  adapter: string,
  action: PmbusOutputAction
): Promise<PmbusOutputReadback> {
  return requestJson<PmbusOutputReadback>("/api/pmbus-output", {
    method: "POST",
    body: JSON.stringify({ address, page, adapter, action })
  });
}

export function readXdpOutput(address: string, page: number, adapter: string): Promise<XdpOutputReadback> {
  const params = new URLSearchParams({ address, page: String(page), adapter });
  return requestJson<XdpOutputReadback>(`/api/xdp-output?${params.toString()}`);
}

export function setXdpOutput(
  address: string,
  page: number,
  adapter: string,
  action: XdpOutputAction
): Promise<XdpOutputReadback> {
  return requestJson<XdpOutputReadback>("/api/xdp-output", {
    method: "POST",
    body: JSON.stringify({ address, page, adapter, action })
  });
}

export function runBodeSweep(config: BodeSweepConfig): Promise<BodeSweepReadback> {
  return fetch("/api/bode/sweep", {
    headers: { "Content-Type": "application/json" },
    method: "POST",
    body: JSON.stringify(config)
  }).then(async (response) => {
    const text = await response.text();
    let payload: any;
    try {
      payload = text ? JSON.parse(text) : { ok: false, error: `Request failed: ${response.status}` };
    } catch {
      const preview = text.slice(0, 80).replace(/\s+/g, " ");
      throw new Error(`Expected JSON from /api/bode/sweep, got ${response.status}: ${preview}`);
    }
    if (!response.ok && !payload.error) {
      payload.error = `Request failed: ${response.status}`;
    }
    return payload as BodeSweepReadback;
  });
}

export function captureScope(config: {
  resource?: string;
  channels: string[];
  measurements: string[];
  points?: number;
  function_generator_frequency_hz?: number;
  scope_axis_settings?: {
    leftMin: number;
    leftMax: number;
    rightMin: number;
    rightMax: number;
    channelAxes: Record<string, "left" | "right">;
  };
}): Promise<ScopeCaptureReadback> {
  return requestJson<ScopeCaptureReadback>("/api/scope", {
    method: "POST",
    body: JSON.stringify(config)
  });
}

export function setScopeAcquisition(running: boolean, resource?: string): Promise<{ ok: boolean; running?: boolean | null; error?: string; duration_s?: number }> {
  return requestJson<{ ok: boolean; running?: boolean | null; error?: string; duration_s?: number }>("/api/scope/acquisition", {
    method: "POST",
    body: JSON.stringify({ resource, running })
  });
}

export function warmScope(resource?: string): Promise<{ ok: boolean; error?: string; duration_s?: number; session_reused?: boolean }> {
  return requestJson<{ ok: boolean; error?: string; duration_s?: number; session_reused?: boolean }>("/api/scope/warmup", {
    method: "POST",
    body: JSON.stringify({ resource })
  });
}

export function readPowerSupply(resource?: string): Promise<PowerSupplyReadback> {
  const params = new URLSearchParams();
  if (resource) params.set("resource", resource);
  return requestJson<PowerSupplyReadback>(`/api/power-supply?${params.toString()}`);
}

export function setPowerSupply(config: {
  resource?: string;
  voltage_v?: number;
  current_limit_a?: number;
  output_enabled?: boolean;
}): Promise<PowerSupplyReadback> {
  return requestJson<PowerSupplyReadback>("/api/power-supply", {
    method: "POST",
    body: JSON.stringify(config)
  });
}

export function readFunctionGenerator(resource?: string, channel = 1): Promise<FunctionGeneratorReadback> {
  const params = new URLSearchParams({ channel: String(channel) });
  if (resource) params.set("resource", resource);
  return requestJson<FunctionGeneratorReadback>(`/api/function-generator?${params.toString()}`);
}

export function setFunctionGenerator(config: {
  resource?: string;
  channel: number;
  mode: string;
  voltage_unit?: string;
  frequency_hz?: number;
  high_v?: number;
  low_v?: number;
  pulse_width_s?: number | null;
  dc_level_v?: number;
  amplitude_vpp?: number;
  offset_v?: number;
  phase_deg?: number | null;
  output_enabled?: boolean;
}): Promise<FunctionGeneratorReadback> {
  return requestJson<FunctionGeneratorReadback>("/api/function-generator", {
    method: "POST",
    body: JSON.stringify(config)
  });
}

export function runSelfTest(): Promise<SelfTestResponse> {
  return requestJson<SelfTestResponse>("/api/self-test", {
    method: "POST",
    body: "{}"
  });
}

export function runSelfTestDevice(device: InstrumentKey): Promise<SelfTestResponse> {
  const params = new URLSearchParams({ device });
  return requestJson<SelfTestResponse>(`/api/self-test?${params.toString()}`, {
    method: "POST",
    body: "{}"
  });
}
