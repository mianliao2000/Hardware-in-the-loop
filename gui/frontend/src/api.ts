import type { InstrumentKey, SelfTestResponse, TuningConfig, TuningStatus, VoutReadback } from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload as T;
}

export function getTuningStatus(): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/status");
}

export function startTuning(config: TuningConfig): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/start", {
    method: "POST",
    body: JSON.stringify({ config })
  });
}

export function stopTuning(): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/stop", { method: "POST", body: "{}" });
}

export function stepTuning(config: TuningConfig): Promise<TuningStatus> {
  return requestJson<TuningStatus>("/api/tuning/step", {
    method: "POST",
    body: JSON.stringify({ config })
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
