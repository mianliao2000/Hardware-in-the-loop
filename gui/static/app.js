const state = {
  page: 0,
  busy: false,
};

const els = {
  addressInput: document.querySelector("#addressInput"),
  voutInput: document.querySelector("#voutInput"),
  setButton: document.querySelector("#setButton"),
  setButtonTop: document.querySelector("#setButtonTop"),
  refreshButton: document.querySelector("#refreshButton"),
  readButton: document.querySelector("#readButton"),
  readButtonTop: document.querySelector("#readButtonTop"),
  readButtonMain: document.querySelector("#readButtonMain"),
  readButtonSide: document.querySelector("#readButtonSide"),
  scanButton: document.querySelector("#scanButton"),
  connectionPill: document.querySelector("#connectionPill"),
  deviceDot: document.querySelector("#deviceDot"),
  loopADot: document.querySelector("#loopADot"),
  loopAMeta: document.querySelector("#loopAMeta"),
  subtitle: document.querySelector("#subtitle"),
  sheetTitle: document.querySelector("#sheetTitle"),
  tableVoutA: document.querySelector("#tableVoutA"),
  tableCmdA: document.querySelector("#tableCmdA"),
  tableModeA: document.querySelector("#tableModeA"),
  tableExpA: document.querySelector("#tableExpA"),
  targetValue: document.querySelector("#targetValue"),
  telemetryValue: document.querySelector("#telemetryValue"),
  modeValue: document.querySelector("#modeValue"),
  rawValue: document.querySelector("#rawValue"),
  lastUpdated: document.querySelector("#lastUpdated"),
  logList: document.querySelector("#logList"),
  loopButtons: [...document.querySelectorAll(".loop-switch button")],
};

function volts(value, digits = 6) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${value.toFixed(digits)} V`;
}

function plainVolts(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return value.toFixed(3);
}

function activeAddress() {
  return els.addressInput.value.trim() || "0x5E";
}

function activeLoopName() {
  return state.page === 0 ? "Loop A" : state.page === 1 ? "Loop B" : `Page ${state.page}`;
}

function setBusy(busy) {
  state.busy = busy;
  document.body.classList.toggle("busy", busy);
  if (busy) {
    els.connectionPill.textContent = "Busy";
  }
}

function setStatus(kind, text) {
  els.connectionPill.textContent = text;
  els.deviceDot.className = `status-dot ${kind}`;
  els.loopADot.className = `status-dot small ${kind}`;
}

function log(message) {
  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
  els.logList.prepend(item);
}

function updateReadback(data) {
  const target = volts(data.vout_command_v);
  const telemetry = volts(data.read_vout_v);
  els.loopAMeta.textContent = `${data.loop || activeLoopName()}: Vout = ${plainVolts(data.read_vout_v)}V`;
  els.subtitle.textContent = `${data.address || activeAddress()} | ${data.loop || activeLoopName()}`;
  els.sheetTitle.textContent = `${data.loop || activeLoopName()} - Controller Settings`;
  els.tableVoutA.textContent = plainVolts(data.read_vout_v);
  els.tableCmdA.textContent = plainVolts(data.vout_command_v);
  els.tableModeA.textContent = data.vout_mode_raw || "--";
  els.tableExpA.textContent = data.exponent ?? "--";
  els.targetValue.textContent = target;
  els.telemetryValue.textContent = telemetry;
  els.modeValue.textContent = `${data.vout_mode || "--"} (${data.exponent ?? "--"})`;
  els.rawValue.textContent = data.raw_written || data.vout_mode_raw || "--";
  els.lastUpdated.textContent = new Date().toLocaleTimeString();
}

async function readVout() {
  if (state.busy) return;
  setBusy(true);
  try {
    const query = new URLSearchParams({
      address: activeAddress(),
      page: String(state.page),
      adapter: "xdp",
    });
    const response = await fetch(`/api/read?${query.toString()}`);
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "Read failed");
    updateReadback(data);
    setStatus("ok", "USB005 V59.28");
    log(`Read ${data.loop}: target ${volts(data.vout_command_v)}, telemetry ${volts(data.read_vout_v)}`);
  } catch (error) {
    setStatus("error", "Error");
    log(`Read failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function setVout() {
  if (state.busy) return;
  const voltage = Number(els.voutInput.value);
  if (!Number.isFinite(voltage)) {
    log("Set failed: invalid voltage");
    return;
  }
  setBusy(true);
  try {
    const response = await fetch("/api/vout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        address: activeAddress(),
        page: state.page,
        adapter: "xdp",
        voltage,
      }),
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "Set failed");
    updateReadback(data);
    setStatus("ok", "USB005 V59.28");
    log(`Set ${data.loop}: requested ${volts(data.requested_v)}, telemetry ${volts(data.read_vout_v)}`);
  } catch (error) {
    setStatus("error", "Error");
    log(`Set failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

function selectPage(page) {
  state.page = page;
  els.loopButtons.forEach((button) => button.classList.toggle("active", Number(button.dataset.page) === page));
  els.sheetTitle.textContent = `${activeLoopName()} - Controller Settings`;
}

els.loopButtons.forEach((button) => {
  button.addEventListener("click", () => {
    selectPage(Number(button.dataset.page));
    readVout();
  });
});

[
  els.readButton,
  els.readButtonTop,
  els.readButtonMain,
  els.readButtonSide,
  els.refreshButton,
  els.scanButton,
].forEach((button) => button.addEventListener("click", readVout));

[els.setButton, els.setButtonTop].forEach((button) => button.addEventListener("click", setVout));

selectPage(0);
log("Google Bench ready");
readVout();
