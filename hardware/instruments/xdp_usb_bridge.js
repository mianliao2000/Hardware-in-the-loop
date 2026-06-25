const usb = require(process.env.XDP_USB_MODULE || process.env.TEMP + "\\xdp_asar_extract\\node_modules\\usb");

function parseHexBytes(text) {
  if (!text) return [];
  return text
    .replace(/^0x/i, "")
    .split(/[\s,;:-]+/)
    .filter(Boolean)
    .flatMap((part) => {
      if (part.length > 2) {
        const bytes = [];
        for (let i = 0; i < part.length; i += 2) bytes.push(parseInt(part.slice(i, i + 2), 16));
        return bytes;
      }
      return [parseInt(part, 16)];
    });
}

function toHex(data) {
  return [...data].map((value) => value.toString(16).padStart(2, "0")).join(" ");
}

function packet(bytes) {
  return Buffer.from([bytes.length & 0xff, (bytes.length >> 8) & 0xff, ...bytes]);
}

function transfer(endpoint, data) {
  return new Promise((resolve, reject) => endpoint.transfer(data, (error) => (error ? reject(error) : resolve())));
}

function read(endpoint, length) {
  return new Promise((resolve, reject) =>
    endpoint.transfer(length, (error, data) => (error ? reject(error) : resolve(data))),
  );
}

function controlTransfer(device, requestType, requestId, value, index, data) {
  return new Promise((resolve) => {
    try {
      device.controlTransfer(requestType, requestId, value, index, data, () => resolve());
    } catch {
      resolve();
    }
  });
}

async function request(outEndpoint, inEndpoint, payload, readLength) {
  await transfer(outEndpoint, packet(payload));
  if (!readLength) return Buffer.alloc(0);
  return read(inEndpoint, readLength);
}

function argValue(name, fallback = undefined) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) return fallback;
  return process.argv[index + 1];
}

function xdpAddress(address7, mode, isRead) {
  if (mode === "7bit") return address7 & 0x7f;
  return ((address7 & 0x7f) << 1) | (isRead ? 1 : 0);
}

async function main() {
  const command = process.argv[2];
  const device = usb.findByIds(0x10c4, 0xea61);
  if (!device) throw new Error("XDP USB dongle not found");

  device.open();
  try {
    const iface = device.interfaces[0];
    iface.claim();
    const outEndpoint = iface.endpoints.find((endpoint) => endpoint.direction === "out");
    const inEndpoint = iface.endpoints.find((endpoint) => endpoint.direction === "in");
    outEndpoint.timeout = Number(argValue("--timeout-ms", "1000"));
    inEndpoint.timeout = Number(argValue("--timeout-ms", "1000"));

    for (const [requestType, requestId, value, index] of [
      [0, 9, 1, 0],
      [65, 2, 2, 0],
      [65, 2, 1, 0],
    ]) {
      await controlTransfer(device, requestType, requestId, value, index, Buffer.from([0]));
    }

    if (command === "identify") {
      const data = await request(outEndpoint, inEndpoint, [1, 0, 1], 6);
      console.log(JSON.stringify({ ok: true, data: toHex(data) }));
      return;
    }

    if (command === "transfer") {
      const address = Number(argValue("--address"));
      const writeData = parseHexBytes(argValue("--write", ""));
      const readLength = Number(argValue("--read", "0"));
      const mode = argValue("--address-mode", "xdp_8bit");
      const busAddress = xdpAddress(address, mode, readLength > 0);
      const payload = [9, 0, 1, busAddress, writeData.length, ...writeData];
      if (readLength > 0) payload.push(readLength);
      const response = await request(outEndpoint, inEndpoint, payload, 1 + readLength);
      const status = response[0] ?? 0xff;
      console.log(JSON.stringify({ ok: status === 0, status, data: toHex(response.slice(1)) }));
      return;
    }

    throw new Error(`Unknown command: ${command}`);
  } finally {
    device.close();
  }
}

main().catch((error) => {
  console.error(JSON.stringify({ ok: false, error: String(error && error.message ? error.message : error) }));
  process.exit(1);
});
