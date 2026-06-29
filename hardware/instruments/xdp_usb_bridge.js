const usb = require(process.env.XDP_USB_MODULE || process.env.TEMP + "\\xdp_asar_extract\\node_modules\\usb");
const readline = require("readline");

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

function controlTransferRead(device, requestType, requestId, value, index, length) {
  return new Promise((resolve, reject) => {
    try {
      device.controlTransfer(requestType, requestId, value, index, length, (error, data) =>
        error ? reject(error) : resolve(data),
      );
    } catch (error) {
      reject(error);
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

function intToBytes(value, length) {
  let next = BigInt(value);
  const bytes = [];
  for (let index = 0; index < length; index += 1) {
    bytes.push(Number(next & 0xffn));
    next >>= 8n;
  }
  return bytes;
}

function bytesToInt(data) {
  let value = 0n;
  [...data].forEach((byte, index) => {
    value |= BigInt(byte) << BigInt(8 * index);
  });
  return value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value.toString();
}

async function setupDevice(timeoutMs) {
  const device = usb.findByIds(0x10c4, 0xea61);
  if (!device) throw new Error("XDP USB dongle not found");

  device.open();
  const iface = device.interfaces[0];
  iface.claim();
  const outEndpoint = iface.endpoints.find((endpoint) => endpoint.direction === "out");
  const inEndpoint = iface.endpoints.find((endpoint) => endpoint.direction === "in");
  outEndpoint.timeout = Number(timeoutMs);
  inEndpoint.timeout = Number(timeoutMs);

  for (const [requestType, requestId, value, index] of [
    [0, 9, 1, 0],
    [65, 2, 2, 0],
    [65, 2, 1, 0],
  ]) {
    await controlTransfer(device, requestType, requestId, value, index, Buffer.from([0]));
  }

  return { device, outEndpoint, inEndpoint };
}

async function executeCommand(command, options, outEndpoint, inEndpoint) {
  if (command === "identify") {
    const data = await request(outEndpoint, inEndpoint, [1, 0, 1], 6);
    return { ok: true, data: toHex(data) };
  }

  if (command === "cp210x-read-latch") {
    const length = Number(options.length || 2);
    const data = await controlTransferRead(outEndpoint.device, 0xc1, 0xff, 0x00c2, 0, length);
    return { ok: true, data: toHex(data), value: bytesToInt(data) };
  }

  if (command === "cp210x-get-part-number") {
    const data = await controlTransferRead(outEndpoint.device, 0xc1, 0xff, 0x370b, 0, 1);
    return { ok: true, data: toHex(data), value: bytesToInt(data) };
  }

  if (command === "transfer") {
    const address = Number(options.address);
    const writeData = parseHexBytes(options.write || "");
    const readLength = Number(options.read || 0);
    const mode = options.addressMode || options["address-mode"] || "xdp_8bit";
    const busAddress = xdpAddress(address, mode, readLength > 0);
    const payload = [9, 0, 1, busAddress, writeData.length, ...writeData];
    if (readLength > 0) payload.push(readLength);
    const response = await request(outEndpoint, inEndpoint, payload, 1 + readLength);
    const status = response[0] ?? 0xff;
    return { ok: status === 0, status, data: toHex(response.slice(1)) };
  }

  if (command === "memory-read") {
    const address = Number(options.address);
    const memoryAddress = Number(options.memoryAddress ?? options["memory-address"]);
    const wordSize = Number(options.wordSize ?? options["word-size"] ?? 4);
    const payload = [
      9,
      0,
      1,
      2 * (address & 0x7f) + 1,
      wordSize,
      ...intToBytes(memoryAddress, wordSize),
      wordSize,
    ];
    const response = await request(outEndpoint, inEndpoint, payload, 1 + wordSize);
    const status = response[0] ?? 0xff;
    const data = response.slice(1);
    return { ok: status === 0, status, data: toHex(data), value: bytesToInt(data) };
  }

  if (command === "memory-write") {
    const address = Number(options.address);
    const memoryAddress = Number(options.memoryAddress ?? options["memory-address"]);
    const value = Number(options.value);
    const wordSize = Number(options.wordSize ?? options["word-size"] ?? 4);
    const payload = [
      9,
      0,
      1,
      2 * (address & 0x7f),
      2 * wordSize,
      ...intToBytes(memoryAddress, wordSize),
      ...intToBytes(value, wordSize),
    ];
    const response = await request(outEndpoint, inEndpoint, payload, 1);
    const status = response[0] ?? 0xff;
    return { ok: status === 0, status };
  }

  throw new Error(`Unknown command: ${command}`);
}

async function serve() {
  const timeoutMs = Number(argValue("--timeout-ms", "1000"));
  const { device, outEndpoint, inEndpoint } = await setupDevice(timeoutMs);
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

  process.on("exit", () => {
    try {
      device.close();
    } catch {}
  });

  for await (const line of rl) {
    if (!line.trim()) continue;
    try {
      const payload = JSON.parse(line);
      const result = await executeCommand(payload.command, payload, outEndpoint, inEndpoint);
      process.stdout.write(`${JSON.stringify(result)}\n`);
    } catch (error) {
      process.stdout.write(
        `${JSON.stringify({ ok: false, error: String(error && error.message ? error.message : error) })}\n`,
      );
    }
  }

  device.close();
}

async function main() {
  const command = process.argv[2];
  if (command === "serve") {
    await serve();
    return;
  }

  const { device, outEndpoint, inEndpoint } = await setupDevice(argValue("--timeout-ms", "1000"));
  try {
    if (command === "identify") {
      console.log(JSON.stringify(await executeCommand("identify", {}, outEndpoint, inEndpoint)));
      return;
    }

    if (command === "cp210x-read-latch") {
      console.log(
        JSON.stringify(
          await executeCommand("cp210x-read-latch", { length: argValue("--length", "2") }, outEndpoint, inEndpoint),
        ),
      );
      return;
    }

    if (command === "cp210x-get-part-number") {
      console.log(JSON.stringify(await executeCommand("cp210x-get-part-number", {}, outEndpoint, inEndpoint)));
      return;
    }

    if (command === "transfer") {
      console.log(
        JSON.stringify(
          await executeCommand(
            "transfer",
            {
              address: argValue("--address"),
              write: argValue("--write", ""),
              read: argValue("--read", "0"),
              addressMode: argValue("--address-mode", "xdp_8bit"),
            },
            outEndpoint,
            inEndpoint,
          ),
        ),
      );
      return;
    }

    if (command === "memory-read") {
      console.log(
        JSON.stringify(
          await executeCommand(
            "memory-read",
            {
              address: argValue("--address"),
              memoryAddress: argValue("--memory-address"),
              wordSize: argValue("--word-size", "4"),
            },
            outEndpoint,
            inEndpoint,
          ),
        ),
      );
      return;
    }

    if (command === "memory-write") {
      console.log(
        JSON.stringify(
          await executeCommand(
            "memory-write",
            {
              address: argValue("--address"),
              memoryAddress: argValue("--memory-address"),
              value: argValue("--value"),
              wordSize: argValue("--word-size", "4"),
            },
            outEndpoint,
            inEndpoint,
          ),
        ),
      );
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
