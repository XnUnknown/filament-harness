const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const readline = require("readline");
const path = require("path");

let backend = null;
let backendReady = false;
let pending = new Map();
let nextId = 1;

function startBackend() {
  const repoRoot = path.resolve(__dirname, "..");
  const python = process.env.FILAMENT_PYTHON || "python";
  backend = spawn(python, ["-m", "filament.backend_server"], {
    cwd: repoRoot,
    env: { ...process.env, PYTHONPATH: repoRoot },
    stdio: ["pipe", "pipe", "pipe"],
  });

  const rl = readline.createInterface({ input: backend.stdout });
  rl.on("line", (line) => {
    let payload;
    try {
      payload = JSON.parse(line);
    } catch (error) {
      console.error("Invalid backend JSON:", line);
      return;
    }
    const pendingRequest = pending.get(payload.id);
    if (!pendingRequest) return;
    pending.delete(payload.id);
    if (payload.ok) pendingRequest.resolve(payload.result);
    else pendingRequest.reject(new Error(payload.error || "Backend error"));
  });

  backend.stderr.on("data", (data) => {
    console.error("backend:", data.toString());
  });
  backend.on("exit", (code) => {
    backendReady = false;
    console.error("backend exited:", code);
    for (const request of pending.values()) request.reject(new Error("Backend exited"));
    pending.clear();
  });
  backendReady = true;
}

function sendCommand(method, params = {}) {
  if (!backend || !backendReady) throw new Error("Backend is not ready");
  const id = String(nextId++);
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    backend.stdin.write(JSON.stringify({ id, method, params }) + "\n");
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(new Error("Backend request timed out"));
      }
    }, 120000);
  });
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1100,
    height: 760,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  window.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(() => {
  startBackend();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (backend) backend.kill();
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("filament:command", async (_event, method, params) => {
  return sendCommand(method, params);
});
