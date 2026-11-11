const { app, BrowserWindow, ipcMain, dialog, Menu, shell } = require("electron");
const { spawn } = require("child_process");
const readline = require("readline");
const path = require("path");

let backend = null;
let backendReady = false;
let mainWindow = null;
const pending = new Map();
let nextId = 1;

const LONG_METHODS = new Set(["chat", "answer", "skip_question"]);

function windowRef() {
  return mainWindow && !mainWindow.isDestroyed() ? mainWindow : null;
}

function startBackend() {
  const repoRoot = path.resolve(__dirname, "..");
  const python = process.env.FILAMENT_PYTHON || "python";
  backend = spawn(python, ["-m", "filament.backend_server"], {
    cwd: repoRoot,
    env: { ...process.env, PYTHONPATH: repoRoot, PYTHONUNBUFFERED: "1" },
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
    // Notifications have an "event" key and no request id.
    if (payload && payload.event) {
      const win = windowRef();
      if (win) win.webContents.send("filament:event", payload);
      return;
    }
    const request = pending.get(String(payload.id));
    if (!request) return;
    pending.delete(String(payload.id));
    if (request.timer) clearTimeout(request.timer);
    if (payload.ok) request.resolve(payload.result);
    else request.reject(new Error(payload.error || "Backend error"));
  });

  backend.stderr.on("data", (data) => {
    console.error("backend:", data.toString());
  });
  backend.on("exit", (code) => {
    backendReady = false;
    for (const request of pending.values()) {
      if (request.timer) clearTimeout(request.timer);
      request.reject(new Error("Backend exited"));
    }
    pending.clear();
    const win = windowRef();
    if (win) {
      win.webContents.send("filament:event", {
        event: "backend_exit",
        code,
      });
    }
  });
  backendReady = true;
}

function sendCommand(method, params = {}) {
  if (!backend || !backendReady) throw new Error("Backend is not ready");
  const id = String(nextId++);
  const timeoutMs = LONG_METHODS.has(method) ? 30 * 60 * 1000 : 60 * 1000;
  return new Promise((resolve, reject) => {
    const timer = timeoutMs
      ? setTimeout(() => {
          if (pending.has(id)) {
            pending.delete(id);
            reject(new Error(`Backend request timed out (${method})`));
          }
        }, timeoutMs)
      : null;
    pending.set(id, { resolve, reject, timer });
    try {
      backend.stdin.write(JSON.stringify({ id, method, params }) + "\n");
    } catch (error) {
      pending.delete(id);
      if (timer) clearTimeout(timer);
      reject(error);
    }
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 940,
    minHeight: 600,
    backgroundColor: "#0e1015",
    title: "Filament",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
  mainWindow.webContents.on("did-fail-load", (_event, code, description, url) => {
    console.error("renderer load failed:", code, description, url);
  });
  mainWindow.webContents.on("console-message", (_event, level, message, line, source) => {
    console.log("renderer console:", level, message, `${source}:${line}`);
  });
  mainWindow.webContents.on("preload-error", (_event, preloadPath, error) => {
    console.error("preload error:", preloadPath, error);
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
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

app.on("before-quit", () => {
  if (backend) backend.kill();
});

ipcMain.handle("filament:select-workspace", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"],
    title: "Select a workspace",
  });
  if (result.canceled || !result.filePaths.length) return null;
  const selected = result.filePaths[0];
  const backendResult = await sendCommand("select_workspace", { path: selected });
  return { path: selected, result: backendResult };
});

ipcMain.handle("filament:command", async (_event, method, params) => {
  return sendCommand(method, params);
});

ipcMain.handle("filament:open-path", async (_event, target) => {
  if (typeof target !== "string" || !target) return { ok: false, error: "No path" };
  try {
    const result = await shell.openPath(target);
    if (result) return { ok: false, error: result };
    return { ok: true };
  } catch (error) {
    return { ok: false, error: String(error) };
  }
});
