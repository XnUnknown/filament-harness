const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("filament", {
  command: (method, params) => ipcRenderer.invoke("filament:command", method, params || {}),
});
