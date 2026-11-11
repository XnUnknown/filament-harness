const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("filament", {
  command: (method, params) => ipcRenderer.invoke("filament:command", method, params),
  selectWorkspace: () => ipcRenderer.invoke("filament:select-workspace"),
  openPath: (target) => ipcRenderer.invoke("filament:open-path", target),
  onEvent: (callback) => {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on("filament:event", handler);
    return () => ipcRenderer.removeListener("filament:event", handler);
  },
  platform: process.platform,
});
