const transcript = document.getElementById("transcript");
const status = document.getElementById("status");
const form = document.getElementById("composer");
const input = document.getElementById("message");
const modeSelect = document.getElementById("mode");

function renderBubble(text, kind = "assistant") {
  const div = document.createElement("div");
  div.className = `bubble ${kind}`;
  div.textContent = text;
  transcript.appendChild(div);
  transcript.scrollTop = transcript.scrollHeight;
  return div;
}

async function refreshStatus() {
  try {
    const result = await window.filament.command("ping", {});
    status.textContent = `Ready: ${result.version}`;
  } catch (error) {
    status.textContent = "Backend error: " + error.message;
  }
}

async function sendMessage(message) {
  renderBubble(message, "user");
  status.textContent = "Planning...";
  try {
    const result = await window.filament.command("chat", { message, mode: modeSelect.value });
    if (result.type === "chat") {
      renderBubble(result.content, "assistant");
    } else {
      renderBubble(`Plan: ${result.summary}\n\n` + (result.tasks || []).map(t => `- ${t.id}: ${t.task} -> ${t.target}`).join("\n"), "assistant");
      if (result.execution) {
        renderBubble(`Execution:\n${JSON.stringify(result.execution, null, 2)}`, "assistant");
      }
    }
  } catch (error) {
    renderBubble("Error: " + error.message, "assistant");
  } finally {
    status.textContent = "Ready";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  await sendMessage(message);
});

document.getElementById("new-session").addEventListener("click", async () => {
  try {
    const result = await window.filament.command("new_session", { mode: modeSelect.value });
    renderBubble(`New session: ${result.session_id}`, "system");
  } catch (error) {
    renderBubble("Error: " + error.message, "system");
  }
});

document.getElementById("list-sessions").addEventListener("click", async () => {
  try {
    const result = await window.filament.command("list_sessions", {});
    renderBubble(JSON.stringify(result, null, 2), "system");
  } catch (error) {
    renderBubble("Error: " + error.message, "system");
  }
});

document.getElementById("compress").addEventListener("click", async () => {
  try {
    const result = await window.filament.command("compress_context", {});
    renderBubble("Context compressed: " + result.compressed, "system");
  } catch (error) {
    renderBubble("Error: " + error.message, "system");
  }
});

refreshStatus();
