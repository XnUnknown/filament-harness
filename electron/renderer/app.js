/* Filament renderer: event-driven agent UI. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const transcript = $("transcript");
  const messageInput = $("message");
  const emptyTemplate = $("empty-state").outerHTML;

  const S = {
    workspace: null,
    sessionId: null,
    title: "New session",
    mode: "build",
    busy: false,
    status: "ready",
    contextTokens: 0,
    contextLimit: 1,
    todos: [],
    files: [],
    subagents: new Map(),
    toolCards: new Map(),
    assistantTurns: new Map(),
    pendingQuestion: null,
    heartbeatStart: 0,
    heartbeatTimer: null,
    config: null,
    lastUserMessage: "",
    lastAssistantText: "",
  };

  const HIDDEN_TOOL_CARDS = new Set(["finish", "ask_user", "spawn_agent", "spawn_agents"]);

  // ---------------------------------------------------------------- helpers
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function truncate(value, limit) {
    const flat = String(value == null ? "" : value).replace(/\s+/g, " ").trim();
    return flat.length <= limit ? flat : flat.slice(0, limit - 1) + "…";
  }

  function formatDuration(ms) {
    if (typeof ms !== "number" || ms < 0) return "";
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  function formatTokens(value) {
    const n = Number(value) || 0;
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}m`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
    return String(n);
  }

  function relativeTime(ts) {
    const seconds = Math.max(0, Math.floor(Date.now() / 1000 - Number(ts || 0)));
    if (seconds < 60) return "just now";
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  }

  function normalizeMode(mode) {
    const value = String(mode || "").toLowerCase();
    if (value === "plan") return "plan";
    return "build";
  }

  function scrollIfNearBottom(force) {
    const near = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 160;
    if (force || near) {
      requestAnimationFrame(() => {
        transcript.scrollTop = transcript.scrollHeight;
      });
    }
  }

  function toast(text, timeout = 3200) {
    const node = $("toast");
    node.textContent = text;
    node.classList.remove("hidden");
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => node.classList.add("hidden"), timeout);
  }

  function setStatus(text) {
    S.status = text;
    $("status").textContent = text;
  }

  function setBusy(busy) {
    S.busy = busy;
    $("send").disabled = busy || Boolean(S.pendingQuestion);
    $("cancel").classList.toggle("hidden", !busy);
    $("cancel").disabled = false;
    messageInput.disabled = false;
    $("hint").textContent = busy
      ? "Working. Stop ends the run after the current model call."
      : S.pendingQuestion
        ? "Answer the question above to continue."
        : "The agent reads, edits, runs commands, and delegates.";
    if (!busy) {
      stopHeartbeat();
      if (S.status.startsWith("thinking")) setStatus("ready");
    }
  }

  function startHeartbeat() {
    stopHeartbeat();
    S.heartbeatStart = Date.now();
    S.heartbeatTimer = setInterval(() => {
      if (!S.busy) return;
      const seconds = Math.floor((Date.now() - S.heartbeatStart) / 1000);
      if (seconds >= 1) setStatus(`thinking · ${seconds}s`);
    }, 1000);
  }

  function stopHeartbeat() {
    if (S.heartbeatTimer) clearInterval(S.heartbeatTimer);
    S.heartbeatTimer = null;
  }

  function hideEmpty() {
    const empty = $("empty-state");
    if (empty) empty.remove();
  }

  function showEmpty() {
    transcript.innerHTML = emptyTemplate;
    S.assistantTurns.clear();
    S.toolCards.clear();
    S.subagents.clear();
    renderSubagentList();
  }

  function clearTranscript() {
    transcript.innerHTML = "";
    S.assistantTurns.clear();
    S.toolCards.clear();
    S.subagents.clear();
    S.lastAssistantText = "";
    S.lastUserMessage = "";
    renderSubagentList();
  }

  // ------------------------------------------------------------- markdown
  function inline(text) {
    let out = esc(text);
    out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
    out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    return out;
  }

  function markdown(text) {
    const source = String(text || "").replace(/\r\n/g, "\n");
    const parts = source.split(/```/);
    let html = "";
    for (let index = 0; index < parts.length; index += 1) {
      if (index % 2 === 1) {
        const chunk = parts[index];
        const newline = chunk.indexOf("\n");
        const lang = newline === -1 ? "" : chunk.slice(0, newline).trim();
        const code = newline === -1 ? chunk : chunk.slice(newline + 1);
        html += `<pre class="code"><code data-lang="${esc(lang)}">${esc(code.replace(/\n$/, ""))}</code></pre>`;
        continue;
      }
      const lines = parts[index].split("\n");
      let listOpen = false;
      for (const line of lines) {
        const bullet = line.match(/^\s*[-*]\s+(.*)$/);
        const heading = line.match(/^(#{1,4})\s+(.*)$/);
        if (bullet) {
          if (!listOpen) {
            html += "<ul>";
            listOpen = true;
          }
          html += `<li>${inline(bullet[1])}</li>`;
          continue;
        }
        if (listOpen) {
          html += "</ul>";
          listOpen = false;
        }
        if (heading) {
          html += `<div class="md-h md-h${heading[1].length}">${inline(heading[2])}</div>`;
        } else if (line.trim() === "") {
          html += "<br>";
        } else {
          html += `<div>${inline(line)}</div>`;
        }
      }
      if (listOpen) html += "</ul>";
    }
    return html;
  }

  // ------------------------------------------------------------ transcript
  function addUser(text) {
    hideEmpty();
    const row = el("div", "row user");
    const bubble = el("div", "bubble user-bubble");
    bubble.textContent = text;
    row.appendChild(bubble);
    transcript.appendChild(row);
    scrollIfNearBottom(true);
  }

  function addSystem(text, kind = "") {
    addSystemTo(transcript, text, kind);
  }

  function addSystemTo(container, text, kind = "") {
    hideEmpty();
    const row = el("div", `system-line ${kind}`.trim());
    row.textContent = text;
    container.appendChild(row);
    scrollIfNearBottom();
  }

  function addAssistantStatic(text) {
    hideEmpty();
    const row = el("div", "row assistant");
    const avatar = el("div", "avatar", "F");
    const body = el("div", "assistant-body");
    const content = el("div", "assistant-text");
    content.innerHTML = markdown(text);
    body.appendChild(content);
    row.append(avatar, body);
    transcript.appendChild(row);
    S.lastAssistantText = String(text || "").trim();
    scrollIfNearBottom();
  }

  function addDone(text) {
    hideEmpty();
    const card = el("div", "done-card");
    card.innerHTML = `<span class="done-check">✓</span><div class="done-text">${markdown(text)}</div>`;
    transcript.appendChild(card);
    scrollIfNearBottom(true);
  }

  function containerFor(event) {
    const id = event.subagent_id;
    if (id && S.subagents.has(id)) return S.subagents.get(id).stream;
    return transcript;
  }

  function ensureAssistant(container) {
    if (S.assistantTurns.has(container)) return S.assistantTurns.get(container);
    const row = el("div", "row assistant");
    const avatar = el("div", "avatar", "F");
    const body = el("div", "assistant-body");
    const text = el("div", "assistant-text");
    const typing = el("div", "typing");
    typing.innerHTML = "<span></span><span></span><span></span>";
    body.append(text, typing);
    row.append(avatar, body);
    container.appendChild(row);
    const turn = { row, body, text, typing, sawDelta: false };
    S.assistantTurns.set(container, turn);
    scrollIfNearBottom(true);
    return turn;
  }

  function toolKey(event) {
    return `${event.subagent_id || ""}::${event.id || ""}`;
  }

  const TOOL_LABELS = {
    read_file: "read",
    write_file: "write",
    edit_file: "edit",
    multi_edit: "edit",
    list_dir: "list",
    glob: "glob",
    grep: "grep",
    run_shell: "shell",
    check_python: "check",
    todo_write: "todo",
    ask_user: "ask",
    spawn_agent: "agent",
    spawn_agents: "agent",
    finish: "finish",
  };

  function toolSummary(name, args) {
    args = args || {};
    if (["read_file", "write_file", "edit_file", "multi_edit", "check_python"].includes(name)) {
      return args.path || "";
    }
    if (name === "run_shell") return truncate(args.command || "", 90);
    if (name === "grep" || name === "glob") {
      return `${args.pattern || ""}${args.path && args.path !== "." ? ` in ${args.path}` : ""}`;
    }
    if (name === "list_dir") return args.path || ".";
    if (name === "spawn_agent") return truncate(args.task || "", 90);
    if (name === "spawn_agents") return `${(args.tasks || []).length} parallel tasks`;
    if (name === "ask_user") return truncate(args.question || "", 90);
    if (name === "todo_write") return `${(args.todos || []).length} items`;
    if (name === "finish") return truncate(args.summary || "", 90);
    return "";
  }

  function addToolCard(event, resultEvent) {
    hideEmpty();
    const name = event.name || "tool";
    const args = event.args || {};
    const card = el("div", "tool running");
    const head = el("button", "tool-head");
    head.type = "button";
    head.innerHTML =
      `<span class="tool-badge">${esc(TOOL_LABELS[name] || name)}</span>` +
      `<span class="tool-name">${esc(name)}</span>` +
      `<span class="tool-summary">${esc(toolSummary(name, args))}</span>` +
      `<span class="tool-status">running</span>` +
      `<span class="chev">▸</span>`;
    const body = el("div", "tool-body hidden");
    const argsPre = el("pre", "tool-args");
    argsPre.textContent = JSON.stringify(args, null, 2);
    body.appendChild(argsPre);
    head.addEventListener("click", () => body.classList.toggle("hidden"));
    card.append(head, body);
    containerFor(event).appendChild(card);

    const entry = { card, status: head.querySelector(".tool-status"), body, result: null };
    S.toolCards.set(toolKey(event), entry);
    if (resultEvent) finishToolCard(entry, resultEvent);
    scrollIfNearBottom();
    return entry;
  }

  function finishToolCard(entry, event) {
    const ok = event.ok !== false;
    entry.card.classList.remove("running");
    entry.card.classList.add(ok ? "ok" : "failed");
    const duration = formatDuration(event.duration_ms);
    entry.status.textContent = ok ? duration || "done" : "failed";
    if (!entry.result) {
      entry.result = el("pre", "tool-result");
      entry.body.appendChild(entry.result);
    }
    entry.result.textContent = String(event.content || "");
    entry.result.classList.remove("hidden");
    if (!ok) entry.body.classList.remove("hidden");
  }

  function addSubagent(event) {
    hideEmpty();
    const card = el("div", "subagent running");
    const head = el("button", "subagent-head");
    head.type = "button";
    head.innerHTML =
      `<span class="tool-badge agent">agent</span>` +
      `<span class="subagent-title">${esc(truncate(event.task || "subagent", 100))}</span>` +
      `<span class="subagent-status">running</span>` +
      `<span class="chev">▸</span>`;
    const body = el("div", "subagent-body");
    const stream = el("div", "subagent-stream");
    body.appendChild(stream);
    head.addEventListener("click", () => body.classList.toggle("hidden"));
    card.append(head, body);
    // Nested agents belong inside their parent's stream; the top-level case
    // still resolves to the transcript because containerFor falls back to it.
    containerFor(event).appendChild(card);
    S.subagents.set(event.id || `agent-${S.subagents.size + 1}`, {
      card,
      status: head.querySelector(".subagent-status"),
      stream,
      task: event.task || "",
      sessionId: event.child_session_id || "",
    });
    renderSubagentList();
    scrollIfNearBottom();
  }

  function finishSubagent(event) {
    const agent = S.subagents.get(event.id);
    if (!agent) return;
    agent.card.classList.remove("running");
    agent.card.classList.add(event.status === "done" ? "ok" : "failed");
    agent.status.textContent = `${event.status || "done"} · ${event.steps || 0} steps · ${formatTokens(event.tokens)} tokens`;
    const summary = String(event.content || "").trim();
    // A child that ends with a plain answer has already streamed that text into
    // its card. Appending result.content as well showed the same paragraph
    // twice; only add a summary the transcript does not already contain.
    const turns = agent.stream.querySelectorAll(".assistant-text");
    const lastText = turns.length
      ? String(turns[turns.length - 1].textContent || "").replace(/\s+/g, " ").trim()
      : "";
    if (summary && summary.replace(/\s+/g, " ").trim() !== lastText) {
      const note = el("div", "subagent-result");
      note.innerHTML = markdown(summary);
      agent.stream.appendChild(note);
    }
    renderSubagentList();
  }

  function renderQuestion(question) {
    if (!question || !question.id) return;
    const existing = Array.from(document.querySelectorAll(".question")).find(
      (node) => node.dataset.questionId === question.id
    );
    if (existing) {
      S.pendingQuestion = question;
      setBusy(S.busy);
      return;
    }
    S.pendingQuestion = question;
    hideEmpty();
    const card = el("div", "question");
    card.dataset.questionId = question.id;
    card.innerHTML =
      `<div class="question-kicker">Needs your input</div>` +
      `<div class="question-text">${esc(question.question || "")}</div>` +
      `<div class="question-options"></div>` +
      `<form class="question-form"><input type="text" placeholder="Type an answer" /><button type="submit">Answer</button></form>`;
    const optionsBox = card.querySelector(".question-options");
    (question.options || []).forEach((option) => {
      const button = el("button", "option", option);
      button.type = "button";
      button.addEventListener("click", () => answerQuestion(option));
      optionsBox.appendChild(button);
    });
    const form = card.querySelector("form");
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const value = form.querySelector("input").value.trim();
      if (value) answerQuestion(value);
    });
    transcript.appendChild(card);
    scrollIfNearBottom(true);
    setStatus("waiting for you");
    setBusy(S.busy);
  }

  function markQuestionAnswered(card, answer) {
    if (!card) return;
    card.classList.add("answered");
    const form = card.querySelector("form");
    if (form) form.remove();
    const options = card.querySelector(".question-options");
    if (options) options.remove();
    S.pendingQuestion = null;
    setBusy(S.busy);
  }

  // --------------------------------------------------------------- inspector
  function renderTodos(todos) {
    S.todos = Array.isArray(todos) ? todos : [];
    const list = $("todo-list");
    list.innerHTML = "";
    $("todo-count").textContent = String(S.todos.length);
    if (!S.todos.length) {
      list.appendChild(el("div", "muted small", "No todos yet"));
      return;
    }
    for (const todo of S.todos) {
      const row = el("div", `todo ${todo.status || "pending"}`);
      const mark = todo.status === "completed" ? "✓" : todo.status === "in_progress" ? "•" : "○";
      row.innerHTML = `<span class="todo-mark">${mark}</span><span>${esc(todo.content || "")}</span>`;
      list.appendChild(row);
    }
  }

  function renderSubagentList() {
    const list = $("subagent-list");
    list.innerHTML = "";
    $("subagent-count").textContent = String(S.subagents.size);
    if (!S.subagents.size) {
      list.appendChild(el("div", "muted small", "None running"));
      return;
    }
    for (const agent of S.subagents.values()) {
      const row = el("div", "subagent-row");
      row.textContent = truncate(agent.task || "subagent", 44);
      row.title = agent.sessionId || "";
      list.appendChild(row);
    }
  }

  function renderFiles(files) {
    S.files = Array.isArray(files) ? files : [];
    const list = $("file-list");
    list.innerHTML = "";
    if (!S.files.length) {
      list.appendChild(el("div", "muted small", "None"));
      return;
    }
    for (const path of S.files.slice(0, 24)) {
      const button = el("button", "file-row");
      button.type = "button";
      button.textContent = path;
      button.title = `Open ${path}`;
      button.addEventListener("click", async () => {
        if (!S.workspace) return;
        const result = await window.filament.openPath(`${S.workspace}/${path}`);
        if (!result.ok) toast(result.error || "Could not open the file");
      });
      list.appendChild(button);
    }
  }

  function renderContext(tokens, limit) {
    S.contextTokens = Number(tokens) || 0;
    S.contextLimit = Math.max(1, Number(limit) || 1);
    const pct = Math.min(100, Math.round((S.contextTokens / S.contextLimit) * 100));
    $("context-value").textContent = `${pct}%`;
    $("context-bar").style.width = `${pct}%`;
    $("context-bar").classList.toggle("warn", pct >= 70);
    $("context-bar").classList.toggle("hot", pct >= 90);
    $("context-detail").textContent = `${formatTokens(S.contextTokens)} / ${formatTokens(S.contextLimit)} tokens`;
  }

  function renderSessions(sessions) {
    const list = $("session-list");
    list.innerHTML = "";
    if (!sessions || !sessions.length) {
      list.appendChild(el("div", "muted small", "No sessions yet"));
      return;
    }
    for (const session of sessions) {
      const button = el("button", "session-item");
      button.type = "button";
      if (session.session_id === S.sessionId) button.classList.add("active");
      button.dataset.session = session.session_id;
      button.innerHTML =
        `<span class="session-dot ${esc(session.status || "idle")}"></span>` +
        `<span class="session-body"><span class="session-title">${esc(truncate(session.title || session.session_id, 52))}</span>` +
        `<span class="session-meta">${esc(relativeTime(session.updated_at))} · ${esc(normalizeMode(session.mode))}</span></span>`;
      button.addEventListener("click", () => loadSession(session.session_id));
      list.appendChild(button);
    }
  }

  function applyWorkspace(path) {
    S.workspace = path || null;
    const name = path ? path.split(/[\\/]/).filter(Boolean).pop() : "Choose a folder";
    $("workspace-name").textContent = name || "Choose a folder";
    $("workspace-path").textContent = path || "No folder selected";
    $("workspace-path").title = path || "";
  }

  function applyProvider(provider) {
    if (!provider) return;
    const kind = provider.kind || "auto";
    const model = provider.model || "";
    $("provider-label").textContent = `${kind}${model ? ` · ${model}` : ""}`;
    $("provider-label").title = $("provider-label").textContent;
    $("provider-dot").className = `dot ${kind === "mock" ? "mock" : "live"}`;
  }

  function applyState(state) {
    if (!state) return;
    if (state.session_id) S.sessionId = state.session_id;
    const messageCount = Number(state.messages) || 0;
    S.title = messageCount > 0 ? state.title || "Session" : "New session";
    S.mode = state.mode || S.mode;
    $("session-title").textContent = S.title || "New session";
    $("session-id").textContent = S.sessionId ? `session ${S.sessionId.slice(-8)}` : "no session";
    setStatus(state.status === "idle" ? "ready" : state.status || "ready");
    document.querySelectorAll("#mode-toggle button").forEach((button) => {
      button.classList.toggle("active", button.dataset.mode === S.mode);
    });
    renderContext(state.tokens, state.limit);
    renderTodos(state.todos);
    renderFiles(state.files_changed);
    if (state.provider) applyProvider(state.provider);
    S.pendingQuestion = state.pending_question || null;
    // Recompute the composer from the state we just adopted. Without this, a
    // session switch after answering left the send button disabled because
    // only the question card used to clear that flag.
    setBusy(S.busy);
  }

  // -------------------------------------------------------------- turn flow
  function handleTurnResponse(result) {
    if (!result) return;
    if (result.state) applyState(result.state);
    if (result.type === "question" && result.question) {
      renderQuestion(result.question);
      return;
    }
    if (result.status && result.status !== "done" && result.content) {
      addSystem(result.content, result.status);
    }
    setStatus(result.status === "done" ? "ready" : result.status || "ready");
  }

  async function sendMessage(text) {
    const message = String(text || "").trim();
    if (!message || S.busy) return;
    if (S.pendingQuestion) {
      toast("Answer the pending question first");
      return;
    }
    S.lastUserMessage = message;
    addUser(message);
    messageInput.value = "";
    resizeComposer();
    setBusy(true);
    setStatus("thinking");
    startHeartbeat();
    try {
      const result = await window.filament.command("chat", { message, mode: S.mode });
      handleTurnResponse(result);
      refreshSessions();
    } catch (error) {
      addSystem(`Request failed: ${error.message}`, "error");
      setStatus("error");
    } finally {
      setBusy(false);
      scrollIfNearBottom();
    }
  }

  async function answerQuestion(answer) {
    if (S.busy || !S.pendingQuestion) return;
    const question = S.pendingQuestion;
    const card = document.querySelector(`.question[data-question-id="${question.id}"]`);
    markQuestionAnswered(card, answer);
    addSystem(`You: ${answer}`, "answer");
    setBusy(true);
    setStatus("thinking");
    startHeartbeat();
    try {
      const result = await window.filament.command("answer", {
        question_id: question.id,
        text: answer,
      });
      handleTurnResponse(result);
      refreshSessions();
    } catch (error) {
      addSystem(`Answer failed: ${error.message}`, "error");
      setStatus("error");
    } finally {
      setBusy(false);
      scrollIfNearBottom();
    }
  }

  async function loadSession(sessionId) {
    if (S.busy || !sessionId || sessionId === S.sessionId) return;
    try {
      const result = await window.filament.command("load_session", { session_id: sessionId });
      clearTranscript();
      applyState(result.state);
      renderHistory(result.messages || []);
      if (result.state && result.state.pending_question) renderQuestion(result.state.pending_question);
      if (!result.messages || !result.messages.length) showEmpty();
      refreshSessions();
      setStatus("ready");
    } catch (error) {
      toast(`Could not load session: ${error.message}`);
    }
  }

  function renderHistory(messages) {
    const results = new Map();
    for (const message of messages) {
      if (message.role === "tool" && message.tool_call_id) results.set(message.tool_call_id, message);
    }
    for (const message of messages) {
      if (message.role === "user") {
        const text = String(message.content || "");
        if (text.startsWith("[context summary]") || text.startsWith("[Earlier conversation was compacted")) {
          addSystem("Earlier context was compacted.", "compact");
        } else {
          addUser(text);
        }
        continue;
      }
      if (message.role === "assistant") {
        const text = String(message.content || "");
        const calls = message.tool_calls || [];
        if (text.trim()) addAssistantStatic(text);
        for (const call of calls) {
          if (call.name === "finish") {
            // Keep each done card at the turn where finish happened. Collecting
            // only the last summary and appending it at the end moved an old
            // finished turn below later messages.
            const summary = String((call.arguments || {}).summary || "").trim();
            if (summary && summary !== S.lastAssistantText) {
              addDone(summary);
              S.lastAssistantText = summary;
            }
            continue;
          }
          if (call.name === "ask_user") {
            const answer = results.get(call.id);
            if (answer) addSystem(`You answered: ${answer.content}`, "answer");
            continue;
          }
          const result = results.get(call.id);
          addToolCard(
            { id: call.id, name: call.name, args: call.arguments || {} },
            result
              ? {
                  ok: !result.is_error,
                  content: result.content,
                  duration_ms: undefined,
                }
              : null
          );
        }
      }
    }
    scrollIfNearBottom(true);
  }

  async function refreshSessions() {
    try {
      const sessions = await window.filament.command("list_sessions", {});
      renderSessions(sessions);
    } catch (error) {
      console.error(error);
    }
  }

  async function newSession() {
    if (S.busy) return;
    try {
      const result = await window.filament.command("new_session", { mode: S.mode });
      S.sessionId = result.session.session_id;
      S.title = "New session";
      clearTranscript();
      showEmpty();
      applyState(result.state);
      setStatus("ready");
      refreshSessions();
      messageInput.focus();
    } catch (error) {
      toast(`Could not create a session: ${error.message}`);
    }
  }

  async function selectWorkspace() {
    if (S.busy) return;
    try {
      const result = await window.filament.selectWorkspace();
      if (!result) return;
      applyWorkspace(result.path);
      const state = await window.filament.command("get_state", {});
      applyWorkspace(state.workspace);
      applyState(state.state);
      applyProvider(state.state.provider);
      clearTranscript();
      showEmpty();
      if (state.state.pending_question) renderQuestion(state.state.pending_question);
      refreshSessions();
      toast("Workspace loaded");
    } catch (error) {
      toast(`Could not load the workspace: ${error.message}`);
    }
  }

  // ---------------------------------------------------------------- settings
  function fillSettings(config) {
    if (!config) return;
    S.config = config;
    $("cfg-kind").value = config.llm.kind || "auto";
    $("cfg-model").value = config.llm.model || "";
    $("cfg-base-url").value = config.llm.base_url || "";
    $("cfg-api-key").value = "";
    $("cfg-timeout").value = config.llm.timeout || 120;
    $("cfg-max-output").value = config.llm.max_output_tokens || 4096;
    $("cfg-max-tokens").value = config.max_tokens || 120000;
    $("cfg-max-steps").value = config.max_steps || 40;
    $("cfg-subagent-tokens").value = config.subagent_max_tokens || 32000;
    $("cfg-subagent-steps").value = config.subagent_max_steps || 24;
    $("cfg-stream").checked = config.llm.stream !== false;
    $("cfg-llm-compact").checked = config.compact_with_llm !== false;
    $("settings-error").textContent = "";
  }

  async function openSettings() {
    try {
      const config = await window.filament.command("get_config", {});
      fillSettings(config);
      $("settings-modal").classList.remove("hidden");
    } catch (error) {
      toast(`Could not read settings: ${error.message}`);
    }
  }

  function numberValue(id, fallback) {
    const value = Number($(id).value);
    return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
  }

  async function saveSettings() {
    try {
      $("settings-error").textContent = "";
      const base = $("cfg-base-url").value.trim();
      const updates = {
        max_tokens: numberValue("cfg-max-tokens", 120000),
        max_steps: numberValue("cfg-max-steps", 40),
        subagent_max_tokens: numberValue("cfg-subagent-tokens", 32000),
        subagent_max_steps: numberValue("cfg-subagent-steps", 24),
        compact_with_llm: $("cfg-llm-compact").checked,
        llm: {
          kind: $("cfg-kind").value,
          model: $("cfg-model").value.trim() || "llama3.2",
          base_url: base,
          timeout: numberValue("cfg-timeout", 120),
          stream: $("cfg-stream").checked,
          max_output_tokens: numberValue("cfg-max-output", 4096),
        },
      };
      const key = $("cfg-api-key").value.trim();
      if (key) updates.llm.api_key = key;
      const config = await window.filament.command("set_config", { config: updates });
      fillSettings(config);
      applyProvider(config.llm);
      $("settings-modal").classList.add("hidden");
      toast("Settings saved");
      refreshSessions();
    } catch (error) {
      $("settings-error").textContent = error.message;
    }
  }

  // ---------------------------------------------------------------- events
  function handleEvent(event) {
    if (!event) return;
    const kind = event.event || event.type;
    if (!kind) return;

    if (event.session_id && S.sessionId && event.session_id !== S.sessionId && !event.subagent_id) {
      // A different main session started (new/load/workspace change).
      S.sessionId = event.session_id;
    }
    if (event.session_id && !S.sessionId) S.sessionId = event.session_id;
    if (event.title) {
      S.title = event.title;
      $("session-title").textContent = S.title;
    }
    if (event.mode) {
      S.mode = event.mode;
      document.querySelectorAll("#mode-toggle button").forEach((button) => {
        button.classList.toggle("active", button.dataset.mode === S.mode);
      });
    }

    const container = containerFor(event);
    const isChild = Boolean(event.subagent_id);
    switch (kind) {
      case "delta": {
        const turn = ensureAssistant(container);
        if (turn.typing) {
          turn.typing.remove();
          turn.typing = null;
        }
        turn.text.textContent += event.text || "";
        turn.sawDelta = true;
        scrollIfNearBottom();
        break;
      }
      case "assistant_start": {
        ensureAssistant(container);
        break;
      }
      case "assistant_end": {
        const turn = S.assistantTurns.get(container);
        if (!turn) break;
        if (!turn.sawDelta && event.content) turn.text.innerHTML = markdown(event.content);
        else if (turn.sawDelta) turn.text.innerHTML = markdown(turn.text.textContent);
        if (turn.typing) {
          turn.typing.remove();
          turn.typing = null;
        }
        const finalText = turn.text.textContent.trim();
        if (!finalText) turn.row.remove();
        else if (container === transcript) S.lastAssistantText = finalText;
        S.assistantTurns.delete(container);
        scrollIfNearBottom();
        break;
      }
      case "tool_start": {
        if (!HIDDEN_TOOL_CARDS.has(event.name)) addToolCard(event);
        break;
      }
      case "tool_end": {
        const entry = S.toolCards.get(toolKey(event));
        if (entry) finishToolCard(entry, event);
        break;
      }
      case "subagent_start": {
        addSubagent(event);
        break;
      }
      case "subagent_end": {
        finishSubagent(event);
        break;
      }
      case "todos": {
        if (!isChild) renderTodos(event.todos);
        break;
      }
      case "context": {
        if (!isChild) renderContext(event.tokens, event.limit);
        break;
      }
      case "compaction": {
        if (isChild) break;
        const before = Number(event.before_tokens);
        const after = Number(event.after_tokens);
        addSystem(
          `Context compacted ${Number.isFinite(before) ? formatTokens(before) : "?"} → ${Number.isFinite(after) ? formatTokens(after) : "?"} tokens`,
          "compact"
        );
        break;
      }
      case "question": {
        if (isChild) break;
        renderQuestion({ id: event.id, question: event.question, options: event.options || [] });
        break;
      }
      case "status": {
        if (isChild) break;
        const state = event.state || "";
        if (state === "thinking") setStatus("thinking");
        else if (state === "cancelling") setStatus("stopping…");
        break;
      }
      case "heartbeat": {
        if (!isChild && S.busy) setStatus(`thinking · ${Number(event.elapsed) || 0}s`);
        break;
      }
      case "error": {
        if (isChild) addSystemTo(container, String(event.message || "Agent error"), "error");
        else addSystem(String(event.message || "Agent error"), "error");
        break;
      }
      case "done": {
        if (isChild) break;
        const content = String(event.content || "").trim();
        if (event.status && event.status !== "done") {
          addSystem(`Run ended: ${event.status}`, event.status === "cancelled" ? "" : "error");
        } else if (content && content !== S.lastAssistantText) {
          addDone(content);
          S.lastAssistantText = content;
        }
        setStatus("ready");
        break;
      }
      case "backend_exit": {
        const detail = event.error ? `: ${event.error}` : "";
        addSystem(`The Python backend stopped${detail}. Restart the app.`, "error");
        setStatus("backend stopped");
        setBusy(false);
        break;
      }
      default:
        break;
    }
  }

  // ------------------------------------------------------------------- init
  function resizeComposer() {
    messageInput.style.height = "auto";
    messageInput.style.height = `${Math.min(200, Math.max(44, messageInput.scrollHeight))}px`;
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("filament-theme", theme);
    } catch (error) {
      /* localStorage may be unavailable */
    }
  }

  function bind() {
    $("composer").addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage(messageInput.value);
    });
    messageInput.addEventListener("input", resizeComposer);
    messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage(messageInput.value);
      }
    });
    $("new-session").addEventListener("click", newSession);
    $("refresh-sessions").addEventListener("click", refreshSessions);
    $("select-workspace").addEventListener("click", selectWorkspace);
    $("settings-button").addEventListener("click", openSettings);
    $("settings-close").addEventListener("click", () => $("settings-modal").classList.add("hidden"));
    $("settings-save").addEventListener("click", saveSettings);
    $("settings-modal").addEventListener("click", (event) => {
      if (event.target === $("settings-modal")) $("settings-modal").classList.add("hidden");
    });
    $("cancel").addEventListener("click", async () => {
      $("cancel").disabled = true;
      setStatus("stopping…");
      try {
        await window.filament.command("cancel", {});
      } catch (error) {
        toast(error.message);
      }
    });
    $("theme-toggle").addEventListener("click", () => {
      applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
    });
    $("mode-toggle").addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-mode]");
      if (!button || S.busy) return;
      const mode = button.dataset.mode;
      try {
        await window.filament.command("set_mode", { mode });
        S.mode = mode;
        document.querySelectorAll("#mode-toggle button").forEach((item) => {
          item.classList.toggle("active", item.dataset.mode === mode);
        });
        toast(mode === "plan" ? "Plan mode: read-only" : "Build mode");
      } catch (error) {
        toast(error.message);
      }
    });
    transcript.addEventListener("click", (event) => {
      const chip = event.target.closest(".chip");
      if (!chip) return;
      messageInput.value = chip.dataset.prompt || "";
      resizeComposer();
      sendMessage(messageInput.value);
    });
  }

  async function init() {
    const saved = (() => {
      try {
        return localStorage.getItem("filament-theme");
      } catch (error) {
        return null;
      }
    })();
    applyTheme(saved || "dark");
    bind();
    window.filament.onEvent(handleEvent);
    resizeComposer();
    try {
      const state = await window.filament.command("get_state", {});
      applyWorkspace(state.workspace);
      applyState(state.state);
      applyProvider(state.state.provider);
      if (Number(state.state.messages) > 0) {
        const history = await window.filament.command("get_messages", {});
        renderHistory(history.messages || []);
      }
      if (state.state.pending_question) renderQuestion(state.state.pending_question);
      await refreshSessions();
      messageInput.focus();
    } catch (error) {
      addSystem(`Could not reach the backend: ${error.message}`, "error");
      setStatus("offline");
    }
  }

  init();
})();
