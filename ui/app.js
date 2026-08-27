"use strict";

const $ = (id) => document.getElementById(id);
const elements = {
  form: $("promptForm"), prompt: $("prompt"), messages: $("messages"), error: $("errorBox"),
  send: $("sendButton"), stop: $("stopButton"), clear: $("clearButton"),
  health: $("healthButton"), healthText: $("healthText"), usage: $("usage"),
  baseUrl: $("baseUrl"), apiKey: $("apiKey"), stream: $("stream"),
  maxTokens: $("maxTokens"), maxTokensValue: $("maxTokensValue"), temperature: $("temperature"),
  temperatureValue: $("temperatureValue"), topK: $("topK"), topP: $("topP"),
  repetitionPenalty: $("repetitionPenalty"), seed: $("seed"), reset: $("resetButton"),
  responseFormat: $("responseFormat"), webSearch: $("webSearch"),
  chatMode: $("chatMode"), modeDescription: $("modeDescription"),
  calculatorTool: $("calculatorTool"), datetimeTool: $("datetimeTool"),
  searchTool: $("searchTool"), toolCount: $("toolCount")
};
const settingsKey = "gopi-playground-settings";
const transcriptKey = "gopi-playground-transcript";
let activeSocket = null;
let activeController = null;
let generationInProgress = false;
let sessionId = sessionStorage.getItem("gopi-session-id") || `chat-${crypto.randomUUID()}`;
sessionStorage.setItem("gopi-session-id", sessionId);
const modeDescriptions = {
  balanced: "Natural answers for everyday questions",
  creative: "Imaginative ideas and expressive writing",
  precise: "Concise, careful, fact-focused answers",
  coding: "Practical code and technical explanations"
};

function baseUrl() { return elements.baseUrl.value.trim().replace(/\/$/, "") || window.location.origin; }
function requestPayload() {
  const seed = elements.seed.value.trim();
  return {
    prompt: elements.prompt.value.trim(), max_tokens: Number(elements.maxTokens.value),
    temperature: Number(elements.temperature.value), top_k: Number(elements.topK.value),
    top_p: Number(elements.topP.value), repetition_penalty: Number(elements.repetitionPenalty.value),
    seed: seed === "" ? null : Number(seed), stop: [], session_id: sessionId,
    mode: elements.chatMode.value,
    tools: [elements.calculatorTool.checked && "calculator", elements.datetimeTool.checked && "datetime"].filter(Boolean),
    response_format: elements.responseFormat.value, web_search: elements.webSearch.checked || elements.searchTool.checked
  };
}
function showError(message = "") { elements.error.textContent = message; elements.error.hidden = !message; }
function setBusy(busy) {
  elements.send.disabled = busy; elements.stop.hidden = !busy; elements.prompt.disabled = busy;
  elements.clear.disabled = busy; elements.send.textContent = busy ? "Generating…" : "Generate";
}
function appendInline(parent, text) {
  const pattern = /(\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))/g;
  let offset = 0;
  for (const match of text.matchAll(pattern)) {
    parent.append(document.createTextNode(text.slice(offset, match.index)));
    let node;
    if (match[2]) { node = document.createElement("strong"); node.textContent = match[2]; }
    else if (match[3]) { node = document.createElement("code"); node.textContent = match[3]; }
    else {
      node = document.createElement("a"); node.textContent = match[4]; node.href = match[5];
      node.target = "_blank"; node.rel = "noopener noreferrer";
    }
    parent.append(node); offset = match.index + match[0].length;
  }
  parent.append(document.createTextNode(text.slice(offset)));
}
function renderMarkdown(container, markdown) {
  container.replaceChildren();
  let code = null;
  for (const line of markdown.split("\n")) {
    if (line.startsWith("```")) {
      if (code) code = null;
      else { const pre = document.createElement("pre"); code = document.createElement("code"); pre.append(code); container.append(pre); }
      continue;
    }
    if (code) { code.textContent += `${line}\n`; continue; }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const bullet = line.match(/^[-*]\s+(.+)$/);
    const ordered = line.match(/^\d+[.)]\s+(.+)$/);
    const node = document.createElement(heading ? `h${heading[1].length + 2}` : bullet || ordered ? "div" : "p");
    if (bullet || ordered) node.className = "markdown-list-item";
    appendInline(node, heading ? heading[2] : bullet ? `• ${bullet[1]}` : line);
    container.append(node);
  }
}
function setMessage(target, text, format = "plain") {
  target.dataset.rawText = text;
  target.dataset.format = format;
  if (format === "markdown") renderMarkdown(target, text);
  else { target.replaceChildren(); target.textContent = text; }
}
function addMessage(role, text = "", format = "plain") {
  const article = document.createElement("article"); article.className = `message ${role}`; article.dataset.role = role;
  const avatar = document.createElement("div"); avatar.className = "avatar"; avatar.textContent = role === "user" ? "Y" : "G";
  const body = document.createElement("div"); body.className = "message-body";
  const header = document.createElement("div"); header.className = "message-header";
  const name = document.createElement("strong"); name.textContent = role === "user" ? "You" : "Gopi";
  const copy = document.createElement("button"); copy.type = "button"; copy.className = "copy-button"; copy.textContent = "Copy";
  const content = document.createElement("div"); content.className = "message-content";
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(content.dataset.rawText || ""); copy.textContent = "Copied"; }
    catch { copy.textContent = "Unavailable"; }
    setTimeout(() => { copy.textContent = "Copy"; }, 1200);
  });
  header.append(name, copy); body.append(header, content); article.append(avatar, body);
  setMessage(content, text, format); elements.messages.append(article);
  elements.messages.scrollTop = elements.messages.scrollHeight; return content;
}
function saveTranscript() {
  const messages = [...elements.messages.querySelectorAll(".message[data-role]")].map((article) => ({
    role: article.dataset.role, text: article.querySelector(".message-content")?.dataset.rawText || "",
    format: article.querySelector(".message-content")?.dataset.format || "plain"
  })).filter((message) => message.text);
  sessionStorage.setItem(transcriptKey, JSON.stringify(messages.slice(-50)));
}
function restoreTranscript() {
  let transcript = [];
  try { transcript = JSON.parse(sessionStorage.getItem(transcriptKey) || "[]"); } catch { transcript = []; }
  if (!Array.isArray(transcript) || !transcript.length) return false;
  elements.messages.replaceChildren(); transcript.forEach((message) => addMessage(message.role, message.text, message.format));
  return true;
}
function updateUsage(usage, reason) {
  if (usage) elements.usage.textContent = `${usage.prompt_tokens} prompt + ${usage.completion_tokens} generated tokens · ${reason}`;
}
function headers() {
  const result = { "Content-Type": "application/json" };
  if (elements.apiKey.value) result.Authorization = `Bearer ${elements.apiKey.value}`;
  return result;
}
async function readError(response) {
  try { const body = await response.json(); return body.error?.message || `Request failed (${response.status})`; }
  catch { return `Request failed (${response.status})`; }
}
async function generateRest(payload, target) {
  activeController = new AbortController();
  const response = await fetch(`${baseUrl()}/v1/generate`, {
    method: "POST", headers: headers(), body: JSON.stringify(payload), signal: activeController.signal
  });
  if (!response.ok) throw new Error(await readError(response));
  const result = await response.json(); setMessage(target, result.text, payload.response_format); updateUsage(result.usage, result.finish_reason);
}
function websocketUrl() {
  const url = new URL(baseUrl()); url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `${url.pathname.replace(/\/$/, "")}/v1/generate/stream`; return url.toString();
}
function generateStream(payload, target) {
  return new Promise((resolve, reject) => {
    if (elements.apiKey.value) return reject(new Error("Browser WebSockets cannot send the API-key header. Disable streaming for authenticated testing."));
    const socket = new WebSocket(websocketUrl()); activeSocket = socket;
    socket.addEventListener("open", () => socket.send(JSON.stringify(payload)));
    socket.addEventListener("message", ({ data }) => {
      let event; try { event = JSON.parse(data); } catch { reject(new Error("Server returned invalid stream data.")); socket.close(); return; }
      if (event.type === "token") { setMessage(target, (target.dataset.rawText || "") + event.token); elements.messages.scrollTop = elements.messages.scrollHeight; }
      if (event.type === "done") { setMessage(target, target.dataset.rawText || "", payload.response_format); updateUsage(event.usage, event.finish_reason); socket.close(1000); resolve(); }
      if (event.type === "error") { reject(new Error(event.error?.message || "Streaming failed.")); socket.close(); }
    });
    socket.addEventListener("error", () => reject(new Error("WebSocket connection failed.")));
    socket.addEventListener("close", (event) => { activeSocket = null; if (event.code !== 1000) reject(new Error("Streaming connection closed.")); });
  });
}
async function checkHealth() {
  elements.health.className = "status"; elements.healthText.textContent = "Checking…";
  try {
    const response = await fetch(`${baseUrl()}/health/ready`); const result = await response.json();
    elements.health.classList.add(result.ready ? "ready" : "failed"); elements.healthText.textContent = result.ready ? `${result.model} ready` : "Model not ready";
  } catch { elements.health.classList.add("failed"); elements.healthText.textContent = "Server offline"; }
}
function saveSettings() {
  localStorage.setItem(settingsKey, JSON.stringify({
    baseUrl: elements.baseUrl.value, stream: elements.stream.checked, maxTokens: elements.maxTokens.value,
    temperature: elements.temperature.value, topK: elements.topK.value, topP: elements.topP.value,
    repetitionPenalty: elements.repetitionPenalty.value, seed: elements.seed.value,
    responseFormat: elements.responseFormat.value, webSearch: elements.webSearch.checked,
    chatMode: elements.chatMode.value, calculatorTool: elements.calculatorTool.checked,
    datetimeTool: elements.datetimeTool.checked, searchTool: elements.searchTool.checked
  }));
}
function loadSettings() {
  let saved = {}; try { saved = JSON.parse(localStorage.getItem(settingsKey) || "{}"); } catch { saved = {}; }
  for (const key of ["baseUrl", "maxTokens", "temperature", "topK", "topP", "repetitionPenalty", "seed", "responseFormat", "chatMode"]) {
    if (saved[key] !== undefined) elements[key].value = saved[key];
  }
  if (saved.stream !== undefined) elements.stream.checked = saved.stream;
  if (saved.webSearch !== undefined) elements.webSearch.checked = saved.webSearch;
  for (const key of ["calculatorTool", "datetimeTool", "searchTool"]) {
    if (saved[key] !== undefined) elements[key].checked = saved[key];
  }
  elements.maxTokensValue.value = elements.maxTokens.value; elements.temperatureValue.value = elements.temperature.value;
  elements.modeDescription.textContent = modeDescriptions[elements.chatMode.value];
  updateToolCount();
}
function updateToolCount() {
  const count = [elements.calculatorTool, elements.datetimeTool, elements.searchTool].filter((tool) => tool.checked).length;
  elements.toolCount.textContent = count; elements.toolCount.hidden = count === 0;
}
elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (generationInProgress) return;
  const prompt = elements.prompt.value.trim(); if (!prompt) return;
  generationInProgress = true;
  const payload = requestPayload(); payload.prompt = prompt; showError(); addMessage("user", prompt);
  const target = addMessage("assistant", ""); elements.prompt.value = ""; elements.prompt.style.height = ""; setBusy(true);
  try { elements.stream.checked ? await generateStream(payload, target) : await generateRest(payload, target); }
  catch (error) { if (error.name !== "AbortError") showError(error.message); if (!target.dataset.rawText) setMessage(target, "Generation stopped."); }
  finally { generationInProgress = false; activeController = null; activeSocket = null; setBusy(false); saveTranscript(); elements.prompt.focus(); }
});
elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!event.repeat && !generationInProgress) elements.form.requestSubmit();
  }
});
elements.prompt.addEventListener("input", () => {
  elements.prompt.style.height = "auto"; elements.prompt.style.height = `${Math.min(elements.prompt.scrollHeight, 220)}px`;
});
elements.stop.addEventListener("click", () => { activeController?.abort(); activeSocket?.close(1000); setBusy(false); });
elements.clear.addEventListener("click", () => {
  sessionId = `chat-${crypto.randomUUID()}`; sessionStorage.setItem("gopi-session-id", sessionId);
  sessionStorage.removeItem(transcriptKey); elements.messages.replaceChildren();
  addMessage("assistant", "Hello! I’m your local AI assistant. What can I help you with?");
  elements.usage.textContent = "New conversation"; showError(); elements.prompt.focus();
});
elements.health.addEventListener("click", checkHealth);
[elements.calculatorTool, elements.datetimeTool, elements.searchTool].forEach((tool) => {
  tool.addEventListener("change", () => { updateToolCount(); saveSettings(); });
});
elements.chatMode.addEventListener("change", () => {
  elements.modeDescription.textContent = modeDescriptions[elements.chatMode.value];
  sessionId = `chat-${crypto.randomUUID()}`;
  sessionStorage.setItem("gopi-session-id", sessionId);
  sessionStorage.removeItem(transcriptKey);
  elements.messages.replaceChildren();
  addMessage("assistant", `${elements.chatMode.options[elements.chatMode.selectedIndex].text.replace(/^[^ ]+ /, "")} mode is ready. How can I help?`);
  elements.usage.textContent = "New mode · new conversation";
  saveSettings();
});
elements.maxTokens.addEventListener("input", () => { elements.maxTokensValue.value = elements.maxTokens.value; });
elements.temperature.addEventListener("input", () => { elements.temperatureValue.value = elements.temperature.value; });
document.querySelector(".controls").addEventListener("change", saveSettings);
elements.reset.addEventListener("click", () => {
  elements.maxTokens.value = "128"; elements.temperature.value = "0.7"; elements.topK.value = "40";
  elements.topP.value = "0.9"; elements.repetitionPenalty.value = "1.2"; elements.seed.value = "";
  elements.responseFormat.value = "plain"; elements.webSearch.checked = false;
  elements.chatMode.value = "balanced"; elements.modeDescription.textContent = modeDescriptions.balanced;
  elements.calculatorTool.checked = false; elements.datetimeTool.checked = false; elements.searchTool.checked = false; updateToolCount();
  elements.maxTokensValue.value = "128"; elements.temperatureValue.value = "0.7"; saveSettings();
});
loadSettings();
if (!restoreTranscript()) addMessage("assistant", "Hello! I’m your local AI assistant. What can I help you with?");
checkHealth();
