"use strict";

const $ = (id) => document.getElementById(id);
const elements = {
  form: $("promptForm"), prompt: $("prompt"), messages: $("messages"), error: $("errorBox"),
  send: $("sendButton"), stop: $("stopButton"), health: $("healthButton"), healthText: $("healthText"),
  usage: $("usage"), baseUrl: $("baseUrl"), apiKey: $("apiKey"), stream: $("stream"),
  maxTokens: $("maxTokens"), maxTokensValue: $("maxTokensValue"), temperature: $("temperature"),
  temperatureValue: $("temperatureValue"), topK: $("topK"), topP: $("topP"),
  repetitionPenalty: $("repetitionPenalty"), seed: $("seed"), reset: $("resetButton")
};
let activeSocket = null;
let activeController = null;

function baseUrl() { return elements.baseUrl.value.trim().replace(/\/$/, "") || window.location.origin; }
function requestPayload() {
  const seed = elements.seed.value.trim();
  return {
    prompt: elements.prompt.value.trim(), max_tokens: Number(elements.maxTokens.value),
    temperature: Number(elements.temperature.value), top_k: Number(elements.topK.value),
    top_p: Number(elements.topP.value), repetition_penalty: Number(elements.repetitionPenalty.value),
    seed: seed === "" ? null : Number(seed), stop: []
  };
}
function showError(message = "") { elements.error.textContent = message; elements.error.hidden = !message; }
function setBusy(busy) {
  elements.send.disabled = busy; elements.stop.hidden = !busy; elements.prompt.disabled = busy;
  elements.send.textContent = busy ? "Generating…" : "Generate";
}
function addMessage(role, text = "") {
  const article = document.createElement("article"); article.className = `message ${role}`;
  const avatar = document.createElement("div"); avatar.className = "avatar"; avatar.textContent = role === "user" ? "Y" : "G";
  const body = document.createElement("div"); const name = document.createElement("strong");
  name.textContent = role === "user" ? "You" : "Gopi"; const paragraph = document.createElement("p");
  paragraph.textContent = text; body.append(name, paragraph); article.append(avatar, body);
  elements.messages.append(article); elements.messages.scrollTop = elements.messages.scrollHeight; return paragraph;
}
function updateUsage(usage, reason) {
  if (!usage) return;
  elements.usage.textContent = `${usage.prompt_tokens} prompt + ${usage.completion_tokens} generated tokens · ${reason}`;
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
  const result = await response.json(); target.textContent = result.text; updateUsage(result.usage, result.finish_reason);
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
      if (event.type === "token") { target.textContent += event.token; elements.messages.scrollTop = elements.messages.scrollHeight; }
      if (event.type === "done") { updateUsage(event.usage, event.finish_reason); socket.close(1000); resolve(); }
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
elements.form.addEventListener("submit", async (event) => {
  event.preventDefault(); const prompt = elements.prompt.value.trim(); if (!prompt) return;
  showError(); addMessage("user", prompt); const target = addMessage("assistant", ""); elements.prompt.value = ""; setBusy(true);
  try { const payload = requestPayload(); payload.prompt = prompt; elements.stream.checked ? await generateStream(payload, target) : await generateRest(payload, target); }
  catch (error) { if (error.name !== "AbortError") showError(error.message); if (!target.textContent) target.textContent = "Generation stopped."; }
  finally { activeController = null; activeSocket = null; setBusy(false); elements.prompt.focus(); }
});
elements.stop.addEventListener("click", () => { activeController?.abort(); activeSocket?.close(1000); setBusy(false); });
elements.health.addEventListener("click", checkHealth);
elements.maxTokens.addEventListener("input", () => { elements.maxTokensValue.value = elements.maxTokens.value; });
elements.temperature.addEventListener("input", () => { elements.temperatureValue.value = elements.temperature.value; });
elements.reset.addEventListener("click", () => {
  elements.maxTokens.value = "80"; elements.temperature.value = "0.8"; elements.topK.value = "40";
  elements.topP.value = "1"; elements.repetitionPenalty.value = "1"; elements.seed.value = "";
  elements.maxTokensValue.value = "80"; elements.temperatureValue.value = "0.8";
});
checkHealth();
