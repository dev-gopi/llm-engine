"use strict";

const $ = (id) => document.getElementById(id);
const el = {
  form: $("promptForm"), prompt: $("prompt"), messages: $("messages"), error: $("errorBox"),
  send: $("sendButton"), stop: $("stopButton"), clear: $("clearButton"), newTop: $("newChatTop"),
  health: $("healthButton"), healthText: $("healthText"), usage: $("usage"),
  baseUrl: $("baseUrl"), apiKey: $("apiKey"), adminApiKey: $("adminApiKey"), apiMode: $("apiMode"), stream: $("stream"), useMemory: $("useMemory"),
  modelSummary: $("modelSummary"), responseFormat: $("responseFormat"), jsonSchemaWrap: $("jsonSchemaWrap"), jsonSchema: $("jsonSchema"),
  maxTokens: $("maxTokens"), maxTokensValue: $("maxTokensValue"), temperature: $("temperature"), temperatureValue: $("temperatureValue"),
  topK: $("topK"), topP: $("topP"), minTokens: $("minTokens"), minP: $("minP"), repetitionPenalty: $("repetitionPenalty"),
  noRepeatNgram: $("noRepeatNgram"), seed: $("seed"), stopStrings: $("stopStrings"),
  chatMode: $("chatMode"), modeDescription: $("modeDescription"), reasoningEffort: $("reasoningEffort"),
  calculatorTool: $("calculatorTool"), datetimeTool: $("datetimeTool"), searchTool: $("searchTool"), ragTool: $("ragTool"), mcpTool: $("mcpTool"), mcpServer: $("mcpServer"),
  conversations: $("conversationList"), historyStatus: $("historyStatus"), refreshHistory: $("refreshHistory"),
  conversationTitle: $("conversationTitle"), transportBadge: $("transportBadge"), attachments: $("attachments"), attachmentPreview: $("attachmentPreview"),
  refreshModel: $("refreshModel"), contextInfo: $("contextInfo"), compactContext: $("compactContext"), metricsButton: $("metricsButton"), auditButton: $("auditButton"), diagnosticOutput: $("diagnosticOutput"),
  embeddingInput: $("embeddingInput"), embeddingButton: $("embeddingButton"), workspaceAction: $("workspaceAction"), workspaceButton: $("workspaceButton"),
  lifecycleStatus: $("lifecycleStatus"), lifecycleLoad: $("lifecycleLoad"), lifecycleReload: $("lifecycleReload"), lifecycleUnload: $("lifecycleUnload"),
  reviewLast: $("reviewLast"), reviewPrompt: $("reviewPrompt"), reviewAnswer: $("reviewAnswer"), approveExample: $("approveExample"), approveTraining: $("approveTraining"), exportExample: $("exportExample"), deleteTraining: $("deleteTraining"), reviewStatus: $("reviewStatus"),
  reset: $("resetButton")
};

const SETTINGS_KEY = "gopi-playground-settings-v2";
const HISTORY_KEY = "gopi-playground-conversations-v2";
const ACTIVE_KEY = "gopi-playground-active-session";
const GREETING = "Hello! I’m your local AI assistant. What can I help you with?";
const MAX_LOCAL_CONVERSATIONS = 12;
const MAX_LOCAL_MESSAGES = 50;
const MAX_LOCAL_MESSAGE_CHARS = 16000;
const MAX_LOCAL_STORAGE_BYTES = 700000;
const DEFAULT_SCHEMA = {name:"response",schema:{type:"object",properties:{answer:{type:"string"}},required:["answer"],additionalProperties:false},strict:true};

const modeDescriptions = {
  balanced: "Natural answers for everyday questions",
  creative: "Imaginative ideas and expressive writing",
  precise: "Concise, careful, fact-focused answers",
  coding: "Practical code and technical explanations"
};

let activeSocket = null;
let activeController = null;
let activeRequestId = null;
let generationInProgress = false;
let lastCompletedExample = null;
let serverHistoryAvailable = false;
let authenticationRequired = false;
let serverSessions = new Map();
let sessionId = sessionStorage.getItem(ACTIVE_KEY) || `chat-${crypto.randomUUID()}`;
sessionStorage.setItem(ACTIVE_KEY, sessionId);

const text = (v) => String(v ?? "").trim();
function apiBase() { const value=el.baseUrl.value.trim().replace(/\/$/, "") || window.location.origin; const url=new URL(value); if(!["http:","https:"].includes(url.protocol)) throw new Error("API base URL must use http:// or https://."); return url.toString().replace(/\/$/,""); }
function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (el.apiKey.value.trim()) headers.Authorization = `Bearer ${el.apiKey.value.trim()}`;
  return headers;
}
function adminHeaders(extra = {}) {
  const headers = authHeaders(extra);
  if (el.adminApiKey.value.trim()) headers["X-Admin-API-Key"] = el.adminApiKey.value.trim();
  return headers;
}
function jsonHeaders() { return authHeaders({"Content-Type":"application/json"}); }

async function responseMessage(response) {
  try {
    const body = await response.json();
    return body?.error?.message || body?.detail || (Array.isArray(body?.detail) ? body.detail.map((x) => x.msg).join("; ") : "") || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}
async function request(path, options = {}) {
  const response = await fetch(`${apiBase()}${path}`, { ...options, headers: options.headers || jsonHeaders() });
  if (!response.ok) throw new Error(await responseMessage(response));
  return response;
}
function showError(message = "") { el.error.textContent = message; el.error.hidden = !message; }
function showDiagnostic(value) { el.diagnosticOutput.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2); }
function setBusy(busy) {
  generationInProgress = busy;
  el.send.disabled = busy;
  el.stop.hidden = !busy;
  el.prompt.disabled = busy;
  el.clear.disabled = busy;
  el.newTop.disabled = busy;
  el.send.textContent = busy ? "Generating…" : "Generate";
  el.reviewLast.disabled = busy || !lastCompletedExample;
}
function setTransport(label) { el.transportBadge.textContent = label; }

function sanitizeMessage(message) {
  const role = ["user","assistant","system","tool"].includes(message?.role) ? message.role : "assistant";
  let content = message?.content;
  if (Array.isArray(content)) {
    content = content.map((part) => part?.type === "text" ? part.text : part?.type === "image_url" ? "[image]" : "").filter(Boolean).join("\n");
  }
  return { role, content: text(content).slice(0, MAX_LOCAL_MESSAGE_CHARS) };
}
function currentMessages() {
  return [...el.messages.querySelectorAll(".message[data-role]")].map((article) => ({
    role: article.dataset.role,
    content: (article.querySelector(".message-content")?.dataset.rawText || "").slice(0, MAX_LOCAL_MESSAGE_CHARS)
  })).filter((m) => m.content);
}
function localConversations() {
  try {
    const rows = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(rows) ? rows.filter((x) => x && typeof x.sessionId === "string") : [];
  } catch { return []; }
}
function localTitle(messages) {
  const first = messages.find((m) => m.role === "user" && m.content);
  return (first?.content || "New conversation").replace(/\s+/g, " ").slice(0, 72);
}
function persistLocalConversation() {
  const messages = currentMessages().slice(-MAX_LOCAL_MESSAGES).map(sanitizeMessage);
  const record = {
    sessionId,
    title: localTitle(messages),
    updated: Date.now(),
    mode: el.chatMode.value,
    messages
  };
  let all = localConversations().filter((x) => x.sessionId !== sessionId);
  all.unshift(record);
  all = all.slice(0, MAX_LOCAL_CONVERSATIONS);
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(all));
  } catch {
    const compact = all.slice(0, 6).map((x) => ({...x, messages:x.messages.slice(-20).map((m)=>({...m,content:m.content.slice(-8000)}))}));
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(compact)); } catch { /* browser storage may be disabled */ }
  }
  renderHistory();
}
function deleteLocalConversation(id) {
  const remaining = localConversations().filter((x) => x.sessionId !== id);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(remaining));
}
function renderHistory() {
  const local = localConversations();
  const merged = new Map(local.map((x) => [x.sessionId, {...x, source:"local"}]));
  for (const item of serverSessions.values()) {
    const previous = merged.get(item.session_id);
    merged.set(item.session_id, {...item, source:"server", title: previous?.title || item.title});
  }
  const rows = [...merged.values()].sort((a,b) => Number(b.updated||0)-Number(a.updated||0));
  el.conversations.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("p"); empty.className = "empty-state"; empty.textContent = "No saved conversations yet."; el.conversations.append(empty); return;
  }
  for (const row of rows) {
    const item = document.createElement("div"); item.className = "conversation-item" + (row.sessionId === sessionId || row.session_id === sessionId ? " active" : "");
    const info = document.createElement("button"); info.type = "button"; info.className = "conversation-open";
    const title = document.createElement("strong"); title.textContent = row.title || "Conversation";
    const meta = document.createElement("small"); meta.textContent = `${row.source === "server" ? "Server" : "Local"} · ${row.message_count ?? row.messages?.length ?? 0} messages · ${formatDate(row.updated)}`;
    info.append(title, meta); info.addEventListener("click", () => openConversation(row.sessionId || row.session_id));
    const del = document.createElement("button"); del.type = "button"; del.className = "danger-button conversation-delete"; del.textContent = "Delete";
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteConversation(row.sessionId || row.session_id); });
    item.append(info, del); el.conversations.append(item);
  }
}
function formatDate(value) {
  const date = new Date(Number(value) * (Number(value) < 100000000000 ? 1000 : 1));
  if (Number.isNaN(date.valueOf())) return "unknown time";
  return new Intl.DateTimeFormat(undefined, {month:"short", day:"numeric", hour:"2-digit", minute:"2-digit"}).format(date);
}

function setConversationTitle() {
  const messages = currentMessages();
  el.conversationTitle.textContent = localTitle(messages);
  el.modeDescription.textContent = modeDescriptions[el.chatMode.value] || "";
}
function renderMarkdown(container, markdown) {
  container.replaceChildren();
  let code = null;
  for (const line of markdown.split("\n")) {
    if (line.startsWith("```")) { code ? code = null : (() => { const pre=document.createElement("pre"); code=document.createElement("code"); pre.append(code); container.append(pre); })(); continue; }
    if (code) { code.textContent += line + "\n"; continue; }
    const match = line.match(/^(#{1,3})\s+(.+)$/); const bullet = line.match(/^[-*]\s+(.+)$/); const ordered = line.match(/^\d+[.)]\s+(.+)$/);
    const node = document.createElement(match ? `h${match[1].length+2}` : (bullet || ordered ? "div" : "p"));
    if (bullet || ordered) node.className = "markdown-list-item";
    appendInline(node, match ? match[2] : bullet ? `• ${bullet[1]}` : line);
    container.append(node);
  }
}
function appendInline(parent, value) {
  const pattern = /(\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))/g;
  let offset = 0;
  for (const match of value.matchAll(pattern)) {
    parent.append(document.createTextNode(value.slice(offset, match.index)));
    if (match[2]) { const n=document.createElement("strong"); n.textContent=match[2]; parent.append(n); }
    else if (match[3]) { const n=document.createElement("code"); n.textContent=match[3]; parent.append(n); }
    else { const n=document.createElement("a"); n.textContent=match[4]; n.href=match[5]; n.target="_blank"; n.rel="noopener noreferrer"; parent.append(n); }
    offset = match.index + match[0].length;
  }
  parent.append(document.createTextNode(value.slice(offset)));
}
function setMessageContent(target, value, format = "plain") {
  const safe = String(value ?? "");
  target.dataset.rawText = safe; target.dataset.format = format || "plain";
  if (format === "markdown") renderMarkdown(target, safe); else { target.replaceChildren(); target.textContent = safe; }
}
function addMessage(role, value = "", format = "plain", meta = "") {
  const article = document.createElement("article"); article.className=`message ${role}`; article.dataset.role=role;
  const avatar=document.createElement("div"); avatar.className="avatar"; avatar.textContent=role === "user" ? "Y" : "G";
  const body=document.createElement("div"); body.className="message-body";
  const header=document.createElement("div"); header.className="message-header";
  const name=document.createElement("strong"); name.textContent=role === "user" ? "You" : role === "tool" ? "Tool" : "Gopi";
  const copy=document.createElement("button"); copy.type="button"; copy.className="copy-button"; copy.textContent="Copy";
  const content=document.createElement("div"); content.className="message-content";
  const metaNode=document.createElement("small"); metaNode.className="message-meta"; metaNode.textContent=meta;
  copy.addEventListener("click", async () => { try { await navigator.clipboard.writeText(content.dataset.rawText || ""); copy.textContent="Copied"; } catch { copy.textContent="Unavailable"; } setTimeout(()=>copy.textContent="Copy",1000); });
  header.append(name,copy); body.append(header,content); if(meta) body.append(metaNode); article.append(avatar,body); setMessageContent(content,value,format); el.messages.append(article); el.messages.scrollTop=el.messages.scrollHeight; return content;
}
function restoreMessages(messages) {
  el.messages.replaceChildren();
  for (const message of messages || []) addMessage(message.role, message.content, "plain");
  if (!messages?.length) addMessage("assistant", GREETING);
  setConversationTitle();
}
function newConversation({persist=true}={}) {
  if (persist && currentMessages().some((m) => m.role === "user")) persistLocalConversation();
  sessionId = `chat-${crypto.randomUUID()}`; sessionStorage.setItem(ACTIVE_KEY,sessionId);
  lastCompletedExample = null; el.reviewLast.disabled = true; el.messages.replaceChildren(); addMessage("assistant",GREETING); el.usage.textContent="New conversation"; showError(); setConversationTitle(); renderHistory();
}

async function openConversation(id) {
  if (generationInProgress) return;
  const local = localConversations().find((x) => x.sessionId === id);
  try {
    if (serverSessions.has(id) && el.apiKey.value.trim() && el.useMemory.checked) {
      const response = await request(`/v1/sessions/${encodeURIComponent(id)}/memory`, {headers: authHeaders()});
      const body = await response.json(); sessionId=id; sessionStorage.setItem(ACTIVE_KEY,id); restoreMessages(body.messages); el.historyStatus.textContent="Server history connected."; return;
    }
  } catch (error) { el.historyStatus.textContent=`Server history unavailable: ${error.message}`; }
  if (local) { sessionId=id; sessionStorage.setItem(ACTIVE_KEY,id); restoreMessages(local.messages); el.chatMode.value=local.mode || el.chatMode.value; setConversationTitle(); return; }
  showError("That conversation is no longer available.");
}
async function deleteConversation(id) {
  if (!confirm("Delete this conversation and its stored training examples?")) return;
  if (el.apiKey.value.trim() && (serverSessions.has(id) || id === sessionId)) {
    try {
      await request(`/v1/sessions/${encodeURIComponent(id)}/memory`, {method:"DELETE", headers:authHeaders()});
    } catch (error) {
      const message=error.message.toLowerCase();
      if (!message.includes("not found") && !message.includes("session memory")) { showError(`Server delete failed: ${error.message}`); return; }
    }
  }
  deleteLocalConversation(id); serverSessions.delete(id);
  if (id === sessionId) newConversation({persist:false}); else renderHistory();
}

async function refreshServerHistory() {
  serverSessions = new Map();
  if (!el.apiKey.value.trim()) { serverHistoryAvailable=false; el.historyStatus.textContent="Local history is available. Add an API key for server history."; renderHistory(); return; }
  try {
    const response=await request("/v1/sessions?limit=100",{headers:authHeaders()}); const body=await response.json();
    for(const item of body.sessions || []) serverSessions.set(item.session_id,item);
    serverHistoryAvailable=true; el.historyStatus.textContent=`Server history: ${serverSessions.size} conversation${serverSessions.size===1?"":"s"}.`;
  } catch(error) { serverHistoryAvailable=false; el.historyStatus.textContent=`Local history · server history unavailable (${error.message})`; }
  renderHistory();
}

function validateUi() {
  const p=text(el.prompt.value); if(!p && !el.attachments.files.length) return "Add a message or a media attachment.";
  if(authenticationRequired && !el.apiKey.value.trim()) return "This server requires an API key. Enter GOPI_API_KEY in Connection settings, then try again.";
  if(p.length>262144) return "Message is too long. Maximum length is 262,144 characters.";
  const nums=[
    ["maximum tokens",Number(el.maxTokens.value),1,8192], ["temperature",Number(el.temperature.value),0,2], ["top K",Number(el.topK.value),0,100000],
    ["top P",Number(el.topP.value),0.000001,1], ["minimum tokens",Number(el.minTokens.value),0,1000000], ["min P",Number(el.minP.value),0,1], ["repetition penalty",Number(el.repetitionPenalty.value),0.1,2], ["n-gram size",Number(el.noRepeatNgram.value),0,16]
  ];
  for(const [name,val,min,max] of nums) if(!Number.isFinite(val)||val<min||val>max) return `${name} must be between ${min} and ${max}.`;
  if(el.stopStrings.value.split("\n").filter(Boolean).length>16) return "Use at most 16 stop strings.";
  if(el.mcpTool.checked && !/^[A-Za-z0-9._-]{1,128}$/.test(text(el.mcpServer.value))) return "MCP server name must contain only letters, numbers, dot, underscore, or hyphen.";
  return "";
}
const SUPPORTED_MEDIA_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac", "video/mp4", "video/webm"]);
async function readAttachments() {
  const files=[...el.attachments.files]; if(files.length>8) throw new Error("Attach at most 8 files.");
  let images=0;
  return Promise.all(files.map(async file=>{
    if(file.type.startsWith("image/")) {
      if(!SUPPORTED_MEDIA_TYPES.has(file.type)) throw new Error(`${file.name} has an unsupported image type.`);
      if(++images>4) throw new Error("Attach at most 4 images.");
      if(file.size>10*1024*1024) throw new Error(`${file.name} exceeds the 10MB image limit.`);
      return {kind:"image",name:file.name,media_type:file.type,data_url:await fileToDataUrl(file)};
    }
    if(file.type.startsWith("audio/") || file.type.startsWith("video/")) {
      if(!SUPPORTED_MEDIA_TYPES.has(file.type)) throw new Error(`${file.name} has an unsupported media type.`);
      return {kind:file.type.startsWith("audio/")?"audio":"video",name:file.name,media_type:file.type,file};
    }
    if(file.size>262144) throw new Error(`${file.name} exceeds the 256KB text attachment limit.`);
    return {kind:"text",name:file.name,media_type:file.type || "text/plain",content:await file.text()};
  }));
}
function fileToDataUrl(file) { return new Promise((resolve,reject)=>{const reader=new FileReader(); reader.onerror=()=>reject(new Error(`Could not read ${file.name}.`)); reader.onload=()=>resolve(String(reader.result)); reader.readAsDataURL(file);}); }
function renderAttachmentPreview() {
  const icon = (file) => file.type.startsWith("image/") ? "🖼" : file.type.startsWith("audio/") ? "🔊" : file.type.startsWith("video/") ? "🎬" : "📄";
  el.attachmentPreview.replaceChildren(); for(const file of [...el.attachments.files]) { const n=document.createElement("span"); n.className="file-pill"; n.textContent=`${icon(file)} ${file.name}`; el.attachmentPreview.append(n); }
}

function buildGeneratePayload(prompt, attachments) {
  const seed=text(el.seed.value); const textAttachments=attachments.filter(a=>a.kind==="text").map(a=>({name:a.name,content:a.content,media_type:a.media_type}));
  return {
    prompt,max_tokens:Number(el.maxTokens.value),temperature:Number(el.temperature.value),top_k:Number(el.topK.value),top_p:Number(el.topP.value),min_p:Number(el.minP.value),repetition_penalty:Number(el.repetitionPenalty.value),
    no_repeat_ngram_size:Number(el.noRepeatNgram.value),min_tokens:Number(el.minTokens.value),seed:seed===""?null:Number(seed),stop:el.stopStrings.value.split("\n").filter(Boolean),
    session_id:el.useMemory.checked?sessionId:null,mode:el.chatMode.value,reasoning_effort:el.reasoningEffort.value,
    tools:[el.calculatorTool.checked?"calculator":null,el.datetimeTool.checked?"datetime":null].filter(Boolean),response_format:el.responseFormat.value === "plain" || el.responseFormat.value === "markdown" ? el.responseFormat.value : "plain",
    web_search:el.searchTool.checked,rag:el.ragTool.checked,mcp:el.mcpTool.checked,mcp_server:el.mcpTool.checked?text(el.mcpServer.value)||"filesystem":null,attachments:textAttachments
  };
}
function buildOpenAITools() {
  const tools=[];
  if(el.calculatorTool.checked) tools.push({type:"function",function:{name:"calculator",description:"Evaluate a safe arithmetic expression.",parameters:{type:"object",properties:{expression:{type:"string"}},required:["expression"],additionalProperties:false}}});
  if(el.datetimeTool.checked) tools.push({type:"function",function:{name:"datetime",description:"Return current date and time information.",parameters:{type:"object",properties:{timezone:{type:"string"}},additionalProperties:false}}});
  return tools;
}
function jsonResponseFormat() {
  if(el.responseFormat.value==="json_object") return {type:"json_object"};
  if(el.responseFormat.value!=="json_schema") return null;
  let value; try { value=JSON.parse(el.jsonSchema.value); } catch { throw new Error("JSON schema must be valid JSON."); }
  if(!value || typeof value!=="object" || typeof value.name!=="string" || !value.schema || typeof value.schema!=="object") throw new Error("JSON schema must contain name and schema fields.");
  return {type:"json_schema",json_schema:value};
}
function buildChatMessages(attachments) {
  const messages=currentMessages().slice(-256).map((m)=>({role:m.role,content:m.content}));
  const parts=[{type:"text",text:text(el.prompt.value)}];
  for(const a of attachments) if(a.kind==="image") parts.push({type:"image_url",image_url:{url:a.data_url,detail:"auto"}}); else parts[0].text += `\n\n[Attached file: ${a.name}]\n${a.content}`;
  messages.push({role:"user",content:parts.length===1?parts[0].text:parts});
  return messages;
}
function buildChatPayload(messages) {
  const seed=text(el.seed.value);
  const payload={model:window.GOPI_MODEL||undefined,messages,stream:el.stream.checked,temperature:Number(el.temperature.value),top_p:Number(el.topP.value),top_k:Number(el.topK.value),min_p:Number(el.minP.value),max_tokens:Number(el.maxTokens.value),repetition_penalty:Number(el.repetitionPenalty.value),no_repeat_ngram_size:Number(el.noRepeatNgram.value),min_tokens:Number(el.minTokens.value),seed:seed===""?null:Number(seed),stop:el.stopStrings.value.split("\n").filter(Boolean),reasoning_effort:el.reasoningEffort.value,session_id:el.useMemory.checked?sessionId:null,mode:el.chatMode.value,web_search:el.searchTool.checked,rag:el.ragTool.checked,mcp:el.mcpTool.checked,mcp_server:el.mcpTool.checked?text(el.mcpServer.value)||"filesystem":null,tools:buildOpenAITools(),tool_choice:"auto"};
  const format=jsonResponseFormat(); if(format) payload.response_format=format;
  delete payload.model; return payload;
}
function responseModeFor(attachments) {
  const requested=el.apiMode.value;
  if(attachments.some(a=>["image","audio","video"].includes(a.kind))) return "responses";
  if((el.responseFormat.value==="json_object" || el.responseFormat.value==="json_schema") && requested==="generate") return "chat";
  return requested;
}

async function generateNativeRest(payload,target) {
  const response=await request("/v1/generate",{method:"POST",headers:jsonHeaders(),body:JSON.stringify(payload),signal:activeController.signal}); const result=await response.json();
  setMessageContent(target,result.text || "",payload.response_format); updateUsage(result.usage,result.finish_reason || "completed"); return result.text || "";
}
function websocketUrl() { const u=new URL(apiBase()); u.protocol=u.protocol==="https:"?"wss:":"ws:"; u.pathname=`${u.pathname.replace(/\/$/,"")}/v1/generate/stream`; return u.toString(); }
async function generateNativeStream(payload,target) {
  return new Promise((resolve,reject)=>{
    try {
      const protocols=el.apiKey.value.trim()?["bearer",el.apiKey.value.trim()]:undefined; activeSocket=new WebSocket(websocketUrl(),protocols); let buffer="", result="";
      activeSocket.onopen=()=>activeSocket.send(JSON.stringify(payload));
      activeSocket.onmessage=(event)=>{ let data; try{data=JSON.parse(event.data);}catch{return;} if(data.error){reject(new Error(data.error.message||"Streaming generation failed.")); return;} if(data.token){result+=data.token; setMessageContent(target,result,el.responseFormat.value);} if(data.usage) updateUsage(data.usage,data.finish_reason||"completed"); if(data.finish_reason!==undefined && data.type==="done"){ resolve(result); activeSocket?.close(1000); } };
      activeSocket.onerror=()=>reject(new Error("WebSocket streaming failed. Check the API URL, CORS, and authentication."));
      activeSocket.onclose=(event)=>{activeSocket=null; if(event.code!==1000 && event.code!==1001 && !result) reject(new Error("WebSocket connection closed before a response was received.")); else resolve(result);};
    } catch(error){reject(error);}
  });
}

async function streamSse(path,payload,onEvent) {
  activeController=new AbortController(); activeRequestId=null;
  const response=await request(path,{method:"POST",headers:jsonHeaders(),body:JSON.stringify(payload),signal:activeController.signal});
  if(!response.body) throw new Error("Streaming response body is unavailable.");
  const reader=response.body.getReader(); const decoder=new TextDecoder(); let buffer="";
  try {
    while(true){ const {value,done}=await reader.read(); if(done) break; buffer+=decoder.decode(value,{stream:true}); const blocks=buffer.split(/\r?\n\r?\n/); buffer=blocks.pop() || ""; for(const block of blocks){ for(const line of block.split(/\r?\n/)){ if(!line.startsWith("data:")) continue; const raw=line.slice(5).trim(); if(raw==="[DONE]") return; try{const parsed=JSON.parse(raw); activeRequestId=activeRequestId || parsed.id || parsed.response_id || parsed.response?.id || null; onEvent(parsed);}catch{/* ignore malformed provider comment */} } } }
    if(buffer.trim().startsWith("data:")){const raw=buffer.trim().slice(5).trim(); if(raw && raw!=="[DONE]") {try{onEvent(JSON.parse(raw));}catch{}}}
  } finally { reader.releaseLock(); }
}
async function generateChat(target) {
  const attachments=await readAttachments(); const messages=buildChatMessages(attachments); const payload=buildChatPayload(messages); const format=el.responseFormat.value;
  let result="", reasoning="", toolCalls=[];
  if(el.stream.checked){
    await streamSse("/v1/chat/completions",payload,(event)=>{
      const choice=event.choices?.[0]; const delta=choice?.delta || {};
      if(delta.content){result+=delta.content; setMessageContent(target,result,format);}
      if(delta.reasoning_content){reasoning+=delta.reasoning_content;}
      if(Array.isArray(delta.tool_calls)) toolCalls.push(...delta.tool_calls);
      if(choice?.finish_reason) updateUsage(null,choice.finish_reason);
    });
  } else {
    payload.stream=false; const response=await request("/v1/chat/completions",{method:"POST",headers:jsonHeaders(),body:JSON.stringify(payload),signal:activeController.signal}); const body=await response.json(); const message=body.choices?.[0]?.message || {};
    result=message.content || ""; reasoning=message.reasoning_content || ""; toolCalls=message.tool_calls || []; setMessageContent(target,result,format); updateUsage(body.usage,body.choices?.[0]?.finish_reason||"completed");
  }
  if(reasoning) addMessage("assistant",`Reasoning (server):\n${reasoning}`,"plain");
  if(toolCalls.length) addMessage("tool",JSON.stringify(toolCalls,null,2),"plain","Tool calls returned by backend");
  return result;
}
async function uploadMediaAsset(attachment) {
  const response=await request("/v1/media/assets",{method:"POST",headers:authHeaders({"Content-Type":attachment.media_type,"X-Filename":attachment.name}),body:attachment.file,signal:activeController.signal});
  const asset=await response.json();
  if(!asset.id) throw new Error(`Media upload did not return an asset ID for ${attachment.name}.`);
  return asset.id;
}
async function responsesInput(attachments) {
  const messages=currentMessages().slice(-256).map((m)=>({role:m.role==="tool"?"tool":m.role,content:m.content}));
  const parts=[{type:"input_text",text:text(el.prompt.value)}];
  for(const attachment of attachments) {
    if(attachment.kind==="text") parts[0].text += `\n\n[Attached file: ${attachment.name}]\n${attachment.content}`;
    else if(attachment.kind==="image") parts.push({type:"input_image",image_url:attachment.data_url,detail:"auto"});
    else parts.push({type:`input_${attachment.kind}`,asset_id:await uploadMediaAsset(attachment)});
  }
  messages.push({role:"user",content:parts});
  return messages;
}
async function generateResponses(target, attachments) {
  const input=await responsesInput(attachments); const seed=text(el.seed.value); const payload={model:window.GOPI_MODEL||"gopi",input,stream:el.stream.checked,max_output_tokens:Number(el.maxTokens.value),temperature:Number(el.temperature.value),top_p:Number(el.topP.value),top_k:Number(el.topK.value),min_p:Number(el.minP.value),seed:seed===""?null:Number(seed),stop:el.stopStrings.value.split("\n").filter(Boolean),reasoning_effort:el.reasoningEffort.value,session_id:el.useMemory.checked?sessionId:null,mode:el.chatMode.value,repetition_penalty:Number(el.repetitionPenalty.value),no_repeat_ngram_size:Number(el.noRepeatNgram.value),min_tokens:Number(el.minTokens.value),rag:el.ragTool.checked,web_search:el.searchTool.checked,mcp:el.mcpTool.checked,mcp_server:el.mcpTool.checked?text(el.mcpServer.value)||"filesystem":null,tools:buildOpenAITools(),tool_choice:"auto"};
  const format=jsonResponseFormat(); if(format) payload.response_format=format;
  let result="",reasoning="",toolCalls=[];
  if(el.stream.checked){ await streamSse("/v1/responses",payload,(event)=>{ if(event.type==="response.output_text.delta"){result+=event.delta||"";setMessageContent(target,result,el.responseFormat.value);} else if(event.type==="response.reasoning.delta") reasoning+=event.delta||""; else if(event.type==="response.tool_calls.delta") toolCalls.push(...(event.tool_calls||[])); else if(event.type==="response.completed") updateUsage(event.response?.usage,null); }); }
  else {payload.stream=false; const response=await request("/v1/responses",{method:"POST",headers:jsonHeaders(),body:JSON.stringify(payload),signal:activeController.signal}); const body=await response.json(); result=body.output?.[0]?.content||""; setMessageContent(target,result,el.responseFormat.value); updateUsage(body.usage,body.status||"completed");}
  if(reasoning) addMessage("assistant",`Reasoning (server):\n${reasoning}`);
  if(toolCalls.length) addMessage("tool",JSON.stringify(toolCalls,null,2),"plain","Tool calls returned by backend");
  return result;
}
function updateUsage(usage,reason) { if(usage){const extras=[];if(usage.cached_tokens)extras.push(`${usage.cached_tokens} cached`);if(usage.reasoning_tokens)extras.push(`${usage.reasoning_tokens} reasoning`);el.usage.textContent=`${usage.prompt_tokens||0} prompt + ${usage.completion_tokens||0} generated tokens${extras.length?` · ${extras.join(" · ")}`:""}${reason?` · ${reason}`:""}`;} else if(reason) el.usage.textContent=`Completed · ${reason}`; }

async function cancelActiveRequest() {
  const id=activeRequestId;
  activeRequestId=null;
  if(!id || !el.apiKey.value.trim()) return;
  try {
    await request(`/v1/requests/${encodeURIComponent(id)}/cancel`,{method:"POST",headers:authHeaders()});
  } catch { /* transport may already be gone; local abort still stops the browser stream */ }
}

async function generate() {
  const invalid=validateUi(); if(invalid){showError(invalid);return;}
  setBusy(true); showError(); lastCompletedExample=null; el.reviewLast.disabled=true; const prompt=text(el.prompt.value); let target;
  try {
    const attachments=await readAttachments();
    addMessage("user",prompt); target=addMessage("assistant",""); el.prompt.value=""; el.prompt.style.height="";
    const requested=responseModeFor(attachments); setTransport(requested==="generate"?"Native WebSocket":requested==="chat"?"Chat Completions":"Responses API");
    activeController=new AbortController(); let answer="";
    if(requested==="generate") {
      const payload=buildGeneratePayload(prompt,attachments); payload.attachments=attachments.filter(a=>a.kind==="text").map(a=>({name:a.name,content:a.content,media_type:a.media_type}));
      answer=el.stream.checked?await generateNativeStream(payload,target):await generateNativeRest(payload,target);
    } else if(requested==="chat") answer=await generateChat(target);
    else answer=await generateResponses(target,attachments);
    if(answer.trim()){lastCompletedExample={prompt,answer};el.reviewLast.disabled=false;}
    persistLocalConversation(); setConversationTitle(); await refreshServerHistory();
  } catch(error) {
    if(error.name!=="AbortError") showError(error.message || "The generation request failed.");
    if(target && !target.dataset.rawText) setMessageContent(target,"Generation stopped.");
    persistLocalConversation();
  } finally { activeController?.abort();activeController=null;activeSocket=null;activeRequestId=null;setBusy(false);el.prompt.focus(); }
}

async function checkHealth() {
  try { const response=await request("/health/ready",{headers:authHeaders()}); const body=await response.json(); authenticationRequired=!!body.authentication_required; el.healthText.textContent=body.ready?"Server ready":"Server not ready"; el.health.classList.toggle("online",!!body.ready); if(authenticationRequired&&!el.apiKey.value.trim()){el.modelSummary.textContent="API key required. Enter GOPI_API_KEY in Connection settings to use protected API features.";el.historyStatus.textContent="Server API key required; local history remains available.";return;} await refreshModel(); await refreshServerHistory(); }
  catch(error){ el.healthText.textContent="Server offline"; el.health.classList.remove("online"); el.modelSummary.textContent=error.message; }
}
async function refreshModel() {
  try {
    const response=await request("/v1/models",{headers:authHeaders()}); const body=await response.json(); const model=body.data?.[0] || body.models?.[0] || body[0] || body; const id=model?.id || model?.model || "gopi"; window.GOPI_MODEL=id;
    const details={id,model:JSON.stringify(model)}; el.modelSummary.textContent=`Model: ${id}`; const [cap,resources,embeddingModels]=await Promise.allSettled([request(`/v1/models/${encodeURIComponent(id)}/capabilities`,{headers:authHeaders()}),request(`/v1/models/${encodeURIComponent(id)}/resources?context_length=${encodeURIComponent(Number(el.maxTokens.value))}&batch_size=1&weight_precision=bf16&kv_precision=bf16`,{headers:authHeaders()}),request(`/v1/embeddings/models`,{headers:authHeaders()})]);
    showDiagnostic({model:details,capabilities:cap.status==="fulfilled"?await cap.value.json():cap.reason?.message,resources:resources.status==="fulfilled"?await resources.value.json():resources.reason?.message,embedding_models:embeddingModels.status==="fulfilled"?await embeddingModels.value.json():embeddingModels.reason?.message});
  } catch(error){ el.modelSummary.textContent=`Model details unavailable: ${error.message}`; }
}
async function diagnosticGet(path) { try { const response=await request(path,{headers:authHeaders()}); showDiagnostic(await response.json()); } catch(error){showDiagnostic(error.message);} }
async function diagnosticPost(path,body={}) { try { const response=await request(path,{method:"POST",headers:jsonHeaders(),body:JSON.stringify(body)}); showDiagnostic(await response.json()); } catch(error){showDiagnostic(error.message);} }

async function loadReview() {
  try { const response=await request(`/v1/sessions/${encodeURIComponent(sessionId)}/training/review`,{headers:authHeaders()}); const body=await response.json(); el.reviewPrompt.value=body.prompt; el.reviewAnswer.value=body.answer; el.approveExample.checked=false; el.reviewStatus.textContent="Review the server copy before approval."; }
  catch(error){ if(lastCompletedExample){el.reviewPrompt.value=lastCompletedExample.prompt;el.reviewAnswer.value=lastCompletedExample.answer;el.reviewStatus.textContent=`Local fallback loaded (${error.message}).`; } else el.reviewStatus.textContent=error.message; }
}
async function approveTraining() {
  if(!el.approveExample.checked){el.reviewStatus.textContent="Check the approval box before approving.";return;}
  const prompt=text(el.reviewPrompt.value),answer=text(el.reviewAnswer.value); if(!prompt||!answer){el.reviewStatus.textContent="Prompt and reviewed answer are required.";return;}
  try { const response=await request(`/v1/sessions/${encodeURIComponent(sessionId)}/training/approve`,{method:"POST",headers:jsonHeaders(),body:JSON.stringify({approved:true,corrected_response:answer})}); const body=await response.json(); el.reviewStatus.textContent=`Approved ${body.example_count} server training example(s).`; el.approveExample.checked=false; }
  catch(error){el.reviewStatus.textContent=error.message;}
}
async function exportTraining() {
  try { const response=await request(`/v1/sessions/${encodeURIComponent(sessionId)}/training/export`,{headers:authHeaders()}); const blob=await response.blob(); downloadBlob(blob,`reviewed-chat-${sessionId}.jsonl`); el.reviewStatus.textContent="Approved training data exported."; }
  catch(error){ const prompt=text(el.reviewPrompt.value),answer=text(el.reviewAnswer.value); if(prompt&&answer&&el.approveExample.checked){ downloadBlob(new Blob([JSON.stringify({messages:[{role:"user",content:prompt},{role:"assistant",content:answer}]} )+"\n"],{type:"application/x-ndjson"}),`reviewed-chat-${sessionId}.jsonl`); el.reviewStatus.textContent=`Local export created; server export unavailable (${error.message}).`; } else el.reviewStatus.textContent=error.message; }
}
async function deleteTraining() { if(!confirm("Delete approved training examples for this conversation?"))return; try{const response=await request(`/v1/sessions/${encodeURIComponent(sessionId)}/training`,{method:"DELETE",headers:authHeaders()}); const body=await response.json(); el.reviewStatus.textContent=`Deleted ${body.deleted_count} approved training example(s).`;}catch(error){el.reviewStatus.textContent=error.message;} }
function downloadBlob(blob,name){const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}

function saveSettings(){
  const data={baseUrl:el.baseUrl.value,apiMode:el.apiMode.value,stream:el.stream.checked,useMemory:el.useMemory.checked,responseFormat:el.responseFormat.value,jsonSchema:el.jsonSchema.value,maxTokens:el.maxTokens.value,temperature:el.temperature.value,topK:el.topK.value,topP:el.topP.value,minTokens:el.minTokens.value,minP:el.minP.value,repetitionPenalty:el.repetitionPenalty.value,noRepeatNgram:el.noRepeatNgram.value,seed:el.seed.value,stopStrings:el.stopStrings.value,chatMode:el.chatMode.value,reasoningEffort:el.reasoningEffort.value,calculatorTool:el.calculatorTool.checked,datetimeTool:el.datetimeTool.checked,searchTool:el.searchTool.checked,ragTool:el.ragTool.checked,mcpTool:el.mcpTool.checked,mcpServer:el.mcpServer.value};
  localStorage.setItem(SETTINGS_KEY,JSON.stringify(data));
}
function loadSettings(){try{const d=JSON.parse(localStorage.getItem(SETTINGS_KEY)||"{}");for(const k of ["baseUrl","apiMode","responseFormat","jsonSchema","maxTokens","temperature","topK","topP","minTokens","minP","repetitionPenalty","noRepeatNgram","seed","stopStrings","chatMode","reasoningEffort","mcpServer"])if(d[k]!==undefined)el[k].value=d[k];for(const k of ["stream","useMemory","calculatorTool","datetimeTool","searchTool","ragTool","mcpTool"])if(d[k]!==undefined)el[k].checked=d[k];}catch{/* ignore corrupt browser settings */}refreshSettingLabels();}
function refreshSettingLabels(){el.maxTokensValue.textContent=el.maxTokens.value;el.temperatureValue.textContent=Number(el.temperature.value).toFixed(1);el.jsonSchemaWrap.hidden=!(["json_schema"].includes(el.responseFormat.value));el.modeDescription.textContent=modeDescriptions[el.chatMode.value]||"";}
function resetSettings(){el.apiMode.value="generate";el.stream.checked=true;el.useMemory.checked=true;el.responseFormat.value="plain";el.jsonSchema.value=JSON.stringify(DEFAULT_SCHEMA);el.maxTokens.value="128";el.temperature.value="0.7";el.topK.value="40";el.topP.value="0.9";el.minTokens.value="1";el.minP.value="0";el.repetitionPenalty.value="1.1";el.noRepeatNgram.value="3";el.seed.value="";el.stopStrings.value="";el.chatMode.value="balanced";el.reasoningEffort.value="none";el.calculatorTool.checked=false;el.datetimeTool.checked=false;el.searchTool.checked=false;el.ragTool.checked=false;el.mcpTool.checked=false;el.mcpServer.value="filesystem";refreshSettingLabels();saveSettings();}

el.form.addEventListener("submit",(e)=>{e.preventDefault();if(!generationInProgress)generate();});
el.prompt.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();if(!generationInProgress)el.form.requestSubmit();}});
el.prompt.addEventListener("input",()=>{el.prompt.style.height="auto";el.prompt.style.height=`${Math.min(el.prompt.scrollHeight,220)}px`;});
el.attachments.addEventListener("change",renderAttachmentPreview);
el.health.addEventListener("click",checkHealth); el.refreshHistory.addEventListener("click",refreshServerHistory); el.refreshModel.addEventListener("click",refreshModel);
el.newTop.addEventListener("click",()=>newConversation()); el.clear.addEventListener("click",()=>newConversation());
el.stop.addEventListener("click",()=>{cancelActiveRequest();activeController?.abort();if(activeSocket){activeSocket.close(1000);activeSocket=null;}setBusy(false);});
el.responseFormat.addEventListener("change",refreshSettingLabels);el.maxTokens.addEventListener("input",refreshSettingLabels);el.temperature.addEventListener("input",refreshSettingLabels);el.chatMode.addEventListener("change",()=>{refreshSettingLabels();saveSettings();});
el.apiKey.addEventListener("change",checkHealth);
el.adminApiKey.addEventListener("change",()=>{ showDiagnostic("Admin API key updated. It will be used for model lifecycle operations."); });
document.querySelector(".controls").addEventListener("change",saveSettings);
el.reset.addEventListener("click",resetSettings);
el.contextInfo.addEventListener("click",()=>diagnosticGet(`/v1/sessions/${encodeURIComponent(sessionId)}/context?reserve_tokens=${encodeURIComponent(Number(el.maxTokens.value))}`));
el.compactContext.addEventListener("click",()=>diagnosticPost(`/v1/sessions/${encodeURIComponent(sessionId)}/context/compact?reserve_tokens=${encodeURIComponent(Number(el.maxTokens.value))}`));
el.metricsButton.addEventListener("click",()=>fetch(`${apiBase()}/metrics`,{headers:authHeaders()}).then(async r=>{if(!r.ok)throw new Error(await responseMessage(r));showDiagnostic(await r.text());}).catch(e=>showDiagnostic(e.message)));
el.auditButton.addEventListener("click",()=>diagnosticGet("/v1/audit/events"));
el.embeddingButton.addEventListener("click",async()=>{try{const value=text(el.embeddingInput.value);if(!value)throw new Error("Embedding text cannot be empty.");const response=await request("/v1/embeddings",{method:"POST",headers:jsonHeaders(),body:JSON.stringify({model:window.GOPI_MODEL||"gopi",input:value})});const body=await response.json();showDiagnostic({model:body.model,data_count:body.data?.length||0,embedding_dimensions:body.data?.[0]?.embedding?.length||0,usage:body.usage});}catch(e){showDiagnostic(e.message);}});
el.workspaceButton.addEventListener("click",async()=>{try{const payload=JSON.parse(el.workspaceAction.value);const response=await request("/v1/workspace/actions",{method:"POST",headers:jsonHeaders(),body:JSON.stringify(payload)});showDiagnostic(await response.json());}catch(e){showDiagnostic(e.message);}});
for(const [button,verb] of [[el.lifecycleStatus,"GET"],[el.lifecycleLoad,"POST"],[el.lifecycleReload,"POST"],[el.lifecycleUnload,"POST"]])button.addEventListener("click",async()=>{const path=`/admin/models/${button===el.lifecycleStatus?"":button===el.lifecycleLoad?"load":button===el.lifecycleReload?"reload":"unload"}`.replace(/\/models\/$/,"/models");try{const response=await request(path,{method:verb,headers:adminHeaders()});showDiagnostic(await response.json());}catch(e){showDiagnostic(e.message);}});
el.reviewLast.addEventListener("click",loadReview);el.approveTraining.addEventListener("click",approveTraining);el.exportExample.addEventListener("click",exportTraining);el.deleteTraining.addEventListener("click",deleteTraining);
el.reviewAnswer.addEventListener("input",()=>{el.approveExample.checked=false;});

loadSettings(); renderAttachmentPreview(); restoreMessages(localConversations().find((x)=>x.sessionId===sessionId)?.messages || []);
setConversationTitle(); checkHealth();
