/* RAG Web Chat (Client‑Only)
 * - Generation: WebLLM
 * - Embedding: Transformers.js (all-MiniLM-L6-v2)
 * - Vector store: IndexedDB via localForage
 */
const statusEl = document.getElementById('status');
const filesEl = document.getElementById('files');
const dropEl = document.getElementById('drop');
const askBtn = document.getElementById('ask');
const indexBtn = document.getElementById('btn-index');
const clearBtn = document.getElementById('btn-clear');
const busy = document.getElementById('busy');
const messagesEl = document.getElementById('messages');
const qEl = document.getElementById('q');
const gpuEl = document.getElementById('gpu');
const modelGenEl = document.getElementById('modelGen');
const modelEmbEl = document.getElementById('modelEmb');
const topkShow = document.getElementById('topkShow');

// Config
const TOP_K = 4;
topkShow.textContent = String(TOP_K);
const GEN_MODEL = 'Qwen2-0.5B-Instruct-q4f16_1-MLC';  // small, demo; change for better KR
const GEN_TEMP = 0.7, GEN_TOP_P = 0.9;
const EMB_MODEL = 'Xenova/all-MiniLM-L6-v2';

modelGenEl.textContent = GEN_MODEL;
modelEmbEl.textContent = EMB_MODEL;

// Local DB
const DB_META = localforage.createInstance({ name: 'rag-meta' });
const DB_VEC = localforage.createInstance({ name: 'rag-vec' });

const supportsWebGPU = !!navigator.gpu;
gpuEl.textContent = supportsWebGPU ? 'WebGPU OK' : 'WebGPU OFF (느림/미지원)';

// Lib handles
let webllmChat = null;
let embedder = null;

// Selected files cache (session only)
let stagedFiles = []; // [{name,text}]

function li(msg) {
  const li = document.createElement('li'); li.textContent = msg; statusEl.appendChild(li);
}
function setBusy(b){ busy.style.display = b ? 'inline-block' : 'none'; }
function addMsg(text, who){
  const el = document.createElement('div');
  el.className = 'msg ' + who;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
function arrChunk(text, size=1100, overlap=200) {
  if (text.length <= size) return [text];
  const out = [];
  for (let i=0; i<text.length; i+= (size-overlap)) {
    out.push(text.slice(i, i+size));
  }
  return out;
}

// File handling
dropEl.addEventListener('dragover', e => { e.preventDefault(); dropEl.style.borderColor = '#4d65ff'; });
dropEl.addEventListener('dragleave', e => { dropEl.style.borderColor = ''; });
dropEl.addEventListener('drop', async e => {
  e.preventDefault(); dropEl.style.borderColor = '';
  const items = e.dataTransfer.items;
  const tasks = [];
  for (const item of items){
    const file = item.getAsFile();
    if (!file) continue;
    tasks.push(readFile(file));
  }
  const results = await Promise.all(tasks);
  results.forEach(r => r && stagedFiles.push(r));
  filesEl.textContent = stagedFiles.map(s => s.name).join(', ');
  if (stagedFiles.length) indexBtn.disabled = false;
});

async function readFile(file){
  const name = file.name.toLowerCase();
  if (!(name.endsWith('.txt') || name.endsWith('.md') || name.endsWith('.pdf'))) return null;
  if (name.endsWith('.pdf')){
    // Use PDF.js? For simplicity, rely on built-in text extraction via PDF.js CDN (light)
    const text = await extractPdfText(file);
    return { name: file.name, text };
  } else {
    const text = await file.text();
    return { name: file.name, text };
  }
}

async function extractPdfText(file){
  // Lightweight pdf.js extraction via dynamic import
  await import('https://cdn.jsdelivr.net/npm/pdfjs-dist@4.7.76/build/pdf.min.mjs');
  const pdfjsLib = window['pdfjs-dist/build/pdf'];
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.7.76/build/pdf.worker.min.mjs';
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  let text = '';
  for (let i=1; i<=pdf.numPages; i++){
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const strings = content.items.map(it => it.str).join(' ');
    text += strings + '\n';
  }
  return text;
}

// Indexing
indexBtn.addEventListener('click', async () => {
  if (!stagedFiles.length) return;
  setBusy(true); statusEl.innerHTML = ''; li('임베딩 모델 로딩...');
  await ensureEmbedder();
  li('청크 생성 및 벡터화...');
  const meta = []; // {id, file, text}
  const vecs = []; // Float32Array
  let id = 0;
  for (const f of stagedFiles){
    const chunks = arrChunk(f.text);
    for (const ch of chunks){
      const emb = await embedder.embed(ch);
      meta.push({ id, file: f.name, text: ch });
      vecs.push(emb);
      id++;
    }
  }
  li(`총 청크: ${meta.length}`);
  await DB_META.setItem('meta', meta);
  await DB_VEC.setItem('vecs', vecs);
  setBusy(false); li('인덱싱 완료');
});

clearBtn.addEventListener('click', async () => {
  await DB_META.removeItem('meta');
  await DB_VEC.removeItem('vecs');
  stagedFiles = [];
  filesEl.textContent = '';
  statusEl.innerHTML = '';
  addMsg('인덱스 비움', 'bot');
});

// Ask
askBtn.addEventListener('click', async () => {
  const q = qEl.value.trim();
  if (!q) return;
  addMsg(q, 'user'); qEl.value='';
  setBusy(true);
  try {
    const { meta, vecs } = await loadIndex();
    if (!meta.length){ addMsg('인덱스가 비었습니다. 먼저 인덱싱하세요.', 'bot'); setBusy(false); return; }
    await ensureEmbedder();
    const qvec = await embedder.embed(q);
    // cosine sim
    const scores = vecs.map(v => dot(v,qvec)/(norm(v)*norm(qvec)+1e-9));
    const idx = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).slice(0, TOP_K).map(x=>x[1]);
    const ctxs = idx.map(i => ({ ...meta[i], score: scores[i] }));
    const prompt = buildPrompt(q, ctxs);
    const ans = await generate(prompt);
    addMsg(ans, 'bot');
    // sources
    const srcs = ctxs.map(c => `- ${c.file} (score ${c.score.toFixed(3)})`).join('\n');
    addMsg(srcs, 'bot');
  } catch (e){
    console.error(e);
    addMsg('오류: ' + e, 'bot');
  } finally {
    setBusy(false);
  }
});

function buildPrompt(q, ctxs){
  const joined = ctxs.map((c,i)=>`[DOC ${i+1}] from ${c.file}:\n${c.text}`).join('\n\n');
  const srcHint = ctxs.map(c => `[source: ${c.file}]`).join(' ');
  const sys = 'You are a helpful RAG assistant. Answer using ONLY the provided documents. If insufficient, say so. Cite filenames like [source: file]. Answer in Korean if user is Korean.';
  return `<<SYS>>${sys}<</SYS>>\nQuestion:\n${q}\n\nContext:\n${joined}\n\nWhen answering, include citations: ${srcHint}\n\nAnswer:`;
}

// Linear algebra helpers
function dot(a,b){ let s=0; for (let i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
function norm(a){ let s=0; for (let i=0;i<a.length;i++) s+=a[i]*a[i]; return Math.sqrt(s); }

async function loadIndex(){
  const meta = await DB_META.getItem('meta') || [];
  const vecs = await DB_VEC.getItem('vecs') || [];
  return { meta, vecs };
}

// Embeddings via Transformers.js
async function ensureEmbedder(){
  if (embedder) return;
  const pipe = await window.transformers.pipeline('feature-extraction', EMB_MODEL, { quantized: true });
  embedder = {
    async embed(text){
      const out = await pipe(text, { pooling: 'mean', normalize: true });
      // out is Tensor; to array
      return Array.from(out.data);
    }
  };
}

// Generation via WebLLM
async function ensureWebLLM(){
  if (webllmChat) return;
  const { CreateWebWorkerMLCEngine, MLCEngineConfig } = await import('https://unpkg.com/@mlc-ai/web-llm/dist/index.js');
  const worker = new Worker('https://unpkg.com/@mlc-ai/web-llm/dist/worker.js', { type: 'module' });
  const initProgressCallback = (p) => {
    if (p.text) li(p.text);
  };
  const config = { model_id: GEN_MODEL };
  webllmChat = await CreateWebWorkerMLCEngine(worker, config, initProgressCallback);
}
async function generate(prompt){
  await ensureWebLLM();
  const tmpl = { role: 'user', content: prompt };
  const reply = await webllmChat.chat.completions.create({
    messages: [tmpl],
    temperature: GEN_TEMP,
    top_p: GEN_TOP_P,
    stream: false
  });
  return reply.choices?.[0]?.message?.content || '(no response)';
}

// Warmup status
li('브라우저 WebGPU: ' + (supportsWebGPU ? 'OK' : 'OFF'));