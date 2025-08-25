/* RAG Web Chat — CDN-robust WebLLM loader + all fixes
 * - Tries multiple CDNs for WebLLM (jsDelivr first, unpkg fallback)
 * - Dynamic import; worker bound to base URL
 * - FAST/HYBRID/FULL retrieval
 * - pdf.js UMD + worker
 * - small embedding model; dynamic import for Transformers.js
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
const pdfStatus = document.getElementById('pdf-status');
const modeSel = document.getElementById('mode');
const prefMEl = document.getElementById('prefM');

// Config
const TOP_K = 4;
topkShow.textContent = String(TOP_K);
const GEN_MODEL = 'Qwen2-0.5B-Instruct-q4f16_1-MLC';  // small demo
const GEN_TEMP = 0.7, GEN_TOP_P = 0.9;
const EMB_MODEL = 'Xenova/paraphrase-MiniLM-L3-v2';

modelGenEl.textContent = GEN_MODEL;
modelEmbEl.textContent = EMB_MODEL;

// Local DB
const DB_META = localforage.createInstance({ name: 'rag-meta' });
const DB_VEC = localforage.createInstance({ name: 'rag-vec' });
const DB_LUNR = localforage.createInstance({ name: 'rag-lunr' });

if (navigator.storage && navigator.storage.persist) {
  navigator.storage.persist().then(p => {
    const li = document.createElement('li');
    li.textContent = 'Storage persist: ' + (p ? 'granted' : 'not granted');
    statusEl.appendChild(li);
  });
}

const supportsWebGPU = !!navigator.gpu;
gpuEl.textContent = supportsWebGPU ? 'WebGPU OK' : 'WebGPU OFF (느림/미지원)';

let webllmChat = null;
let webllmBase = null;
let embedder = null;
let lunrIndex = null;
let stagedFiles = [];

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
function arrChunk(text, size=900, overlap=150) {
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
    const text = await extractPdfText(file);
    return { name: file.name, text };
  } else {
    const text = await file.text();
    return { name: file.name, text };
  }
}

// PDF text via pdf.js UMD
async function extractPdfText(file){
  const pdfjsLib = window['pdfjs-dist/build/pdf'];
  if (!pdfjsLib) { pdfStatus.textContent = 'pdf.js 로딩 실패'; return ''; }
  const arrayBuffer = await file.arrayBuffer();
  const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
  const pdf = await loadingTask.promise;
  pdfStatus.textContent = `PDF 페이지: ${pdf.numPages}`;
  let text = '';
  for (let i=1; i<=pdf.numPages; i++){
    const page = await pdf.getPage(i);
    const content = await page.getTextContent({ includeMarkedContent: true });
    const strings = content.items.map(it => it.str).join(' ');
    text += strings + '\n';
    if (i % 5 === 0) pdfStatus.textContent = `페이지 처리 중... ${i}/${pdf.numPages}`;
  }
  pdfStatus.textContent = `PDF 텍스트 추출 완료 (${pdf.numPages}p)`;
  return text;
}

// Indexing
indexBtn.addEventListener('click', async () => {
  if (!stagedFiles.length) return;
  setBusy(true); statusEl.innerHTML = ''; li('청크 생성...');

  const meta = [];
  let id = 0;
  const docsForBM25 = [];
  for (const f of stagedFiles){
    const chunks = arrChunk(f.text);
    for (const ch of chunks){
      meta.push({ id, file: f.name, text: ch });
      docsForBM25.push({ id: String(id), file: f.name, text: ch });
      id++;
    }
  }
  li(`총 청크: ${meta.length}`);
  await DB_META.setItem('meta', meta);

  // BM25 index
  li('BM25 인덱스 생성...');
  const idx = elasticlunr(function (){
    this.setRef('id');
    this.addField('text');
  });
  for (const d of docsForBM25){ idx.addDoc(d); }
  await DB_LUNR.setItem('bm25', idx.toJSON());
  li('BM25 인덱스 완료.');

  const mode = modeSel.value;
  if (mode === 'full'){
    li('임베딩 모델 로딩...');
    await ensureEmbedder();
    li('전 청크 임베딩...');
    const vecs = [];
    for (const m of meta){ vecs.push(await embedder.embed(m.text)); }
    await DB_VEC.setItem('vecs', vecs);
    li('임베딩 완료');
  } else {
    await DB_VEC.removeItem('vecs');
  }

  setBusy(false); li('인덱싱 완료');
});

clearBtn.addEventListener('click', async () => {
  await DB_META.removeItem('meta');
  await DB_VEC.removeItem('vecs');
  await DB_LUNR.removeItem('bm25');
  stagedFiles = [];
  filesEl.textContent = '';
  statusEl.innerHTML = '';
  pdfStatus.textContent = '';
  addMsg('인덱스 비움', 'bot');
});

// Ask
askBtn.addEventListener('click', async () => {
  const q = qEl.value.trim();
  if (!q) return;
  addMsg(q, 'user'); qEl.value='';
  setBusy(true);
  try {
    const { meta } = await loadMeta();
    if (!meta.length){ addMsg('인덱스가 비었습니다. 먼저 인덱싱하세요.', 'bot'); setBusy(false); return; }

    const mode = modeSel.value;
    let ctxs = [];
    if (mode === 'fast'){
      ctxs = await bm25Top(q, meta, TOP_K);
    } else if (mode === 'hybrid'){
      const M = Math.max( TOP_K, parseInt(prefMEl.value||'120',10) );
      const pre = await bm25Top(q, meta, M);
      await ensureEmbedder();
      const qv = await embedder.embed(q);
      const scores = pre.map(p => ({ sc: cosine(p.text, qv), it: p }));
      scores.sort((a,b)=>b.sc-a.sc);
      ctxs = scores.slice(0, TOP_K).map(s => ({ ...s.it, score: s.sc }));
    } else {
      const { vecs } = await loadVecs();
      if (!vecs.length){ addMsg('벡터 인덱스가 없습니다. FULL 모드로 다시 인덱싱하세요.', 'bot'); setBusy(false); return; }
      await ensureEmbedder();
      const qv = await embedder.embed(q);
      const scores = vecs.map((v,i)=>({ sc: cosine(v, qv), idx:i }));
      scores.sort((a,b)=>b.sc-a.sc);
      ctxs = scores.slice(0, TOP_K).map(s => ({ ...meta[s.idx], score: s.sc }));
    }

    const prompt = buildPrompt(q, ctxs);
    const ans = await generate(prompt);
    addMsg(ans, 'bot');
    const srcs = ctxs.map(c => `- ${c.file} (score ${c.score!==undefined? c.score.toFixed(3):'BM25'})`).join('\n');
    addMsg(srcs, 'bot');
  } catch (e){
    console.error(e);
    addMsg('오류: ' + e.message, 'bot');
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

// Helpers
function cosine(a,b){
  let s=0, na=0, nb=0;
  for (let i=0;i<a.length;i++){ s+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return s/(Math.sqrt(na)*Math.sqrt(nb)+1e-9);
}
async function loadMeta(){ const meta = await DB_META.getItem('meta') || []; return { meta }; }
async function loadVecs(){ const vecs = await DB_VEC.getItem('vecs') || []; return { vecs }; }

// BM25 via elasticlunr
async function bm25Top(q, meta, k){
  let idxJson = await DB_LUNR.getItem('bm25');
  if (!idxJson){ return meta.slice(0, k); }
  const idx = elasticlunr.Index.load(idxJson);
  const hits = idx.search(q, { expand:true });
  const pick = hits.slice(0, k).map(h => {
    const m = meta.find(mm => String(mm.id) === h.ref);
    return { ...m, score: h.score };
  });
  return pick;
}

// Embeddings via Transformers.js — dynamic ESM import (jsDelivr)
async function ensureEmbedder(){
  if (embedder) return;
  try {
    const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js');
    const pipe = await mod.pipeline('feature-extraction', EMB_MODEL, { quantized: true });
    embedder = { async embed(text){ const out = await pipe(text, { pooling:'mean', normalize:true }); return Array.from(out.data);} };
  } catch (e) {
    console.error('임베딩 로더 오류', e);
    addMsg('임베딩 모델 로딩 실패: 네트워크(CDN) 접근을 확인하거나 FAST 모드로 사용하세요.', 'bot');
    throw e;
  }
}

// WebLLM multi-CDN loader
async function ensureWebLLM(){
  if (webllmChat) return;
  const bases = [
    'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm/dist/',
    'https://unpkg.com/@mlc-ai/web-llm/dist/',
  ];
  let lastErr = null;
  for (const base of bases){
    try {
      li('WebLLM 불러오는 중: ' + base);
      const mod = await importWithTimeout(base + 'index.js', 12000);
      const worker = new Worker(base + 'worker.js', { type: 'module' });
      const initProgressCallback = (p) => { if (p.text) li(p.text); };
      const config = { model_id: GEN_MODEL };
      webllmChat = await mod.CreateWebWorkerMLCEngine(worker, config, initProgressCallback);
      webllmBase = base;
      return;
    } catch (e){
      console.warn('WebLLM 로드 실패 @', base, e);
      lastErr = e;
    }
  }
  addMsg('WebLLM 라이브러리 로드 실패(네트워크 차단 가능). FAST 모드에서 검색만 이용 가능합니다.', 'bot');
  throw lastErr || new Error('WebLLM load failed');
}

function importWithTimeout(url, ms){
  return Promise.race([
    import(url),
    new Promise((_,rej)=>setTimeout(()=>rej(new Error('Timeout loading ' + url)), ms))
  ]);
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

// Warmup
li('브라우저 WebGPU: ' + (navigator.gpu ? 'OK' : 'OFF'));
