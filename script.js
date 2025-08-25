/* RAG Web Chat — No‑LLM Fallback (v3)
 * - If LLM can't load or 'LLM 사용' is off, produces extractive answers from top-K chunks
 * - FAST/HYBRID/FULL retrieval; pdf.js; dynamic transformers import (only when needed)
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
const topkShow = document.getElementById('topkShow');
const pdfStatus = document.getElementById('pdf-status');
const modeSel = document.getElementById('mode');
const prefMEl = document.getElementById('prefM');
const useLLM = document.getElementById('useLLM');

// Config
const TOP_K = 4;
topkShow.textContent = String(TOP_K);
// Smaller embedding for speed
const EMB_MODEL = 'Xenova/paraphrase-MiniLM-L3-v2';

// DB
const DB_META = localforage.createInstance({ name: 'rag-meta' });
const DB_VEC = localforage.createInstance({ name: 'rag-vec' });
const DB_LUNR = localforage.createInstance({ name: 'rag-lunr' });

const supportsWebGPU = !!navigator.gpu;
gpuEl.textContent = supportsWebGPU ? 'WebGPU OK' : 'WebGPU OFF (느림/미지원)';

let embedder = null;
let stagedFiles = [];

function li(msg) { const li = document.createElement('li'); li.textContent = msg; statusEl.appendChild(li); }
function setBusy(b){ busy.style.display = b ? 'inline-block' : 'none'; }
function addMsg(text, who){ const el = document.createElement('div'); el.className = 'msg ' + who; el.textContent = text; messagesEl.appendChild(el); messagesEl.scrollTop = messagesEl.scrollHeight; }
function arrChunk(text, size=900, overlap=150) { if (text.length <= size) return [text]; const out=[]; for (let i=0;i<text.length;i+=(size-overlap)) out.push(text.slice(i, i+size)); return out; }

dropEl.addEventListener('dragover', e => { e.preventDefault(); dropEl.style.borderColor = '#4d65ff'; });
dropEl.addEventListener('dragleave', e => { dropEl.style.borderColor = ''; });
dropEl.addEventListener('drop', async e => {
  e.preventDefault(); dropEl.style.borderColor = '';
  const items = e.dataTransfer.items; const tasks = [];
  for (const item of items){ const file = item.getAsFile(); if (file) tasks.push(readFile(file)); }
  const results = await Promise.all(tasks); results.forEach(r => r && stagedFiles.push(r));
  filesEl.textContent = stagedFiles.map(s => s.name).join(', ');
  if (stagedFiles.length) indexBtn.disabled = false;
});

async function readFile(file){
  const name = file.name.toLowerCase();
  if (!(name.endsWith('.txt') || name.endsWith('.md') || name.endsWith('.pdf'))) return null;
  if (name.endsWith('.pdf')){ const text = await extractPdfText(file); return { name: file.name, text }; }
  const text = await file.text(); return { name: file.name, text };
}

// PDF via pdf.js
async function extractPdfText(file){
  const pdfjsLib = window['pdfjs-dist/build/pdf'];
  if (!pdfjsLib) { pdfStatus.textContent = 'pdf.js 로딩 실패'; return ''; }
  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  pdfStatus.textContent = `PDF 페이지: ${pdf.numPages}`;
  let text = ''; for (let i=1; i<=pdf.numPages; i++){
    const page = await pdf.getPage(i);
    const content = await page.getTextContent({ includeMarkedContent: true });
    text += content.items.map(it => it.str).join(' ') + '\n';
    if (i % 5 === 0) pdfStatus.textContent = `페이지 처리 중... ${i}/${pdf.numPages}`;
  }
  pdfStatus.textContent = `PDF 텍스트 추출 완료 (${pdf.numPages}p)`; return text;
}

indexBtn.addEventListener('click', async () => {
  if (!stagedFiles.length) return;
  setBusy(true); statusEl.innerHTML = ''; li('청크 생성...');

  const meta = []; let id = 0; const docsForBM25 = [];
  for (const f of stagedFiles){
    const chunks = arrChunk(f.text);
    for (const ch of chunks){ meta.push({ id, file: f.name, text: ch }); docsForBM25.push({ id: String(id), file: f.name, text: ch }); id++; }
  }
  li(`총 청크: ${meta.length}`); await DB_META.setItem('meta', meta);

  li('BM25 인덱스 생성...');
  const idx = elasticlunr(function (){ this.setRef('id'); this.addField('text'); });
  for (const d of docsForBM25){ idx.addDoc(d); }
  await DB_LUNR.setItem('bm25', idx.toJSON()); li('BM25 인덱스 완료.');

  // HYBRID/FULL은 질문 시에만 임베딩 필요 → upfront 임베딩 제거
  await DB_VEC.removeItem('vecs');

  setBusy(false); li('인덱싱 완료');
});

clearBtn.addEventListener('click', async () => {
  await DB_META.removeItem('meta'); await DB_VEC.removeItem('vecs'); await DB_LUNR.removeItem('bm25');
  stagedFiles = []; filesEl.textContent = ''; statusEl.innerHTML = ''; pdfStatus.textContent = ''; addMsg('인덱스 비움', 'bot');
});

askBtn.addEventListener('click', async () => {
  const q = qEl.value.trim(); if (!q) return; addMsg(q, 'user'); qEl.value=''; setBusy(true);
  try {
    const meta = await DB_META.getItem('meta') || [];
    if (!meta.length){ addMsg('인덱스가 비었습니다. 먼저 인덱싱하세요.', 'bot'); setBusy(false); return; }

    const mode = modeSel.value;
    let ctxs = [];
    if (mode === 'fast'){
      ctxs = await bm25Top(q, meta, TOP_K);
    } else if (mode === 'hybrid'){
      const M = Math.max(TOP_K, parseInt(prefMEl.value||'120',10));
      const pre = await bm25Top(q, meta, M);
      await ensureEmbedder(); const qv = await embedder.embed(q);
      const scores = pre.map(p => ({ sc: cosine(p.text, qv), it: p })).sort((a,b)=>b.sc-a.sc);
      ctxs = scores.slice(0, TOP_K).map(s => ({ ...s.it, score: s.sc }));
    } else {
      const vecs = await DB_VEC.getItem('vecs') || []; // 없도록 설계
      if (!vecs.length){
        addMsg('FULL 모드를 쓰려면 먼저 FULL로 다시 인덱싱하세요. 지금은 HYBRID/FAST 권장.', 'bot');
        setBusy(false); return;
      }
    }

    // ======== No-LLM fallback ========
    if (!useLLM.checked){
      addMsg(renderExtractiveAnswer(q, ctxs), 'bot');
      addMsg(ctxs.map(c => `- ${c.file} (score ${c.score!==undefined? c.score.toFixed(3):'BM25'})`).join('\n'), 'bot');
      setBusy(false); return;
    }

    // If user enabled LLM, try to load via CDN (may fail in blocked networks)
    try {
      const ans = await generateLLM(buildPrompt(q, ctxs));
      addMsg(ans, 'bot');
    } catch (e){
      addMsg('LLM 로드 실패 → 추출형 답변으로 대체합니다.', 'bot');
      addMsg(renderExtractiveAnswer(q, ctxs), 'bot');
    }
    addMsg(ctxs.map(c => `- ${c.file} (score ${c.score!==undefined? c.score.toFixed(3):'BM25'})`).join('\n'), 'bot');
  } catch (e){ console.error(e); addMsg('오류: ' + e.message, 'bot'); } finally { setBusy(false); }
});

function renderExtractiveAnswer(q, ctxs){
  const header = '※ LLM 없이 문서에서 직접 발췌한 답변(요약 아님)';
  const body = ctxs.map((c,i)=>`[${i+1}] ${c.file}\n${snippet(c.text)}\n`).join('\n');
  return `${header}\n\n${body}`;
}
function snippet(t){
  const s = t.replace(/\s+/g,' ').trim();
  return s.slice(0, 500) + (s.length>500 ? ' …' : '');
}
function buildPrompt(q, ctxs){
  const joined = ctxs.map((c,i)=>`[DOC ${i+1}] from ${c.file}:\n${c.text}`).join('\n\n');
  const sys = 'You are a helpful RAG assistant. Answer using ONLY the provided documents. If insufficient, say so. Cite filenames like [source: file]. Answer in Korean if user is Korean.';
  const srcHint = ctxs.map(c => `[source: ${c.file}]`).join(' ');
  return `<<SYS>>${sys}<</SYS>>\nQuestion:\n${q}\n\nContext:\n${joined}\n\nWhen answering, include citations: ${srcHint}\n\nAnswer:`;
}

// Embeddings (only when HYBRID)
async function ensureEmbedder(){
  if (embedder) return;
  const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js');
  const pipe = await mod.pipeline('feature-extraction', EMB_MODEL, { quantized: true });
  embedder = { async embed(text){ const out = await pipe(text, { pooling:'mean', normalize:true }); return Array.from(out.data);} };
}

// Generate via WebLLM (CDN; may fail → handled by try/catch)
async function ensureWebLLM(){
  const bases = [
    'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm/dist/',
    'https://unpkg.com/@mlc-ai/web-llm/dist/',
  ];
  for (const base of bases){
    try {
      const mod = await import(base + 'index.js');
      const worker = new Worker(base + 'worker.js', { type: 'module' });
      const initProgressCallback = (p) => { if (p.text) li(p.text); };
      return { engine: await mod.CreateWebWorkerMLCEngine(worker, { model_id: 'Qwen2-0.5B-Instruct-q4f16_1-MLC' }, initProgressCallback) };
    } catch {}
  }
  throw new Error('WebLLM blocked');
}
async function generateLLM(prompt){
  const { engine } = await ensureWebLLM();
  const reply = await engine.chat.completions.create({ messages: [{ role:'user', content: prompt }], temperature: 0.7, top_p: 0.9, stream: false });
  return reply.choices?.[0]?.message?.content || '(no response)';
}

// BM25 + helpers
async function bm25Top(q, meta, k){
  let idxJson = await DB_LUNR.getItem('bm25'); if (!idxJson) return meta.slice(0,k);
  const idx = elasticlunr.Index.load(idxJson); const hits = idx.search(q, { expand:true });
  return hits.slice(0, k).map(h => { const m = meta.find(mm => String(mm.id) === h.ref); return { ...m, score: h.score }; });
}
function cosine(a,b){ let s=0, na=0, nb=0; for (let i=0;i<a.length;i++){ s+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; } return s/(Math.sqrt(na)*Math.sqrt(nb)+1e-9); }

li('브라우저 WebGPU: ' + (navigator.gpu ? 'OK' : 'OFF'));
