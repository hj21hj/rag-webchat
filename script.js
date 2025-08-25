// Smart RAG Workspace (Tokenless) — FAST/HYBRID without external LLMs
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

const statusEl = $("#status");
const filesEl  = $("#files");
const msgEl    = $("#messages");

function li(t){ const el=document.createElement('li'); el.textContent=t; statusEl.appendChild(el); }
function addMsg(t,who='bot'){
  const w=document.createElement('div'); w.className='message';
  const bubble=document.createElement('div');
  bubble.className = 'msg ' + (who==='user' ? 'user' : 'bot');
  bubble.innerHTML = t;
  w.appendChild(bubble); msgEl.appendChild(w); msgEl.scrollTop = msgEl.scrollHeight;
}

const DB_META = localforage.createInstance({ name:'srw-meta' });
const DB_LUNR = localforage.createInstance({ name:'srw-lunr' });
const DB_VEC  = localforage.createInstance({ name:'srw-vec' });
const DB_ROOMS= localforage.createInstance({ name:'srw-rooms' });
const DB_CHAT = localforage.createInstance({ name:'srw-chat' });

let currentRoom = null;
let stagedFiles = [];
let settings = { chunking:'fixed', chunkSize:900, overlap:150, searchK:4 };
let embedder = null;

document.getElementById('kShow').textContent = String(settings.searchK);
document.getElementById('gpu').textContent = (navigator.gpu ? 'WebGPU OK' : 'WebGPU OFF');

// Rooms
async function loadRooms(){ return (await DB_ROOMS.getItem('rooms')) || []; }
async function saveRooms(rooms){ await DB_ROOMS.setItem('rooms', rooms); }
async function addRoom(name){
  const rooms = await loadRooms();
  const r = { id:String(Date.now()), name, created_at:new Date().toISOString(), chat_count:0 };
  rooms.push(r); await saveRooms(rooms); renderRooms(rooms); await selectRoom(r.id);
}
async function removeRoom(id){
  const rooms = (await loadRooms()).filter(r=>r.id!==id);
  await saveRooms(rooms);
  if (currentRoom?.id===id){ currentRoom=null; msgEl.innerHTML=''; filesEl.textContent='없음'; }
  await DB_META.removeItem('meta:'+id);
  await DB_LUNR.removeItem('idx:'+id);
  await DB_VEC.removeItem('vec:'+id);
  await DB_CHAT.removeItem('chat:'+id);
  renderRooms(rooms);
}
async function selectRoom(id){
  const rooms = await loadRooms();
  currentRoom = rooms.find(r=>r.id===id);
  document.title = currentRoom?.name ? currentRoom.name + ' · Smart RAG Workspace' : 'Smart RAG Workspace';
  msgEl.innerHTML='';
  const chat = await loadChat();
  for (const m of chat) addMsg(m.content, m.sender==='user'?'user':'bot');
  filesEl.textContent='없음';
}
function renderRooms(rooms){
  const list = $("#room-list"); list.innerHTML='';
  rooms.forEach(r=>{
    const el=document.createElement('div'); el.className='room-item';
    el.innerHTML = \`
      <div class="room-name" onclick="window._sel('\${r.id}')">\${r.name}</div>
      <button class="icon-btn" title="삭제" onclick="window._del('\${r.id}')">🗑</button>\`;
    list.appendChild(el);
  });
}
window._sel = (id)=>selectRoom(id);
window._del = (id)=>removeRoom(id);

async function loadChat(){ return (await DB_CHAT.getItem('chat:'+currentRoom?.id)) || []; }
async function pushChat(sender, content){
  if (!currentRoom) return;
  const key='chat:'+currentRoom.id;
  const hist = (await DB_CHAT.getItem(key)) || [];
  hist.push({ sender, content, ts: new Date().toISOString() });
  await DB_CHAT.setItem(key, hist);
}

$("#btn-new-room").addEventListener('click', async ()=>{
  const name = prompt('새 룸 이름을 입력하세요');
  if (name) await addRoom(name.trim());
});

// Chat ask
$("#q").addEventListener('keydown', e=>{
  if (e.key==='Enter' && !e.shiftKey){
    e.preventDefault(); $("#btn-ask").click();
  }
});
$("#btn-ask").addEventListener('click', async ()=>{
  if (!currentRoom) return alert('먼저 룸을 만드세요.');
  const q = $("#q").value.trim(); if (!q) return;
  $("#q").value=''; addMsg(q,'user'); await pushChat('user', q);

  const meta = (await DB_META.getItem('meta:'+currentRoom.id)) || [];
  if (!meta.length){ addMsg('인덱스가 없습니다. “업로드→인덱싱”을 먼저 실행하세요.'); return; }

  const mode = $("#mode").value;
  const ctxs = await searchWithMode(q, meta, mode, settings.searchK);
  const body = ctxs.map((c,i)=>\`[\${i+1}] \${escapeHtml(c.file)}<br>\${escapeHtml(snippet(c.text))}\`).join('<br><br>');
  const ans  = \`<div class="tiny">※ 발췌형 답변</div><div>\${body}</div>\`;
  addMsg(ans, 'bot'); await pushChat('assistant', ans);
});

function escapeHtml(s){ return s.replace(/[&<>]/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;' }[m])); }

// Search
async function searchWithMode(q, meta, mode, k){
  if (mode==='fast') return await bm25Top(q, meta, k);
  if (mode==='hybrid'){
    const pre = await bm25Top(q, meta, Math.max(k, 120));
    try{
      await ensureEmbedder();
      const qv = await embedder.embed(q);
      return pre.map(p=>({ it:p, sc:cosine(p.text, qv) }))
                .sort((a,b)=>b.sc-a.sc).slice(0,k)
                .map(s=>({ ...s.it, score:s.sc }));
    }catch(e){
      li('임베딩 로드 실패 → FAST로 대체');
      return pre.slice(0,k);
    }
  }
  // FULL (optional precomputed vector route — not precomputed by default)
  const vecs = (await DB_VEC.getItem('vec:'+currentRoom.id)) || [];
  if (!vecs.length){ addMsg('벡터 인덱스가 없습니다. FULL 모드로 다시 인덱싱하세요.'); return await bm25Top(q, meta, k); }
  await ensureEmbedder();
  const qv = await embedder.embed(q);
  return vecs.map((v,i)=>({ sc:cosine(v,qv), idx:i }))
             .sort((a,b)=>b.sc-a.sc).slice(0,k)
             .map(s=>({ ...meta[s.idx], score:s.sc }));
}

async function bm25Top(q, meta, k){
  const idxJson = await DB_LUNR.getItem('idx:'+currentRoom.id);
  if (!idxJson) return meta.slice(0,k);
  const idx = elasticlunr.Index.load(idxJson);
  const hits = idx.search(q, { expand:true });
  return hits.slice(0,k).map(h=>{ const m=meta.find(mm=>String(mm.id)===h.ref); return { ...m, score:h.score }; });
}
function cosine(a,b){ let s=0,na=0,nb=0; for(let i=0;i<a.length;i++){ s+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; } return s/(Math.sqrt(na)*Math.sqrt(nb)+1e-9); }

// Embedder (on-demand for HYBRID)
async function ensureEmbedder(){
  if (embedder) return;
  const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js');
  const pipe = await mod.pipeline('feature-extraction', 'Xenova/paraphrase-MiniLM-L3-v2', { quantized:true });
  embedder = { async embed(text){ const out = await pipe(text, { pooling:'mean', normalize:true }); return Array.from(out.data); } };
}

// Upload modal
const modalUpload = $("#modal-upload");
$("#btn-upload").addEventListener('click', ()=>{
  if (!currentRoom) return alert('먼저 룸을 만드세요.');
  openModal('modal-upload');
});
$$('[data-close]').forEach(b=>b.addEventListener('click', e=>closeModal(e.currentTarget.dataset.close)));
function openModal(id){ const el = document.getElementById(id); el.classList.remove('hidden'); }
function closeModal(id){ const el = document.getElementById(id); el.classList.add('hidden'); }

const drop = $("#drop"), fileInput = $("#file");
drop.addEventListener('click', ()=>fileInput.click());
drop.addEventListener('dragover', e=>{ e.preventDefault(); drop.classList.add('hover'); });
drop.addEventListener('dragleave', ()=>drop.classList.remove('hover'));
drop.addEventListener('drop', async e=>{
  e.preventDefault(); drop.classList.remove('hover');
  const tasks=[]; for (const it of e.dataTransfer.items){ const f=it.getAsFile(); if(f) tasks.push(readFile(f)); }
  const rs = await Promise.all(tasks); rs.forEach(r=>r&&stagedFiles.push(r));
  filesEl.textContent = stagedFiles.map(s=>s.name).join(', ') || '없음';
  document.getElementById('btn-index').disabled = stagedFiles.length===0;
});
fileInput.addEventListener('change', async e=>{
  const tasks=[]; for (const f of e.target.files){ tasks.push(readFile(f)); }
  const rs = await Promise.all(tasks); rs.forEach(r=>r&&stagedFiles.push(r));
  filesEl.textContent = stagedFiles.map(s=>s.name).join(', ') || '없음';
  document.getElementById('btn-index').disabled = stagedFiles.length===0;
});

async function readFile(file){
  const name=file.name.toLowerCase();
  if (!(name.endsWith('.txt')||name.endsWith('.md')||name.endsWith('.pdf'))) return null;
  if (name.endsWith('.pdf')){ const text = await extractPdfText(file); return { name:file.name, text }; }
  const text = await file.text(); return { name:file.name, text };
}
async function extractPdfText(file){
  const pdfjsLib = window['pdfjs-dist/build/pdf'];
  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  document.getElementById('pdf-status').textContent = `PDF 페이지: ${pdf.numPages}`;
  let text=''; for (let i=1;i<=pdf.numPages;i++){
    const page=await pdf.getPage(i);
    const c=await page.getTextContent({ includeMarkedContent:true });
    text += c.items.map(it=>it.str).join(' ') + '\n';
    if (i%5===0) document.getElementById('pdf-status').textContent = `페이지 처리 중... ${i}/${pdf.numPages}`;
  }
  document.getElementById('pdf-status').textContent = `PDF 텍스트 추출 완료 (${pdf.numPages}p)`;
  return text;
}

document.getElementById('btn-index').addEventListener('click', async ()=>{
  if (!currentRoom || !stagedFiles.length) return;
  li('청크 생성...');
  const { chunking, chunkSize, overlap } = settings;
  const meta=[], docs=[]; let id=0;
  for (const f of stagedFiles){
    const chunks = chunkText(f.text, chunking, chunkSize, overlap);
    for (const ch of chunks){ meta.push({ id, file:f.name, text:ch }); docs.push({ id:String(id), file:f.name, text:ch }); id++; }
  }
  li('BM25 인덱스 생성...');
  const idx = elasticlunr(function(){ this.setRef('id'); this.addField('text'); });
  for (const d of docs) idx.addDoc(d);
  await DB_META.setItem('meta:'+currentRoom.id, meta);
  await DB_LUNR.setItem('idx:'+currentRoom.id, idx.toJSON());
  document.getElementById('perf-docs').textContent = new Set(stagedFiles.map(s=>s.name)).size;
  document.getElementById('perf-chunks').textContent = meta.length;
  li('인덱싱 완료');
  stagedFiles=[]; filesEl.textContent='업로드 완료';
  document.getElementById('btn-index').disabled = true;
  document.getElementById('modal-upload').classList.add('hidden');
});

function chunkText(text, strategy, size, overlap){
  if (strategy==='sentence'){
    const parts = text.split(/[.!?]\s+/);
    const out=[]; let cur='';
    for (const s of parts){ if (cur.length + s.length > size && cur){ out.push(cur.trim()); cur=s; } else cur += (cur?'. ':'') + s; }
    if (cur.trim()) out.push(cur.trim()); return out;
  }
  if (strategy==='semantic'){
    const parts = text.split(/\n\s*\n/);
    const out=[]; let cur='';
    for (const p of parts){ if (cur.length + p.length > size && cur){ out.push(cur.trim()); cur=p; } else cur += (cur?'\n\n':'') + p; }
    if (cur.trim()) out.push(cur.trim()); return out;
  }
  // fixed
  const out=[]; for (let i=0;i<text.length;i+=(size-overlap)) out.push(text.slice(i,i+size)); return out;
}
function snippet(t){ const s=t.replace(/\s+/g,' ').trim(); return s.slice(0,600) + (s.length>600 ? ' …' : ''); }

// Settings
$("#btn-settings").addEventListener('click', ()=>{
  document.getElementById('chunking').value = settings.chunking;
  document.getElementById('chunkSize').value = settings.chunkSize;
  document.getElementById('overlap').value = settings.overlap;
  document.getElementById('searchK').value = settings.searchK;
  openModal('modal-settings');
});
$("#btn-save-settings").addEventListener('click', ()=>{
  settings = {
    chunking: document.getElementById('chunking').value,
    chunkSize: parseInt(document.getElementById('chunkSize').value,10),
    overlap: parseInt(document.getElementById('overlap').value,10),
    searchK: parseInt(document.getElementById('searchK').value,10)
  };
  document.getElementById('kShow').textContent = String(settings.searchK);
  document.getElementById('modal-settings').classList.add('hidden');
});

// Clear chat
$("#btn-clear-chat").addEventListener('click', async ()=>{
  if (!currentRoom) return;
  await DB_CHAT.removeItem('chat:'+currentRoom.id);
  msgEl.innerHTML='';
});

// Init
(async function init(){
  let rooms = await loadRooms();
  if (!rooms.length){ await addRoom('내 문서'); rooms = await loadRooms(); }
  renderRooms(rooms);
  if (!currentRoom) await selectRoom(rooms[0].id);
  addMsg('문서를 업로드(업로드 버튼) → 인덱싱 → 질문하세요. 생성형 없이도 동작합니다.', 'bot');
})();