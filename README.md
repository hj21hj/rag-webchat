# RAG Web Chat (GitHub Pages, No Server/No Tokens)

- 100% **클라이언트 사이드**: OpenAI 토큰/백엔드 없이 브라우저에서 바로 동작
- **GitHub Pages**에 그대로 업로드하면 누구나 URL로 접근 가능
- **WebGPU**(+WebAssembly)로 모델 실행: 
  - **생성(LLM)**: WebLLM
  - **임베딩**: Transformers.js (all-MiniLM-L6-v2)

## 배포 방법 (GitHub Pages)

1. GitHub 새 repo 생성 (예: `rag-webchat`)
2. 이 폴더의 파일들을 전부 업로드 (`index.html`, `script.js`, `style.css`, `README.md` 등)
3. **Settings → Pages → Build and deployment**
   - Source: **Deploy from a branch**
   - Branch: `main` / `/ (root)` 선택
4. 배포 후 Pages URL 접속 (예: `https://사용자명.github.io/rag-webchat/`)

## 사용법
- 브라우저: **Chrome/Edge (WebGPU 활성)** 권장
- 페이지 접속 → 상단에 **문서 드래그앤드롭**(.txt/.md/.pdf) → **인덱싱** → **질의**
- 인덱스/임베딩은 **IndexedDB**에 저장되어 다음 접속 시에도 유지(같은 브라우저/도메인 기준)

## 모델/성능
- 기본 생성 모델: 작은 WebLLM 모델(브라우저 로딩 수십~수백MB)
- 기본 임베딩 모델: `Xenova/all-MiniLM-L6-v2` (int8)
- 성능/속도는 PC GPU/브라우저/모델 크기에 따라 달라짐
- 필요시 `script.js`의 모델 ID를 더 큰 한국어 모델로 변경 가능 (로딩 시간↑)

## 제한
- 스캔 PDF는 텍스트 추출 불가
- 모바일 브라우저는 WebGPU 미지원일 수 있음
- 회사망에서 외부 CDN/Hugging Face 차단 시 모델 로드가 안 될 수 있음
