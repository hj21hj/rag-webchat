# RAG Web Chat — Drop‑in Replacement (All Fixes Included)
이 패키지 파일 4개(`index.html`, `script.js`, `style.css`, `README.md`)를 **레포 루트**(또는 `docs/`)에 그대로 교체 업로드하면 끝입니다.

## 포함된 수정 사항
- **임베딩 로딩 오류 해결**: Transformers.js를 **동적 import**로 로드(전역 `window.transformers` 의존 제거)
- **PDF 안정화**: pdf.js **UMD + workerSrc** 고정 (GitHub Pages에서 안정적으로 동작)
- **빠른 모드 추가**: `FAST / HYBRID / FULL` 선택
  - FAST: BM25만 (즉시 사용)
  - HYBRID: BM25 선발 → 상위 M만 임베딩 (기본 120)
  - FULL: 전체 임베딩 (정확도↑, 로딩 느림)
- **임베딩 모델 경량화**: `Xenova/paraphrase-MiniLM-L3-v2` (기본)
- IndexedDB 유지 확률을 높이기 위한 `storage.persist()` 요청
- 진행상태/오류 표시 강화

## 배포
- **루트 배포**: 레포 루트에 4개 파일 업로드 → Settings > Pages: Branch=`main`, Folder=`/(root)`
- **docs 배포**: `docs/` 폴더에 4개 파일 업로드 → Settings > Pages: Branch=`main`, Folder=`/docs`

## 사용
1) 페이지 접속 → 문서(.txt/.md/.pdf) 드래그앤드롭
2) 상단 **모드** 선택 (`HYBRID` 추천)
3) **[인덱싱]** → 질문 → **[질의하기]**
