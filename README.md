# RAG Web Chat — FAST/HYBRID/FULL
- **FAST**: BM25(elasticlunr)만 사용 → 임베딩 모델 로딩 없음(즉시)
- **HYBRID (추천)**: BM25로 상위 M개 선발 → 그때만 임베딩 모델 로딩/계산 → 빠른 체감
- **FULL**: 모든 청크 임베딩(초기 오래 걸리지만 검색 정확도↑)
- 더 작은 임베딩 모델 `Xenova/paraphrase-MiniLM-L3-v2`로 기본 변경
- 브라우저 저장소 **persist** 요청으로 캐시 유지 확률 상승
- pdf.js UMD + worker 경로 고정(페이지스 호환)