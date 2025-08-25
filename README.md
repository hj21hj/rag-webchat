# RAG Web Chat — Vendoring (dual path)
- 이 패키지는 Actions가 `@mlc-ai/web-llm`을 받아 **루트(`vendor/...`)와 `docs/vendor/...`**에 동시에 복사합니다.
- Pages가 루트든 `docs`든 상관없이 `./vendor/web-llm/dist/` 경로가 항상 살아있게 됩니다.

## 사용 순서
1) ZIP 풀기 → 레포(루트 또는 `docs/`)에 **그대로 업로드/교체**  
2) Settings → Actions → General → **Read and write permissions** 체크  
3) Actions 탭 → **Vendor WebLLM dist (dual path)** → **Run workflow** 실행  
4) 완료 후 접속 → 개발자도구 Network에서 `.../vendor/web-llm/dist/index.js`가 200으로 떨어지면 정상

문제시 레포 구조(루트 or docs)만 알려주세요. 딱 맞춘 버전으로 다시 만들어 드립니다.
