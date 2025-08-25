# RAG Web Chat — CDN-FREE (auto vendoring)
이 패키지는 **GitHub Actions**가 `@mlc-ai/web-llm`을 npm에서 받아 `./vendor/web-llm/dist`에 복사해두도록 구성되어 있습니다.
페이지는 먼저 **로컬 vendored 경로**(same origin)에서 WebLLM을 로드하므로, 회사망에서 CDN이 막혀 있어도 동작합니다.

## 사용 방법
1. ZIP을 레포 루트(또는 docs/)에 **그대로 업로드** (기존 파일 교체 포함)
2. 레포 **Settings → Pages**에서 배포 폴더(루트 또는 docs) 확인
3. 레포 **Settings → Actions → General**에서 **Workflow permissions**를 `Read and write permissions`로 설정
4. 레포의 **Actions 탭 → "Vendor WebLLM dist" 워크플로우** 실행 (Run workflow)
   - 실행 후 커밋이 생성되며, `vendor/web-llm/dist/`에 파일이 채워집니다.
5. 배포 완료 후 접속 → 이제 WebLLM이 **./vendor/web-llm/dist/** 경로에서 로드됩니다.

## 참고
- 임베딩/검색은 CDN을 쓰지 않거나(FAST) jsDelivr에서만 한 번 받습니다.
- 필요 시 Transformers.js와 pdf.js도 동일 방식으로 vendoring하도록 워크플로우를 확장할 수 있습니다.
