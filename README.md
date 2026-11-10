# Multimodal RAG

PDF 속 **텍스트·표·이미지(그래프)** 를 모두 검색 대상으로 삼는 멀티모달 RAG 실습 저장소입니다.

## 문제 정의

일반 RAG는 텍스트 청크만 검색하기 때문에, 수치가 **그래프나 표에만** 있는 문서에서는 "자료에 없다"고 답합니다.
이 프로젝트는 표와 이미지도 요약해 검색 대상에 넣고, 답변을 만들 때는 **요약이 아닌 원본(표·이미지)** 을 LLM에 넘겨 그래프에만 있는 정보까지 답하게 합니다.

- **검색에는 요약, 답변에는 원본**: 이미지는 그 자체로 임베딩·검색되지 않아 텍스트 요약이 필요합니다. 하지만 요약에는 정확한 수치가 빠지므로, 답변 생성에는 같은 `doc_id`로 연결된 원본을 씁니다.


## 폴더 구조

```
multimodal-rag/
├── .env.example            # 필요한 환경변수 안내 (실제 키는 .env, Git 제외)
├── requirements.txt        # 의존성 목록
├── requirements.lock.txt   # 동작 확인한 버전 고정
├── data/                   # 원본 PDF·추출 결과 (Git 제외)
├── notebooks/
│   └── Multi_modal_RAG.ipynb             # 설치 검증 + 데이터 추출
└── src/
    ├── extract.py          # PDF 파싱·분류·청크 분할
    ├── summarize.py        # 텍스트·표·이미지 요약
    └── retriever.py        # 다중 벡터 검색기·최종 체인
```

## 실행 방법

사전 설치 (파이썬 패키지가 아닌 OS 프로그램):

```bash
brew install poppler tesseract    # 윈도우: 설치 후 Path 등록
pdftoppm -h && tesseract -v       # 둘 다 출력되면 성공
```

```bash
git clone https://github.com/Navi-Up/multimodal-rag.git
cd multimodal-rag

# 가상환경 생성 및 활성화 (Python 3.11 / 3.12)
python -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt     # 또는 requirements.lock.txt

# 커널 등록
python -m ipykernel install --user --name multimodal-rag

# .env 작성
cp .env.example .env                # 윈도우: copy .env.example .env
# .env 파일을 열어 실제 API 키를 입력
```

PDF를 `data/`에 넣고 노트북을 `Multi_modal_RAG` → `02` → `03` 순서로 실행합니다.

## 보안

- `.env`(API 키), 원본 PDF, 추출 이미지(`figures/`), 중간 결과(`*.pkl`), 벡터 DB, 도구 로그(`.bkit/`)는 `.gitignore`로 제외합니다.
- 필요한 환경변수는 `.env.example`에 키 없이 안내합니다.
