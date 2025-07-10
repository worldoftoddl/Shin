# IFRS PDF 벡터 데이터베이스 처리 (Enhanced)

향상된 IFRS PDF 벡터 데이터베이스 처리 스크립트입니다.

## 주요 개선사항

### 1. 문장 단위 청크 분할
- NLTK/spaCy를 사용한 정확한 문장 분할
- 의미 있는 청크 생성으로 검색 품질 향상

### 2. 구조화된 로깅 시스템
- 상세한 처리 과정 로깅
- 오류 추적 및 디버깅 지원
- 로그 파일 자동 생성

### 3. 텍스트 전처리 및 정제
- 중복 문장 자동 제거
- 특수문자 및 공백 정리
- 문장 길이 필터링

### 4. 성능 최적화
- `convert_to_tensor=True`로 임베딩 속도 개선
- 배치 처리 지원

### 5. CLI 인자 처리
- 다양한 실행 옵션 지원
- 쿼리 전용 모드, 재구축 모드 등

### 6. 다중 PDF 처리
- 여러 PDF 파일 일괄 처리
- 확장 가능한 구조 설계

## 설치 및 설정

### 1. 의존성 설치

```bash
# 가상환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# spaCy 모델 설치 (선택사항)
python -m spacy download en_core_web_sm
```

### 2. NLTK 데이터 다운로드
스크립트 실행 시 자동으로 다운로드됩니다.

## 사용법

### 기본 사용법
```bash
# 현재 디렉토리의 모든 PDF 파일 처리
python create_ifrs_vectordb_enhanced.py

# 특정 PDF 파일 처리
python create_ifrs_vectordb_enhanced.py --pdf-files IFRS.pdf document2.pdf

# 특정 디렉토리의 PDF 파일들 처리
python create_ifrs_vectordb_enhanced.py --pdf-dir /path/to/pdf/directory
```

### 고급 옵션
```bash
# 데이터베이스 재구축
python create_ifrs_vectordb_enhanced.py --rebuild

# 사용자 정의 설정
python create_ifrs_vectordb_enhanced.py \\
    --chunk-size 600 \\
    --overlap 150 \\
    --model jhgan/ko-simcse-roberta \\
    --db-path my_custom_db

# 상세 로그 출력
python create_ifrs_vectordb_enhanced.py --verbose
```

### 쿼리 모드
```bash
# 쿼리 테스트
python create_ifrs_vectordb_enhanced.py --query "국제회계기준"

# 복합 쿼리
python create_ifrs_vectordb_enhanced.py --query "재무제표 인식 기준"
```

## CLI 옵션 상세

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--pdf-dir` | PDF 파일 디렉토리 | 현재 디렉토리 |
| `--pdf-files` | 특정 PDF 파일들 | None |
| `--db-path` | 데이터베이스 저장 경로 | chroma_db/ifrs_db |
| `--rebuild` | 기존 DB 재구축 | False |
| `--query` | 쿼리 테스트 | None |
| `--chunk-size` | 청크 크기 | 500 |
| `--overlap` | 청크 오버랩 | 100 |
| `--model` | 임베딩 모델 | jhgan/ko-simcse-roberta |
| `--verbose` | 상세 로그 | False |

## 처리 과정

1. **의존성 확인**: 필요한 라이브러리 설치 상태 확인
2. **PDF 텍스트 추출**: PyMuPDF로 텍스트 추출
3. **텍스트 전처리**: 공백, 특수문자 정리
4. **문장 분할**: NLTK/spaCy로 문장 단위 분할
5. **중복 제거**: 동일 문장 제거
6. **청크 생성**: 의미 있는 청크로 재구성
7. **임베딩 생성**: 한국어 특화 모델로 벡터화
8. **데이터베이스 저장**: ChromaDB에 영구 저장

## 출력 파일

### 생성되는 파일들
- `chroma_db/ifrs_db/`: ChromaDB 벡터 데이터베이스
- `pdf_vectordb_YYYYMMDD_HHMMSS.log`: 처리 로그 파일

### 로그 파일 내용
- 처리 과정 상세 정보
- 오류 및 경고 메시지
- 성능 통계

## 성능 향상 팁

1. **SSD 사용**: 빠른 디스크 I/O로 처리 속도 향상
2. **메모리 확보**: 대용량 PDF 처리 시 충분한 RAM 필요
3. **배치 처리**: 여러 파일을 한 번에 처리하여 효율성 증대
4. **GPU 사용**: CUDA 지원 환경에서 더 빠른 임베딩 생성

## 문제해결

### 일반적인 문제들

1. **메모리 부족**: 청크 크기를 줄이거나 배치 크기 조정
2. **모델 다운로드 실패**: 인터넷 연결 확인
3. **PDF 읽기 오류**: 손상된 PDF 파일 확인
4. **임베딩 오류**: GPU 메모리 부족 시 CPU 모드로 전환

### 로그 확인
```bash
# 최신 로그 파일 확인
ls -la pdf_vectordb_*.log | tail -1

# 오류 로그 필터링
grep ERROR pdf_vectordb_*.log
```

## 예제 스크립트

### 배치 처리 예제
```bash
#!/bin/bash
# 여러 PDF 디렉토리 처리

for dir in documents1 documents2 documents3; do
    echo "Processing $dir..."
    python create_ifrs_vectordb_enhanced.py \\
        --pdf-dir "$dir" \\
        --db-path "chroma_db/${dir}_db" \\
        --verbose
done
```

### 쿼리 테스트 예제
```bash
#!/bin/bash
# 다양한 쿼리 테스트

queries=(
    "국제회계기준"
    "재무제표 인식"
    "자산 측정"
    "수익 인식"
)

for query in "${queries[@]}"; do
    echo "Testing query: $query"
    python create_ifrs_vectordb_enhanced.py --query "$query"
    echo "---"
done
```

## 확장 가능성

- **다국어 지원**: 다양한 언어별 임베딩 모델 지원
- **웹 인터페이스**: Flask/FastAPI로 웹 서비스 확장
- **실시간 처리**: 파일 변경 감지 및 자동 업데이트
- **분산 처리**: 대용량 문서 처리를 위한 분산 시스템 구축