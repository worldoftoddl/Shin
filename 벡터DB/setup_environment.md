# IFRS 벡터 데이터베이스 설정 가이드

## 필요한 시스템 패키지 설치

```bash
# Ubuntu/Debian 시스템에서
sudo apt update
sudo apt install python3-pip python3-venv

# 또는 WSL에서
sudo apt install python3-pip python3-dev python3-venv
```

## 가상 환경 설정

```bash
# 가상 환경 생성
python3 -m venv venv

# 가상 환경 활성화
source venv/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 스크립트 실행

```bash
# 가상 환경이 활성화된 상태에서
python create_ifrs_vectordb.py
```

## 문제 해결

만약 가상 환경 생성이 실패하면:
1. `sudo apt install python3.12-venv` 실행
2. 또는 `--break-system-packages` 옵션으로 시스템 패키지 직접 설치

## 생성되는 파일 구조

```
first_project/
├── IFRS.pdf
├── create_ifrs_vectordb.py
├── requirements.txt
└── chroma_db/
    └── ifrs_db/
        └── [ChromaDB 벡터 데이터베이스 파일들]
```