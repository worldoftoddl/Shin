#!/usr/bin/env python3
"""
IFRS PDF를 벡터 데이터베이스로 변환하는 스크립트
"""
import os
import sys
from pathlib import Path

def check_dependencies():
    """필요한 라이브러리가 설치되어 있는지 확인"""
    required_modules = {
        'fitz': 'PyMuPDF',
        'sentence_transformers': 'sentence-transformers', 
        'chromadb': 'chromadb'
    }
    
    missing = []
    for module, package in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {package} 사용 가능")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} 설치 필요")
    
    if missing:
        print(f"\n다음 패키지를 설치해주세요:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def extract_text_from_pdf(pdf_path):
    """PDF에서 텍스트 추출"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"PDF 텍스트 추출 오류: {e}")
        return None

def split_text_into_chunks(text, chunk_size=500, overlap=100):
    """텍스트를 청크로 분할"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def create_vector_db():
    """벡터 데이터베이스 생성"""
    print("IFRS PDF 벡터 데이터베이스 생성 시작...")
    
    # 1. 의존성 확인
    if not check_dependencies():
        return False
    
    # 2. 필요한 모듈 import
    try:
        import fitz
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings
    except ImportError as e:
        print(f"모듈 import 오류: {e}")
        return False
    
    # 3. 파일 경로 설정
    pdf_path = Path(__file__).parent / "IFRS.pdf"
    chroma_db_path = Path(__file__).parent / "chroma_db" / "ifrs_db"
    
    if not pdf_path.exists():
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return False
    
    # 4. PDF 텍스트 추출
    print("PDF에서 텍스트 추출 중...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("PDF에서 텍스트를 추출할 수 없습니다.")
        return False
    
    print(f"추출된 텍스트 길이: {len(text)} 문자")
    
    # 5. 텍스트 청크 분할
    print("텍스트를 청크로 분할 중...")
    chunks = split_text_into_chunks(text, chunk_size=500, overlap=100)
    print(f"생성된 청크 수: {len(chunks)}")
    
    # 6. 임베딩 모델 로드
    print("임베딩 모델 로드 중...")
    try:
        model = SentenceTransformer('jhgan/ko-simcse-roberta')
        print("임베딩 모델 로드 완료")
    except Exception as e:
        print(f"임베딩 모델 로드 오류: {e}")
        return False
    
    # 7. ChromaDB 설정
    print("ChromaDB 설정 중...")
    chroma_db_path.mkdir(parents=True, exist_ok=True)
    
    try:
        client = chromadb.PersistentClient(path=str(chroma_db_path))
        
        # 기존 컬렉션이 있으면 삭제
        try:
            client.delete_collection(name="ifrs_collection")
        except:
            pass
        
        collection = client.create_collection(
            name="ifrs_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("ChromaDB 컬렉션 생성 완료")
    except Exception as e:
        print(f"ChromaDB 설정 오류: {e}")
        return False
    
    # 8. 임베딩 생성 및 저장
    print("임베딩 생성 및 저장 중...")
    try:
        embeddings = model.encode(chunks)
        
        # 메타데이터 생성
        metadatas = [{"source": "IFRS", "chunk_id": i} for i in range(len(chunks))]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ {len(chunks)}개 청크가 성공적으로 저장되었습니다.")
        print(f"✓ 벡터 DB 저장 위치: {chroma_db_path}")
        
        # 9. 저장 확인
        count = collection.count()
        print(f"✓ 저장된 문서 수: {count}")
        
        return True
        
    except Exception as e:
        print(f"임베딩 생성/저장 오류: {e}")
        return False

def query_test():
    """간단한 쿼리 테스트"""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        chroma_db_path = Path(__file__).parent / "chroma_db" / "ifrs_db"
        
        client = chromadb.PersistentClient(path=str(chroma_db_path))
        collection = client.get_collection(name="ifrs_collection")
        
        model = SentenceTransformer('jhgan/ko-simcse-roberta')
        
        # 테스트 쿼리
        test_query = "국제회계기준"
        query_embedding = model.encode([test_query])
        
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )
        
        print(f"\n테스트 쿼리: '{test_query}'")
        print("검색 결과:")
        for i, doc in enumerate(results['documents'][0]):
            print(f"{i+1}. {doc[:100]}...")
            
        return True
        
    except Exception as e:
        print(f"쿼리 테스트 오류: {e}")
        return False

if __name__ == "__main__":
    # 벡터 DB 생성
    if create_vector_db():
        print("\n벡터 데이터베이스 생성 완료!")
        
        # 간단한 테스트
        print("\n쿼리 테스트 실행...")
        query_test()
    else:
        print("\n벡터 데이터베이스 생성 실패!")
        sys.exit(1)