#!/usr/bin/env python3
"""
IFRS PDF를 벡터 데이터베이스로 변환하는 개선된 스크립트
- 문장 단위 청크 분할 (NLTK/spaCy)
- 구조화된 로깅
- 텍스트 전처리 및 중복 제거
- CLI 인자 처리
- 다중 PDF 파일 처리 지원
"""

import os
import sys
import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProcessingConfig:
    """처리 설정 데이터클래스"""
    chunk_size: int = 500
    overlap: int = 100
    min_sentence_length: int = 10
    max_sentence_length: int = 1000
    embedding_model: str = 'jhgan/ko-simcse-roberta'
    collection_name: str = 'ifrs_collection'
    db_path: str = 'chroma_db'


class PDFVectorDBProcessor:
    """PDF 벡터 데이터베이스 처리 클래스"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.sentence_tokenizer = None
        self.model = None
        self.client = None
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('PDFVectorDB')
        logger.setLevel(logging.INFO)
        
        # 로그 파일 핸들러
        log_file = f'pdf_vectordb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def check_dependencies(self) -> bool:
        """필요한 라이브러리가 설치되어 있는지 확인"""
        required_modules = {
            'fitz': 'PyMuPDF',
            'sentence_transformers': 'sentence-transformers', 
            'chromadb': 'chromadb',
            'nltk': 'nltk',
            'spacy': 'spacy'
        }
        
        missing = []
        for module, package in required_modules.items():
            try:
                __import__(module)
                self.logger.info(f"✓ {package} 사용 가능")
            except ImportError:
                missing.append(package)
                self.logger.error(f"✗ {package} 설치 필요")
        
        if missing:
            self.logger.error(f"다음 패키지를 설치해주세요: {' '.join(missing)}")
            return False
        return True
    
    def _initialize_nlp_tools(self):
        """NLP 도구 초기화"""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # NLTK 데이터 다운로드
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                self.logger.info("NLTK punkt 데이터 다운로드 중...")
                nltk.download('punkt')
            
            self.sentence_tokenizer = sent_tokenize
            self.logger.info("NLTK 문장 토크나이저 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"NLTK 초기화 실패: {e}")
            # 백업: spaCy 사용
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                self.sentence_tokenizer = lambda text: [sent.text for sent in nlp(text).sents]
                self.logger.info("spaCy 문장 토크나이저 초기화 완료")
            except Exception as e2:
                self.logger.error(f"spaCy 초기화도 실패: {e2}")
                # 최후 수단: 정규식 기반 분할
                self.sentence_tokenizer = self._regex_sentence_split
                self.logger.warning("정규식 기반 문장 분할 사용")
    
    def _regex_sentence_split(self, text: str) -> List[str]:
        """정규식 기반 문장 분할 (백업용)"""
        sentences = re.split(r'[.!?]+\\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""
        
        # 1. 기본 정제
        text = re.sub(r'\\s+', ' ', text)  # 여러 공백을 하나로
        text = re.sub(r'[\\x00-\\x1f\\x7f-\\x9f]', '', text)  # 제어 문자 제거
        text = text.strip()
        
        # 2. 특수 문자 정리 (필요시)
        text = re.sub(r'[•▪▫◦‣⁃]', '- ', text)  # 불릿 포인트 통일
        text = re.sub(r'[""''‚„]', '"', text)  # 따옴표 통일
        
        self.logger.debug(f"텍스트 전처리 완료: {len(text)} 문자")
        return text
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> Optional[str]:
        """PDF에서 텍스트 추출"""
        try:
            import fitz
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                self.logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
                return None
            
            self.logger.info(f"PDF 텍스트 추출 시작: {pdf_path.name}")
            
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text + "\\n"
                
                if page_num % 10 == 0:
                    self.logger.debug(f"페이지 {page_num + 1}/{len(doc)} 처리 중...")
            
            doc.close()
            
            # 텍스트 전처리
            text = self.preprocess_text(text)
            
            self.logger.info(f"텍스트 추출 완료: {len(text)} 문자, {len(doc)} 페이지")
            return text
            
        except Exception as e:
            self.logger.error(f"PDF 텍스트 추출 오류: {e}")
            return None
    
    def split_text_into_sentence_chunks(self, text: str) -> List[str]:
        """문장 단위로 청크 분할"""
        if not self.sentence_tokenizer:
            self._initialize_nlp_tools()
        
        try:
            # 1. 문장 분할
            sentences = self.sentence_tokenizer(text)
            self.logger.info(f"문장 분할 완료: {len(sentences)} 문장")
            
            # 2. 문장 필터링
            filtered_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if (self.config.min_sentence_length <= len(sentence) <= self.config.max_sentence_length):
                    filtered_sentences.append(sentence)
            
            self.logger.info(f"문장 필터링 완료: {len(filtered_sentences)} 문장")
            
            # 3. 중복 문장 제거
            unique_sentences = list(dict.fromkeys(filtered_sentences))
            removed_count = len(filtered_sentences) - len(unique_sentences)
            if removed_count > 0:
                self.logger.info(f"중복 문장 {removed_count}개 제거")
            
            # 4. 청크 생성
            chunks = []
            current_chunk = ""
            
            for sentence in unique_sentences:
                # 현재 청크에 문장을 추가했을 때 크기 확인
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential_chunk) <= self.config.chunk_size:
                    current_chunk = potential_chunk
                else:
                    # 현재 청크 저장
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # 새 청크 시작
                    current_chunk = sentence
            
            # 마지막 청크 추가
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            self.logger.info(f"청크 생성 완료: {len(chunks)} 청크")
            return chunks
            
        except Exception as e:
            self.logger.error(f"문장 청크 분할 오류: {e}")
            # 백업: 기본 청크 분할
            return self._basic_chunk_split(text)
    
    def _basic_chunk_split(self, text: str) -> List[str]:
        """기본 청크 분할 (백업용)"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.config.overlap
        return chunks
    
    def initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info(f"임베딩 모델 로드 중: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
            self.logger.info("임베딩 모델 로드 완료")
            
        except Exception as e:
            self.logger.error(f"임베딩 모델 로드 오류: {e}")
            raise
    
    def initialize_chromadb(self, db_path: Path):
        """ChromaDB 초기화"""
        try:
            import chromadb
            
            db_path.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=str(db_path))
            self.logger.info(f"ChromaDB 초기화 완료: {db_path}")
            
        except Exception as e:
            self.logger.error(f"ChromaDB 초기화 오류: {e}")
            raise
    
    def process_pdf_files(self, pdf_paths: List[Path], db_path: Path, rebuild: bool = False):
        """여러 PDF 파일 처리"""
        self.logger.info(f"PDF 파일 처리 시작: {len(pdf_paths)} 파일")
        
        # 의존성 확인
        if not self.check_dependencies():
            return False
        
        # 모델 및 DB 초기화
        self.initialize_embedding_model()
        self.initialize_chromadb(db_path)
        
        # 컬렉션 설정
        try:
            if rebuild:
                try:
                    self.client.delete_collection(name=self.config.collection_name)
                    self.logger.info("기존 컬렉션 삭제 완료")
                except:
                    pass
            
            collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
        except Exception as e:
            # 컬렉션이 이미 존재하는 경우
            collection = self.client.get_collection(name=self.config.collection_name)
            self.logger.info("기존 컬렉션 사용")
        
        # 각 PDF 파일 처리
        total_chunks = 0
        
        for pdf_path in pdf_paths:
            try:
                self.logger.info(f"처리 중: {pdf_path.name}")
                
                # 텍스트 추출
                text = self.extract_text_from_pdf(pdf_path)
                if not text:
                    continue
                
                # 청크 분할
                chunks = self.split_text_into_sentence_chunks(text)
                if not chunks:
                    continue
                
                # 임베딩 생성
                self.logger.info("임베딩 생성 중...")
                embeddings = self.model.encode(
                    chunks, 
                    convert_to_tensor=True,
                    show_progress_bar=True
                )
                
                # 메타데이터 생성
                metadatas = [
                    {
                        "source": pdf_path.stem,
                        "file_path": str(pdf_path),
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    } 
                    for i in range(len(chunks))
                ]
                
                # ID 생성
                ids = [f"{pdf_path.stem}_chunk_{i}" for i in range(len(chunks))]
                
                # ChromaDB에 저장
                collection.add(
                    embeddings=embeddings.cpu().numpy().tolist(),
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_chunks += len(chunks)
                self.logger.info(f"✓ {pdf_path.name}: {len(chunks)} 청크 저장 완료")
                
            except Exception as e:
                self.logger.error(f"파일 처리 오류 {pdf_path.name}: {e}")
                continue
        
        self.logger.info(f"모든 파일 처리 완료: 총 {total_chunks} 청크")
        return True
    
    def query_database(self, query: str, n_results: int = 5) -> Dict:
        """데이터베이스 쿼리"""
        try:
            if not self.client:
                db_path = Path(self.config.db_path) / "ifrs_db"
                self.initialize_chromadb(db_path)
            
            if not self.model:
                self.initialize_embedding_model()
            
            collection = self.client.get_collection(name=self.config.collection_name)
            
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            
            # 검색 실행
            results = collection.query(
                query_embeddings=query_embedding.cpu().numpy().tolist(),
                n_results=n_results
            )
            
            self.logger.info(f"쿼리 실행 완료: '{query}' - {len(results['documents'][0])} 결과")
            return results
            
        except Exception as e:
            self.logger.error(f"쿼리 실행 오류: {e}")
            return {}


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='IFRS PDF 벡터 데이터베이스 처리')
    parser.add_argument('--pdf-dir', type=str, default='.', 
                       help='PDF 파일이 있는 디렉토리 (기본값: 현재 디렉토리)')
    parser.add_argument('--pdf-files', type=str, nargs='+', 
                       help='처리할 특정 PDF 파일들')
    parser.add_argument('--db-path', type=str, default='chroma_db/ifrs_db',
                       help='ChromaDB 저장 경로')
    parser.add_argument('--rebuild', action='store_true',
                       help='기존 데이터베이스 재구축')
    parser.add_argument('--query', type=str,
                       help='쿼리 테스트 실행')
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='청크 크기 (기본값: 500)')
    parser.add_argument('--overlap', type=int, default=100,
                       help='청크 오버랩 (기본값: 100)')
    parser.add_argument('--model', type=str, default='jhgan/ko-simcse-roberta',
                       help='임베딩 모델')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = ProcessingConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.model,
        db_path=args.db_path
    )
    
    # 프로세서 생성
    processor = PDFVectorDBProcessor(config)
    
    # 로그 레벨 설정
    if args.verbose:
        processor.logger.setLevel(logging.DEBUG)
    
    # 쿼리 모드
    if args.query:
        results = processor.query_database(args.query)
        if results and 'documents' in results:
            print(f"\\n쿼리: '{args.query}'")
            print("=" * 50)
            for i, doc in enumerate(results['documents'][0]):
                print(f"{i+1}. {doc[:200]}...")
                print("-" * 30)
        return
    
    # PDF 파일 수집
    pdf_paths = []
    
    if args.pdf_files:
        # 특정 파일들
        for pdf_file in args.pdf_files:
            path = Path(pdf_file)
            if path.exists() and path.suffix.lower() == '.pdf':
                pdf_paths.append(path)
    else:
        # 디렉토리에서 PDF 파일 검색
        pdf_dir = Path(args.pdf_dir)
        pdf_paths = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_paths:
        print("처리할 PDF 파일을 찾을 수 없습니다.")
        return
    
    # 데이터베이스 처리
    db_path = Path(args.db_path)
    success = processor.process_pdf_files(pdf_paths, db_path, args.rebuild)
    
    if success:
        print(f"\\n✓ 벡터 데이터베이스 생성 완료: {db_path}")
        print(f"✓ 처리된 파일: {len(pdf_paths)} 개")
        
        # 간단한 테스트 쿼리
        if not args.query:
            test_results = processor.query_database("국제회계기준", n_results=3)
            if test_results:
                print("\\n📋 테스트 쿼리 결과:")
                for i, doc in enumerate(test_results['documents'][0]):
                    print(f"{i+1}. {doc[:100]}...")
    else:
        print("\\n✗ 벡터 데이터베이스 생성 실패")


if __name__ == "__main__":
    main()