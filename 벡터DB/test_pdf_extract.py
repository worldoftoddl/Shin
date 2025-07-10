#!/usr/bin/env python3
"""
PDF 텍스트 추출 테스트
"""
import fitz
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """PDF에서 텍스트 추출"""
    try:
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

def main():
    pdf_path = Path("IFRS.pdf")
    
    if not pdf_path.exists():
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    print(f"PDF 파일 발견: {pdf_path}")
    print(f"파일 크기: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    text = extract_text_from_pdf(pdf_path)
    
    if text:
        print(f"텍스트 추출 성공!")
        print(f"추출된 텍스트 길이: {len(text)} 문자")
        print(f"첫 500 문자 미리보기:")
        print("-" * 50)
        print(text[:500])
        print("-" * 50)
        
        # 간단한 청크 분할 테스트
        chunks = []
        chunk_size = 500
        overlap = 100
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            if len(chunks) >= 10:  # 처음 10개 청크만 테스트
                break
        
        print(f"생성된 청크 수 (처음 10개): {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"청크 {i+1}: {chunk[:100]}...")
        
    else:
        print("텍스트 추출 실패")

if __name__ == "__main__":
    main()