"""
make_scan.py — 텍스트 PDF를 스캔본처럼 바꿉니다.
설치: pip install pymupdf
실행: python make_scan.py
※ PyMuPDF는 Poppler 없이 동작하므로 수강생도 사용 가능합니다.
"""
import fitz  # PyMuPDF

SRC = "./data/sample_text.pdf"      # 원본 (텍스트 PDF)
DST = "./data/sample_scan.pdf"      # 결과 (스캔본)
DPI = 150                           # 낮출수록 OCR 난이도 상승

src_doc = fitz.open(SRC)
dst_doc = fitz.open()

for page in src_doc:
    # 1) 페이지를 이미지로 렌더링합니다 → 텍스트 레이어가 사라집니다
    pix = page.get_pixmap(dpi=DPI)

    # 2) 이미지를 새 PDF 페이지에 그대로 붙입니다
    new_page = dst_doc.new_page(width=pix.width, height=pix.height)
    new_page.insert_image(new_page.rect, pixmap=pix)

dst_doc.save(DST)
print(f"스캔본 생성 완료: {DST}")