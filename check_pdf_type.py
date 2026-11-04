"""
check_pdf_type.py — PDF가 어떤 유형인지 판별합니다.
설치: pip install pymupdf
"""
import fitz

for path in ["./data/sample_text.pdf",
             "./data/sample_table.pdf",
             "./data/sample_scan.pdf"]:
    doc = fitz.open(path)
    # 첫 3페이지에서 추출되는 글자 수를 셉니다
    chars = sum(len(doc[i].get_text()) for i in range(min(3, len(doc))))

    if chars < 50:
        verdict = "스캔본 (OCR 필요)"
    else:
        verdict = "텍스트 PDF"

    print(f"{path}")
    print(f"  추출 글자 수: {chars} → {verdict}\n")