"""① 문서 파싱: PDF에서 텍스트·표·이미지를 분리 추출합니다."""

import os

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(path, fname, figures_dir=None, languages=None):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.

    path       : PDF가 있는 폴더 경로 (끝에 / 포함)
    fname      : PDF 파일 이름
    figures_dir: 추출 이미지를 저장할 폴더 (기본값: path + "figures")
    languages  : OCR 언어 목록. 한국어 PDF면 ["kor", "eng"] 권장
                 (Tesseract 한국어 데이터 필요: brew install tesseract-lang)
    """
    figures_dir = figures_dir or os.path.join(path, "figures")
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                       # 레이아웃 모델로 정밀 추출 (표·이미지 정확도)
        extract_images_in_pdf=True,              # PDF에서 이미지를 추출
        extract_image_block_output_dir=figures_dir,  # 추출 이미지 저장 위치
        infer_table_structure=True,              # 테이블 구조를 추론
        languages=languages,                     # None 이면 영어로 OCR
        chunking_strategy="by_title",            # 제목 기준으로 블록 분할
        max_characters=4000,                     # 블록 최대 4000자
        new_after_n_chars=3800,                  # 3800자 이후 새 블록
        combine_text_under_n_chars=2000,         # 2000자 이하 블록은 합침
    )


def categorize_elements(raw_pdf_elements):
    """추출한 요소를 텍스트와 테이블로 분류합니다."""
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """
    텍스트 블록을 하나로 합친 뒤 토큰 기준으로 다시 나눕니다.
    겹침(chunk_overlap)은 블록 경계에서 문장이 잘려 뜻이 사라지는 것을 막습니다.
    """
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(" ".join(texts))
