#PDF에서 표, 텍스트, 이미지 같은 요소를 추출하기 위해 사용되는 기능
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.
    path: 파일 경로 (추출된 이미지 파일이 저장될 위치)
    fname: 파일 이름
    """
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                # 레이아웃 분석 모델로 정밀 추출 (교재에 없음! 필수 추가)
        extract_images_in_pdf=True,       # PDF에서 이미지를 추출
        infer_table_structure=True,       # 테이블 구조를 추론
        chunking_strategy="by_title",     # 타이틀을 기준으로 텍스트를 블록으로 분할
        max_characters=4000,              # 최대 4000자로 텍스트 블록을 제한
        new_after_n_chars=3800,           # 3800자 이후에 새로운 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하의 텍스트는 결합
        image_output_dir_path=path,       # 이미지가 저장될 경로 설정
    )


def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.
    raw_pdf_elements: unstructured.documents.elements 리스트
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))   # 테이블 요소를 저장
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))    # 텍스트 요소를 저장
    return texts, tables
