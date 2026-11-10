from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM

def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 및 표 데이터를 요약하여 검색에 활용할 수 있는 요약본을 생성합니다.
    texts: 텍스트 리스트
    tables: 표 리스트
    summarize_texts: 텍스트 요약을 활성화할지 여부를 결정하는 불리언 값
    """

    # Prompt 한국어 버전
    prompt_text_kor = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
    이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
    주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
    요약할 표 또는 텍스트: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text_kor)

    # 모델: GPT-4o-mini or Llama3.1 모델 사용
    # llamaModel = OllamaLLM(model="llama3.1:8b")
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # 요약 체인 = 프롬프트 + LLM + 출력 파서
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    # 텍스트 요약을 활성화한 경우
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    # 텍스트 요약을 사용하지 않는 경우
    elif texts:
        text_summaries = texts

    # 테이블 데이터 요약
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries


import base64
import os
import requests
import json
from langchain_core.messages import HumanMessage

def encode_image(image_path):
    """이미지를 base64 문자열로 변환"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """이미지 요약 생성"""
    # ❌ model="gpt-4-vision-preview" → 지원 종료(deprecated)
    # ✅ gpt-4o-mini 가 비전 기능을 포함하며 비용이 훨씬 저렴합니다
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        # Base64 문자열 앞에 'data:image/jpeg;base64,' 접두사가 꼭 필요합니다
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def image_summarize_llava(img_base64, prompt):
    """LLaVA를 사용하여 이미지 요약 생성"""
    payload = {
        "model": "llava:7b",
        "prompt": prompt,
        "images": [img_base64],
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    print("Status Code: ", response.status_code)

    if response.status_code == 200:
        try:
            # 응답을 줄 단위로 처리
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_line = json.loads(line)
                    if "response" in json_line:
                        full_response += json_line["response"]
            print(full_response)
            return full_response
        except json.JSONDecodeError as e:
            return f"JSON 파싱 오류: {str(e)}\n응답 내용: {response.text}"
    else:
        return f"Error: {response.status_code}, {response.text}"


def generate_img_summaries(path):
    """
    이미지의 요약과 Base64 인코딩 문자열을 생성
    path: Unstructured에 의해 추출된 .jpg 파일의 경로
    """
    img_base64_list = []   # Base64로 인코딩된 이미지를 저장할 리스트
    image_summaries = []   # 이미지 요약을 저장할 리스트

    # Prompt_kor 한국어
    prompt_kor = """You are an assistant tasked with summarizing images for retrieval.
    These summaries will be embedded and used to retrieve the raw image.
    Provide a concise summary of the image that is well optimized for retrieval.
    The summary should be written in Korean (Hangul)."""

    # Prompt 영어
    prompt = """You are an assistant tasked with summarizing images for retrieval.
    These summaries will be embedded and used to retrieve the raw image.
    Provide a concise summary of the image that is well optimized for retrieval."""

    # 주어진 경로에서 파일 목록을 가져와 정렬한 후, 각 파일을 처리합니다.
    for img_file in sorted(os.listdir(path)):
        # 파일이 .jpg, .png, .jpeg 확장자 중 하나일 경우에만 처리합니다.
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt_kor))

    return img_base64_list, image_summaries
