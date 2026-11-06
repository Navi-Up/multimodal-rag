"""② 저장·검색과 ③ 답변 생성: 다중 벡터 검색기와 최종 멀티모달 체인."""

import re
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ID_KEY = "doc_id"
BASE64_PATTERN = "^[A-Za-z0-9+/]{100,}={0,2}$"


def create_retriever(collection_name="multimodal_rag"):
    """요약은 벡터 DB(색인 카드), 원본은 docstore(서가)에 두는 검색기를 만듭니다."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        id_key=ID_KEY,
    )


def add_documents(retriever, doc_summaries, doc_contents):
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)            # 요약 → 검색용
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))    # 원본 → 답변용


def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트로 나눕니다."""
    b64_images, texts = [], []
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        if re.match(BASE64_PATTERN, content):
            b64_images.append(content)
        else:
            texts.append(content)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """검색된 원본(이미지 + 텍스트)을 멀티모달 LLM 입력으로 조립합니다."""
    messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
        for image in data_dict["context"]["images"]
    ]
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append(
        {
            "type": "text",
            "text": (
                "자료를 근거로 질문에 한국어로 답하세요\n"
                "그래프로 나타난 수치는 이미지를 분석하여 대략적인 값을 요약에 포함하세요\n"
                "자료에 없는 내용은 모른다고 답하세요.\n\n"
                f"질문: {data_dict['question']}\n\n"
                f"참고 텍스트 :\n {formatted_texts}"
            ),
        }
    )
    return [HumanMessage(content=messages)]


def build_multimodal_chain(retriever, model_name="gpt-4o-mini"):
    """질문 → 요약으로 검색 → 원본으로 답변 생성."""
    model = ChatOpenAI(temperature=0, model=model_name, max_tokens=1024)
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
