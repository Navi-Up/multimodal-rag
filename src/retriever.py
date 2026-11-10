import uuid
# 교안: from langchain.retrievers.multi_vector import MultiVectorRetriever
#       from langchain.storage import InMemoryStore
# langchain 1.x 에서는 두 경로가 langchain-classic 으로 분리되어 위 경로로는 ModuleNotFoundError 가 납니다.
# (requirements.txt 의 "langchain-classic>=1.0  # MultiVectorRetriever. langchain 1.x에서 분리됨 (필수)")
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 요약문이 들어갈 벡터 DB (= 색인 카드 서랍)
vectorstore = Chroma(
    collection_name="multimodal_rag",
    embedding_function=OpenAIEmbeddings(),
)

# 원본이 들어갈 저장소 (= 서가)
store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)


def add_documents(retriever, doc_summaries, doc_contents):
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)            # 요약 → 검색용
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))    # 원본 → 전달용


import re

def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트로 나눕니다."""
    b64_images = []
    texts = []
    for doc in docs:
        doc = doc.page_content if isinstance(doc, Document) else doc
        # Base64 문자열은 영문/숫자/+/= 만으로 길게 이어집니다
        if re.match("^[A-Za-z0-9+/]{100,}={0,2}$", doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

def img_prompt_func(data_dict):
    """검색된 원본을 멀티모달 LLM이 읽을 형태로 조립합니다."""
    messages = []

    # 1) 이미지를 먼저 넣습니다
    for image in data_dict["context"]["images"]:
        messages.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
        )

    # 2) 텍스트 컨텍스트와 질문을 넣습니다
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append(
        {
            "type": "text",
            "text": (
                "아래 자료를 근거로 질문에 한국어로 답하세요.\n"
                "질문에 나온 기준 시점은 답변에 그대로 포함하세요.\n"   # 추가: "24년말 대비" 같은 기준이 빠지면 수치의 의미가 사라짐
                "자료에 없는 내용은 모른다고 답하세요.\n\n"
                f"질문: {data_dict['question']}\n\n"
                f"참고 텍스트:\n{formatted_texts}"
            ),
        }
    )
    return [HumanMessage(content=messages)]

model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)

chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(img_prompt_func)
    | model
    | StrOutputParser()
)


translate_prompt = ChatPromptTemplate.from_template(
    "다음 영문 답변을 자연스러운 한국어로 번역하세요. 숫자와 고유명사는 그대로 두세요.\n\n{answer}"
)

# 번역은 로컬 오픈소스 모델로 처리해 비용을 줄일 수 있습니다
translator = OllamaLLM(model="llama3.1:8b")

final_chain = chain | {"answer": RunnablePassthrough()} | translate_prompt | translator | StrOutputParser()
