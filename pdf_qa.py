import streamlit as st
from langchain.callbacks import get_openai_callback

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection"


def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="📂"
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []
    # emb_model_nameを初期化
    if 'emb_model_name' not in st.session_state:
        st.session_state.emb_model_name = "text-embedding-ada-002"

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    else:
        st.session_state.model_name = "gpt-4"
    
    # 300: 本文以外の指示のトークン数 (以下同じ)
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)

def get_pdf_text():
    uploaded_file = st.file_uploader(
        label='Upload your PDF here📂',
        type='pdf' #アップロードを許可する拡張子
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.emb_model_name,
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
            # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
            chunk_size=500,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None

# ベクトルDBを操作するクライアントを準備
def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)
    # qdrant cloud への保存 (次の章で詳しく話します)
    # client = QdrantClient(
    #     url="https://oreno-qdrant-db.us-east-1-0.aws.cloud.qdrant.io:6333",
    #     api_key="api-key-hoge123fuga456"
    # )

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')
    # clientを用いて生成したEmbeddingをベクトルDBのcollection_nameに保管する
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )

# PDFのテキストをEmbedding（ベクトル化）してベクトルDBに保存する
def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    # 以下で与えられたテキストのEmbeddingとベクトルDBへの保存が実行される
    qdrant.add_texts(pdf_text)

# ベクトルDBを利用して質問応答を行うLangChainの機能（RetrievalQA）を呼び出す
def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        search_type="similarity", # "mmr",  "similarity_score_threshold" などもある
        search_kwargs={"k":10} # 文書を何個取得するか (default: 4)
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

# RetrievalQGを利用して質問応答を実行する
def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa(query) # query / result / source_documents
    return answer, cb.total_cost


# PDFアップロードのページの実装
def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)


# アップロードしたPDFをもとにGPTに質問するページの実装
def page_ask_my_pdf():
    st.title("Ask My PDF(s)")
    llm = select_model()
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)


def main():
    init_page()

    # ページの切り替え
    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf()

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

# ベクトルDBの中身を確認
# client = QdrantClient(path=QDRANT_PATH)
# collection_info = client.get_collection(collection_name="my_collection")
# print(f"Collection Info: {collection_info}")
# vectors = client.scroll(
#     collection_name="my_collection",
#     limit=10,  # 必要に応じて適切な数を設定
# )
# for vector in vectors[0]:
#     print(f"ベクトルDB: {vector}")


if __name__ == '__main__':
    main()
