import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader, UnstructuredPDFLoader
import unstructured
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from sklearn.metrics import precision_score, recall_score
import time
from langchain_core.documents import Document
import pandas as pd
from langchain_core.documents import Document

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ChatMessageHistory()
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
history = st.session_state.get("chat_history", ChatMessageHistory())

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses context from uploaded documents."),
    ("human", "Conversation so far:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}")
])

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0,api_key="")
def ask_question(question):
    st.session_state["chat_history"].add_user_message(question)

    retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 3})

    history_text = "\n".join([m.content for m in st.session_state["chat_history"].messages])

    qa_chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "history": lambda x: history_text
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = qa_chain.invoke({"question": question})
    st.session_state["chat_history"].add_ai_message(result)
    return result



st.title("🤖 I'm Your Personalized Chatbot")
st.write("Kindly use me as your personal assistant")

uploaded_files = st.file_uploader("Upload your documents (PDF, Docx, txt, csv, xlsx)", type=["pdf", "txt", "csv", "xlsx", "docx"],accept_multiple_files=True)

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        suffix = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if suffix == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())

        elif suffix == "txt":
            loader = TextLoader(tmp_path)
            docs.extend(loader.load())

        elif suffix == "csv":
            df = pd.read_csv(tmp_path)
            text = df.to_string()
            docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))

        elif suffix == "xlsx":
            df = pd.read_excel(tmp_path)
            text = df.to_string()
            docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))

        elif suffix == "docx":
            doc = docx.Document(tmp_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))

    st.session_state["vectorstore"] = FAISS.from_documents(docs, embeddings)


for msg in st.session_state["chat_history"].messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)

if prompt_text := st.chat_input("Ask me something about your documents..."):
    if st.session_state["vectorstore"] is None:
        st.warning("Please upload documents first.")
    else:
        answer = ask_question(prompt_text)
        with st.chat_message("assistant"):
            st.write(answer)
