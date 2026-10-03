import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
INDEX_NAME = "medical-chatbot"

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
st.title("🩺 Medical Chatbot")
st.caption("RAG based chatbot: LangChain + Groq + Pinecone")

@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
    return store.as_retriever(search_kwargs={"k": 4}), llm

retriever, llm = load_components()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Apna medical sawal poochho...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a helpful medical information assistant.
Answer ONLY using the context below. If the answer is not in the context, say you don't know.
Keep the answer short and clear.

Context:
{context}

Question: {question}"""

    with st.chat_message("assistant"):
        answer = llm.invoke(prompt).content
        st.markdown(answer)
        st.caption("⚠️ Ye sirf information ke liye hai, doctor ki salah ka vikalp nahi.")

    st.session_state.messages.append({"role": "assistant", "content": answer})