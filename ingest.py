import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
INDEX_NAME = "medical-chatbot"

# 1. PDF load
docs = PyPDFLoader("medical.pdf").load()

# 2. Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print("Total chunks:", len(chunks))

# 3. Embeddings model (384 dimensions)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Pinecone index banao (agar pehle se nahi hai)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# 5. Chunks ko Pinecone me store
PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
print("Done! Data Pinecone me store ho gaya.")