from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_load_data import load_shl_documents

def build_vectorstore():
    docs = load_shl_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("rag_faiss_index")
    print("FAISS index saved")

if __name__ == "__main__":
    build_vectorstore()
