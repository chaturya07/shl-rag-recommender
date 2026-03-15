import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load cleaned assessment data
df = pd.read_csv("data/shl_assessments_cleaned.csv")

documents = []
for _, row in df.iterrows():
    documents.append(
        Document(
            page_content=row["text_for_embedding"],
            metadata={
                "Assessment Name": row["Assessment Name"],
                "URL": row["URL"]
            }
        )
    )

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build FAISS index
vectorstore = FAISS.from_documents(documents, embeddings)

# Save index
vectorstore.save_local("rag_faiss_index")

print("✅ FAISS index rebuilt with metadata")
