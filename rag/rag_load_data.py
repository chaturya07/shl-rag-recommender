import pandas as pd
from langchain.schema import Document

def load_shl_documents(csv_path="data/shl_assessments_cleaned.csv"):
    """
    Load SHL assessments as LangChain Documents for RAG.
    """
    df = pd.read_csv(csv_path)
    docs = []

    for _, row in df.iterrows():
        # Create rich text content for embedding
        text = f"{row['Assessment Name']}. {row['text_for_embedding']}"
        
        # Include all metadata
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "url": row["URL"],
                    "name": row["Assessment Name"],
                    "assessment_type": row.get("assessment_type", "General")
                }
            )
        )
    return docs

if __name__ == "__main__":
    docs = load_shl_documents()
    print(f"Loaded {len(docs)} documents")
    print(f"Sample: {docs[0].metadata}")
