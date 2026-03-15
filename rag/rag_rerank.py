import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import RateLimitError, AuthenticationError


# -----------------------------
# Load Vector Store (FAISS)
# -----------------------------
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

vectorstore = FAISS.load_local(
    "rag_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# -----------------------------
# RAG Recommendation Function
# -----------------------------
def rag_recommend(query: str, k: int = 5):
    """
    1. Retrieve top 20 documents using FAISS (cosine similarity)
    2. Re-rank using DeepSeek LLM (LangChain)
    3. Fallback to pure retrieval if LLM fails
    """

    # Step 1: Retrieve top candidates
    docs = vectorstore.similarity_search(query, k=20)

    # Try LLM re-ranking
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            base_url="https://api.deepseek.com"
        )

        context = "\n".join(
            [f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)]
        )

        prompt = f"""
You are an expert HR assessment recommendation system.

User query:
"{query}"

Below are SHL assessment descriptions:
{context}

Task:
Select the {k} most relevant assessments for the query.
Return ONLY a numbered list with assessment titles.
Do NOT explain.
"""

        response = llm.invoke(prompt)
        selected_titles = [
            line.split(". ", 1)[1].strip()
            for line in response.content.split("\n")
            if ". " in line
        ]

        # Match LLM-selected titles back to documents
        final_results = []
        for title in selected_titles:
            for doc in docs:
                if title.lower() in doc.page_content.lower():
                    final_results.append(doc)
                    break
            if len(final_results) == k:
                break

        print("✅ LLM re-ranking enabled")
        return final_results

    except (RateLimitError, AuthenticationError, Exception):
        print("⚠️ LLM unavailable. Falling back to embedding-based retrieval.")
        return docs[:k]


# -----------------------------
# CLI Test
# -----------------------------
if __name__ == "__main__":
    results = rag_recommend(
        "Neural Network Engineer role involving AI and deep learning",
        k=5
    )

    print("\nRecommended Assessments:\n")
    for doc in results:
        print(f"- {doc.metadata.get('Assessment Name')}")
        print(f"  {doc.metadata.get('URL')}\n")
