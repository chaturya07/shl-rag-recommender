import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Load lightweight data at startup (no heavy ML models)
_df = None
_emb = None
_query_model = None


def get_data():
    global _df, _emb
    if _df is None:
        _df = pd.read_csv("data/shl_assessments_cleaned.csv")
        _emb = np.load("data/embeddings/embeddings.npy")
    return _df, _emb


def get_query_model():
    global _query_model
    if _query_model is None:
        from sentence_transformers import SentenceTransformer
        _query_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _query_model


def rag_recommend(query: str, k: int = 5):
    """
    1. Retrieve top 20 using pre-computed embeddings (no FAISS, no heavy model load at startup)
    2. Re-rank using Groq LLM
    3. Fallback to top retrieval if LLM fails
    """
    df, emb = get_data()
    model = get_query_model()

    # Encode query and retrieve top 20
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, emb)[0]
    df2 = df.copy()
    df2["score"] = sims
    df2 = df2[df2["assessment_type"].notna()]
    top20 = df2.sort_values("score", ascending=False).head(20)

    # Build simple doc-like objects for compatibility
    class Doc:
        def __init__(self, row):
            self.page_content = f"{row['Assessment Name']}: {row.get('description', row['Assessment Name'])}"
            self.metadata = {
                "name": row["Assessment Name"],
                "url": row["URL"],
                "assessment_type": row.get("assessment_type", "General"),
                "score": row["score"]
            }

    docs = [Doc(row) for _, row in top20.iterrows()]

    # Try Groq re-ranking
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("⚠️ GROQ_API_KEY not set. Using retrieval only.")
        return docs[:k]

    try:
        client = Groq(api_key=api_key)
        context = "\n".join([f"{i+1}. {doc.metadata['name']}" for i, doc in enumerate(docs)])

        prompt = f"""You are an expert HR assessment recommendation system.

User query: "{query}"

Below are candidate SHL assessments:
{context}

Select the {k} most relevant assessments for this job role.
Return ONLY a numbered list of assessment names, nothing else."""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256
        )

        content = response.choices[0].message.content
        selected_titles = [
            line.split(". ", 1)[1].strip()
            for line in content.strip().split("\n")
            if ". " in line and line[0].isdigit()
        ]

        final_results = []
        for title in selected_titles:
            for doc in docs:
                if title.lower() in doc.metadata["name"].lower() or doc.metadata["name"].lower() in title.lower():
                    final_results.append(doc)
                    break
            if len(final_results) == k:
                break

        # Pad if needed
        if len(final_results) < k:
            seen = {id(d) for d in final_results}
            for doc in docs:
                if id(doc) not in seen:
                    final_results.append(doc)
                if len(final_results) == k:
                    break

        print(f"✅ Groq re-ranking complete ({len(final_results)} results)")
        return final_results

    except Exception as e:
        print(f"⚠️ Groq failed: {e}. Using retrieval fallback.")
        return docs[:k]
