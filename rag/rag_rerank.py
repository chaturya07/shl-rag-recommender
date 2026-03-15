import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

# Load data at startup - lightweight, just CSV + numpy
df = pd.read_csv("data/shl_assessments_cleaned.csv")
emb = np.load("data/embeddings/embeddings.npy")

# TF-IDF for query encoding - no heavy ML model needed
_tfidf = None
_tfidf_matrix = None


def get_tfidf():
    global _tfidf, _tfidf_matrix
    if _tfidf is None:
        _tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        corpus = df["Assessment Name"].fillna("").tolist()
        _tfidf_matrix = _tfidf.fit_transform(corpus)
    return _tfidf, _tfidf_matrix


def retrieve_top_k(query: str, k: int = 20):
    """Retrieve top k using TF-IDF similarity - no heavy model needed."""
    tfidf, tfidf_matrix = get_tfidf()
    query_vec = tfidf.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]

    df2 = df.copy()
    df2["score"] = sims
    df2 = df2[df2["assessment_type"].notna()]
    top = df2.sort_values("score", ascending=False).head(k)
    return top


def rag_recommend(query: str, k: int = 5):
    """
    1. Retrieve top 20 using TF-IDF (lightweight, no GPU/heavy model)
    2. Re-rank using Groq LLM
    3. Fallback to top retrieval if LLM fails
    """
    top20 = retrieve_top_k(query, k=20)

    class Doc:
        def __init__(self, row):
            self.page_content = str(row["Assessment Name"])
            self.metadata = {
                "name": row["Assessment Name"],
                "url": row["URL"],
                "assessment_type": row.get("assessment_type", "General"),
                "score": float(row["score"])
            }

    docs = [Doc(row) for _, row in top20.iterrows()]

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("⚠️ GROQ_API_KEY not set. Using retrieval only.")
        return docs[:k]

    try:
        client = Groq(api_key=api_key)
        context = "\n".join([f"{i+1}. {doc.metadata['name']}" for i, doc in enumerate(docs)])

        prompt = f"""You are an expert HR assessment recommendation system.

User query: "{query}"

Candidate SHL assessments:
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
