import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

# Load data at startup (lightweight)
df = pd.read_csv("data/shl_assessments_cleaned.csv")

# Build TF-IDF index over assessment text — no heavy model needed
_tfidf = None
_tfidf_matrix = None

def get_tfidf():
    global _tfidf, _tfidf_matrix
    if _tfidf is None:
        corpus = df["text_for_embedding"].fillna("").tolist()
        _tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
        _tfidf_matrix = _tfidf.fit_transform(corpus)
        print("✅ TF-IDF index built")
    return _tfidf, _tfidf_matrix


def retrieve_top_k(query: str, k: int = 20):
    tfidf, matrix = get_tfidf()
    q_vec = tfidf.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]

    df2 = df.copy()
    df2["score"] = sims
    df2 = df2[df2["assessment_type"].notna()]
    return df2.sort_values("score", ascending=False).head(k)


def rag_recommend(query: str, k: int = 5):
    """
    1. Retrieve top 20 via TF-IDF (lightweight, no OOM)
    2. Re-rank using Groq LLM
    3. Fallback to retrieval order if LLM fails
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

Candidate assessments:
{context}

Select the {k} most relevant assessments for this job role.
Return ONLY a numbered list of assessment names, nothing else."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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

        # Rank-based scores: top result keeps its score, each rank drops ~3%
        if final_results:
            top_score = max(final_results[0].metadata["score"], 0.3)
            for i, doc in enumerate(final_results):
                doc.metadata["score"] = round(min(top_score * (1.0 - i * 0.03), 0.99), 4)

        print(f"✅ Groq re-ranking complete ({len(final_results)} results)")
        return final_results

    except Exception as e:
        print(f"⚠️ Groq failed: {e}. Using retrieval fallback.")
        return docs[:k]
