import os
from groq import Groq

EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy-loaded to avoid OOM crash on startup
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        _vectorstore = FAISS.load_local(
            "rag_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vectorstore


def rag_recommend(query: str, k: int = 5):
    """
    1. Retrieve top 20 documents using FAISS
    2. Re-rank using Groq LLM
    3. Fallback to pure retrieval if LLM fails
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=20)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("⚠️ GROQ_API_KEY not set. Using retrieval only.")
        return docs[:k]

    try:
        client = Groq(api_key=api_key)

        context = "\n".join(
            [f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)]
        )

        prompt = f"""You are an expert HR assessment recommendation system.

User query: "{query}"

Below are SHL assessment descriptions:
{context}

Select the {k} most relevant assessments for the query.
Return ONLY a numbered list with assessment titles, nothing else.
Example format:
1. Assessment Title Here
2. Another Assessment Title"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512
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
                if title.lower() in doc.page_content.lower():
                    final_results.append(doc)
                    break
            if len(final_results) == k:
                break

        # Pad with top retrieval if LLM returned fewer
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
        print(f"⚠️ LLM re-ranking failed: {e}. Using retrieval fallback.")
        return docs[:k]


if __name__ == "__main__":
    results = rag_recommend("Neural Network Engineer role involving AI and deep learning", k=5)
    print("\nRecommended Assessments:\n")
    for doc in results:
        print(f"- {doc.metadata.get('name', 'Unknown')}")
