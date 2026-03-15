from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time
import csv
import json
import io
import uuid
from typing import Optional
import pandas as pd
from database import save_search, get_history, clear_history, get_analytics_summary

app = FastAPI(
    title="SHL Assessment RAG Recommendation API",
    version="2.0.0",
    description="RAG-based API to recommend SHL assessments using retrieval and LLM re-ranking."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Schemas
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    k: int = 10
    assessment_type: Optional[str] = None
    session_id: Optional[str] = None  # for search history


class ExportRequest(BaseModel):
    results: list
    query: str
    format: str = "csv"  # "csv" or "json"


# -----------------------------
# Cache
# -----------------------------
query_cache = {}
MAX_CACHE_SIZE = 1000


# -----------------------------
# RAG with fallback
# -----------------------------
def rag_recommend_with_fallback(query: str, k: int = 10, assessment_type: Optional[str] = None):
    try:
        from rag.rag_rerank import rag_recommend
        docs = rag_recommend(query, k=k * 2)

        results = []
        for doc in docs:
            results.append({
                "Assessment Name": doc.metadata.get("name", "Unknown"),
                "URL": doc.metadata.get("url", ""),
                "score": 0.95,
                "assessment_type": doc.metadata.get("assessment_type", "General")
            })

        df = pd.DataFrame(results)
        if assessment_type and assessment_type.lower() != "all":
            df = df[df["assessment_type"].str.lower() == assessment_type.lower()]

        return df.head(k), "rag"

    except Exception as e:
        print(f"⚠️ RAG failed: {e}. Falling back to semantic search.")
        from retrieval import recommend
        return recommend(query, k=k, assessment_type=assessment_type), "semantic"


def get_cached_or_compute(query: str, k: int, assessment_type: Optional[str]):
    cache_key = f"{query}:{k}:{assessment_type}"
    if cache_key in query_cache:
        return query_cache[cache_key], True, "cached"

    result, method = rag_recommend_with_fallback(query, k=k, assessment_type=assessment_type)

    if len(query_cache) >= MAX_CACHE_SIZE:
        del query_cache[next(iter(query_cache))]

    query_cache[cache_key] = result
    return result, False, method


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "cache_size": len(query_cache),
        "mode": "RAG (Retrieval-Augmented Generation)"
    }


@app.get("/stats")
def get_stats():
    db_stats = get_analytics_summary()
    return {
        "total_assessments": 377,
        "cache_entries": len(query_cache),
        "supported_types": ["Personality & Behavior", "Cognitive", "Simulation",
                            "Technical Skills", "Language", "General"],
        **db_stats
    }


@app.post("/recommend")
def recommend_assessments(req: QueryRequest):
    start_time = time.time()
    session_id = req.session_id or str(uuid.uuid4())

    try:
        results, cached, method = get_cached_or_compute(
            req.query, req.k, req.assessment_type
        )
        query_time_ms = (time.time() - start_time) * 1000
        results_list = results.to_dict(orient="records")

        # Save to search history
        if not cached:
            save_search(
                session_id=session_id,
                query=req.query,
                result_count=len(results_list),
                method=method,
                query_time_ms=round(query_time_ms, 2),
                assessment_type=req.assessment_type
            )

        print(f"✅ [{method}] {len(results_list)} results in {query_time_ms:.0f}ms")
        return results_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}")
def search_history(session_id: str):
    """Get search history for a session."""
    return get_history(session_id)


@app.delete("/history/{session_id}")
def delete_history(session_id: str):
    """Clear search history for a session."""
    clear_history(session_id)
    return {"message": "History cleared"}


@app.post("/export")
def export_results(req: ExportRequest):
    """Export search results as CSV or JSON."""
    if req.format == "json":
        content = json.dumps({
            "query": req.query,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(req.results),
            "results": req.results
        }, indent=2)
        return StreamingResponse(
            io.StringIO(content),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=shl_results.json"}
        )

    # Default: CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Assessment Name", "URL", "score", "assessment_type"])
    writer.writeheader()
    writer.writerows(req.results)

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=shl_results.csv"}
    )
