import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data and model at module level for performance
try:
    df = pd.read_csv("data/shl_assessments_cleaned.csv")
    emb = np.load("data/embeddings/embeddings.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info(f"Loaded {len(df)} assessments and embeddings successfully")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

def recommend(query: str, k: int = 10, assessment_type: str | None = None):
    """
    Recommend SHL assessments based on semantic similarity.
    
    Args:
        query: Job description or role context
        k: Number of results to return (1-50)
        assessment_type: Optional filter by assessment type
        
    Returns:
        DataFrame with top k recommendations
    """
    try:
        # Encode query
        q_emb = model.encode([query])
        
        # Compute similarities
        sims = cosine_similarity(q_emb, emb)[0]

        # Create result dataframe
        df2 = df.copy()
        df2["score"] = sims

        # Filter: keep only valid assessments
        df2 = df2[df2["assessment_type"].notna()]

        # Optional filter by assessment_type
        if assessment_type and assessment_type.lower() != "all":
            df2 = df2[df2["assessment_type"].str.lower() == assessment_type.lower()]

        # Sort by score descending
        df2 = df2.sort_values("score", ascending=False)

        # Return top k results
        result = df2.head(k)[["Assessment Name", "URL", "score", "assessment_type"]]
        
        logger.info(f"Query processed: {len(result)} results returned")
        return result
        
    except Exception as e:
        logger.error(f"Error in recommend function: {e}")
        raise


if __name__ == "__main__":
    q = "Consultant role involving talent assessment and analytics"
    result = recommend(q, k=5)
    print(result)
