import pandas as pd
from retrieval import recommend

df = pd.read_excel("Gen_AI Dataset.xlsx")

def extract_slug(url):
    return str(url).lower().strip().rstrip("/").split("/")[-1]

# Group by query — each query has multiple correct assessments
grouped = df.groupby("Query")["Assessment_url"].apply(
    lambda urls: [extract_slug(u) for u in urls]
).reset_index()
grouped.columns = ["Query", "true_slugs"]

def hit_at_k(pred_slugs, true_slugs, k):
    """1 if ANY of the true assessments appear in top k predictions."""
    return int(any(s in pred_slugs[:k] for s in true_slugs))

def recall_at_k(pred_slugs, true_slugs, k):
    """Fraction of true assessments found in top k predictions."""
    found = sum(1 for s in true_slugs if s in pred_slugs[:k])
    return found / len(true_slugs) if true_slugs else 0

hit5, hit10 = [], []
recall5, recall10 = [], []

for _, row in grouped.iterrows():
    query = row["Query"]
    true_slugs = row["true_slugs"]

    preds = recommend(query, k=10)["URL"].tolist()
    pred_slugs = [extract_slug(u) for u in preds]

    hit5.append(hit_at_k(pred_slugs, true_slugs, 5))
    hit10.append(hit_at_k(pred_slugs, true_slugs, 10))
    recall5.append(recall_at_k(pred_slugs, true_slugs, 5))
    recall10.append(recall_at_k(pred_slugs, true_slugs, 10))

n = len(grouped)
print(f"Evaluated on {n} unique queries\n")
print(f"Hit@5  (any correct in top 5):  {sum(hit5)/n:.2%}")
print(f"Hit@10 (any correct in top 10): {sum(hit10)/n:.2%}")
print(f"Recall@5  (avg fraction found): {sum(recall5)/n:.2%}")
print(f"Recall@10 (avg fraction found): {sum(recall10)/n:.2%}")
