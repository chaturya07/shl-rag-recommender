import pandas as pd
from retrieval import recommend

# Load dataset
df = pd.read_excel("Gen_AI Dataset.xlsx")

def extract_slug(url):
    url = url.lower().strip().rstrip("/")
    return url.split("/")[-1]

def hit_at_k(pred_slugs, true_slug, k):
    return int(true_slug in pred_slugs[:k])

hit5 = []
hit10 = []

for _, row in df.iterrows():
    query = row["Query"]
    true_slug = extract_slug(row["Assessment_url"])

    preds = recommend(query, k=10)["URL"].tolist()
    pred_slugs = [extract_slug(u) for u in preds]

    hit5.append(hit_at_k(pred_slugs, true_slug, 5))
    hit10.append(hit_at_k(pred_slugs, true_slug, 10))

print("Hit@5:", sum(hit5) / len(hit5))
print("Hit@10:", sum(hit10) / len(hit10))
