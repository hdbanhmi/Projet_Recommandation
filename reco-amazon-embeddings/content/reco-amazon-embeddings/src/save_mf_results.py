
import json
import os

def save_mf_results(
    recall_10=0.0092,
    ndcg_10=0.0046,
    path="/content/reco-amazon-embeddings/results/mf_results.json",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results = {
        "Recall@10": float(recall_10),
        "NDCG@10": float(ndcg_10),
    }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("✔ Résultats MF sauvegardés →", path)
    print(results)
    return results


if __name__ == "__main__":
    save_mf_results()
