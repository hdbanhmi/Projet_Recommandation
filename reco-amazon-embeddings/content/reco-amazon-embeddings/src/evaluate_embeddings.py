
import numpy as np
from sklearn.metrics import ndcg_score
from dataset_textual import load_textual_dataset
from sklearn.metrics.pairwise import cosine_similarity
import json


def evaluate_embeddings(
    user_emb_path="/content/reco-amazon-embeddings/data/user_embeddings_concat_2.npy",
    item_emb_path="/content/reco-amazon-embeddings/data/item_embeddings.npy",
    save_path="/content/reco-amazon-embeddings/results/embeddings_results.json",
    K=10
):
    ds = load_textual_dataset()

    # Chargement embeddings
    user_embeddings = np.load(user_emb_path, allow_pickle=True).item()
    item_data = np.load(item_emb_path, allow_pickle=True).item()

    item_ids  = item_data["item_ids"]
    item_vecs = item_data["embeddings"]

    item_index = {item_ids[i]: i for i in range(len(item_ids))}
    item_matrix = np.vstack(item_vecs)

    Recall_total = 0
    NDCG_total   = 0
    user_count   = 0

    print("📊 Évaluation modèle embeddings...")

    for user in ds.users:
        if user not in user_embeddings:
            continue

        u_vec = user_embeddings[user].reshape(1,-1)
        scores = cosine_similarity(u_vec, item_matrix)[0]

        # items consommés = ground truth
        gt_items = {i for (u,i,r) in ds.test if u == user}
        if len(gt_items)==0:
            continue

        # exclude train items
        train_items = {i for (u,i,r) in ds.train if u == user}
        candidate_scores = [(item_ids[i], scores[i]) for i in range(len(item_ids)) if item_ids[i] not in train_items]

        # ranking
        ranked = sorted(candidate_scores, key=lambda x: x[1], reverse=True)[:K]
        recommended = [item for item,score in ranked]

        # recall@10
        hits = len([i for i in recommended if i in gt_items])
        recall = hits / len(gt_items)

        # NDCG@10
        y_true = [[1 if item in gt_items else 0 for item,score in ranked]]
        y_score= [[score for item,score in ranked]]
        ndcg   = ndcg_score(y_true,y_score)

        Recall_total += recall
        NDCG_total   += ndcg
        user_count   += 1

    results = {
        "Recall@10": float(Recall_total/user_count),
        "NDCG@10":   float(NDCG_total/user_count)
    }

    with open(save_path,"w") as f:
        json.dump(results,f,indent=2)

    print("
📁 Résultats embeddings sauvegardés →", save_path)
    print(results)
    return results


if __name__ == "__main__":
    evaluate_embeddings()
