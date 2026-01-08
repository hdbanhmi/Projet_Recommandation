
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataset_textual import load_textual_dataset


def recommend_top_k(
    user_emb_path="/content/reco-amazon-embeddings/data/user_embeddings_concat_2.npy",
    item_emb_path="/content/reco-amazon-embeddings/data/item_embeddings.npy",
    K=10
):
    ds = load_textual_dataset()

    # Chargement embeddings
    user_emb = np.load(user_emb_path, allow_pickle=True).item()   # dict user -> vector
    item_data = np.load(item_emb_path, allow_pickle=True).item()
    item_ids  = item_data["item_ids"]
    item_vecs = item_data["embeddings"]

    item_matrix = np.vstack(item_vecs)            # (nb_items, 384)
    item_index  = { item_ids[i]: i for i in range(len(item_ids)) }

    recommendations = {}

    print(f"🔎 Génération des recommandations Top-{K} pour {len(ds.users)} utilisateurs...")

    for user in ds.users:

        # vecteur utilisateur
        u_vec = user_emb[user].reshape(1,-1)

        # similarité user ↔ tous les items
        scores = cosine_similarity(u_vec, item_matrix)[0]

        # items déjà vus → exclusion
        seen = { i for (u,i,r) in ds.train if u == user }
        candidate_scores = [(item_ids[i], scores[i]) for i in range(len(item_ids)) if item_ids[i] not in seen]

        # tri Top-K
        top_k = sorted(candidate_scores, key=lambda x: x[1], reverse=True)[:K]
        recommendations[user] = top_k

    print("✔ Recommandations générées avec succès !")
    return recommendations


if __name__ == "__main__":
    recs = recommend_top_k(K=10)
