
import numpy as np
import os
from dataset_textual import load_textual_dataset
from sentence_transformers import SentenceTransformer


def build_user_embeddings_concat(
    item_embedding_path="/content/reco-amazon-embeddings/data/item_embeddings.npy",
    save_path="/content/reco-amazon-embeddings/data/user_embeddings_concat_2.npy",
    X=2,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    ds = load_textual_dataset()

    print("📥 Chargement des embeddings items...")
    data = np.load(item_embedding_path, allow_pickle=True).item()
    item_ids = data["item_ids"]
    item_vecs = data["embeddings"]

    # création dictionnaire item -> index
    index = { item_ids[i]: i for i in range(len(item_ids)) }

    print("🧠 Chargement du modèle BERT pour encoding utilisateur...")
    model = SentenceTransformer(model_name)

    user_embeddings = {}

    print(f"🔧 Génération embeddings user avec stratégie X={X}")
    for user in ds.users:

        # items consommés par user (train uniquement)
        consumed = [i for (u,i,r) in ds.train if u == user]

        if len(consumed) == 0:
            user_embeddings[user] = np.zeros(item_vecs.shape[1])
            continue

        # tri pour simuler dernier enregistrement (pas d'horodatage => ordre brut)
        consumed = consumed[-X:] if len(consumed) >= X else consumed

        # concat texte des items
        text_concat = " ".join([ ds.item_texts[i] for i in consumed ])

        # embedding unique
        embedding = model.encode(text_concat, convert_to_numpy=True)
        user_embeddings[user] = embedding

    os.makedirs("/content/reco-amazon-embeddings/data", exist_ok=True)
    np.save(save_path, user_embeddings)

    print("✔ user_embeddings généré et enregistré !")
    print(f"📁 fichier → {save_path}")
    print(f"📐 Dimensions → {len(user_embeddings)} utilisateurs × {len(next(iter(user_embeddings.values())))} features")


if __name__ == "__main__":
    build_user_embeddings_concat()
