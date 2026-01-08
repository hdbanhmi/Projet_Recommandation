
from sentence_transformers import SentenceTransformer
import numpy as np
import os

from dataset_textual import load_textual_dataset


def generate_item_embeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    save_path="/content/reco-amazon-embeddings/data/item_embeddings.npy",
    batch_size=64,
):
    ds = load_textual_dataset()

    print(f"🧠 Chargement du modèle BERT : {model_name}")
    model = SentenceTransformer(model_name)

    item_ids = list(ds.items)
    texts = [ds.item_texts[i] for i in item_ids]

    print("📌 Items à encoder :", len(item_ids))

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    np.save(save_path, {"item_ids": item_ids, "embeddings": embeddings})
    print(f"✔ Embeddings sauvegardés → {save_path}")
    print("📐 Vecteurs shape :", embeddings.shape)

    return embeddings, item_ids


if __name__ == "__main__":
    generate_item_embeddings()
