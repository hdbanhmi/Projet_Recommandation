
import cornac
from cornac.datasets import amazon_clothing
from dataset_textual import load_textual_dataset
import json


def run_mf_baseline(k=50, output_path="/content/reco-amazon-embeddings/results/mf_results.json"):
    print("📥 Chargement dataset pour MF...")
    ds = load_textual_dataset()

    # Cornac requiert feedback sous forme (user,item,rating)
    train = ds.train
    test  = ds.test

    # Config MF baseline
    mf = cornac.models.MF(k=k, max_iter=50, learning_rate=0.005, lambda_reg=0.02, verbose=True)

    # Métriques obligatoires
    metrics = [
        cornac.metrics.Recall(k=10),
        cornac.metrics.NDCG(k=10),
    ]

    # Expérimentation Cornac
    print("🚀 Entraînement MF baseline...")
    exp = cornac.Experiment(
        models=[mf],
        eval_method=cornac.eval_methods.RatioSplit(
            data=train + test,
            test_size=0.2,  # on recalcule split propre MF
            rating_threshold=0.0,
            exclude_unknowns=True,
        ),
        metrics=metrics
    )

    exp.run()

    results = {
        "Recall@10": float(exp.result_dict["MF"]["Recall@10"]),
        "NDCG@10": float(exp.result_dict["MF"]["NDCG@10"]),
    }

    with open(output_path,"w") as f:
        json.dump(results,f,indent=2)

    print(f"✔ Résultats MF sauvegardés → {output_path}")
    print(results)

    return results


if __name__ == "__main__":
    run_mf_baseline()
