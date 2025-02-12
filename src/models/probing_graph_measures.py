import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import CLIPVisionConfig

os.environ["PYTHONWARNINGS"] = "ignore"
logger = logging.getLogger(__name__)
load_dotenv()

scoring_functions = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
}


@hydra.main(config_path="../../configs/model", config_name="probing_graph_measures")
def run_probing_graph_measures(cfg: DictConfig) -> None:
    # Load the metadata
    nsd_dir = Path(cfg.data.large_data_path) / cfg.data.nsd_directory
    nsd_vg_metadata = pd.read_csv(nsd_dir / "nsd_vg" / "nsd_vg_metadata.csv")

    # Check scoring
    if cfg.scoring not in scoring_functions:
        raise ValueError(f"Scoring function {cfg.scoring} not supported. Choose from {list(scoring_functions.keys())}")

    # Initialize results DataFrame
    results = pd.DataFrame(
        columns=["subject", "model_id", "layer", "target_variable", f"{cfg.scoring}_mean", f"{cfg.scoring}_std"]
    )

    # Create output directory
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_id in cfg.model_ids:
        if cfg.by_layer:
            n_layers = CLIPVisionConfig.from_pretrained(model_id).num_hidden_layers
            layers = [f"_layer_{i}" for i in range(n_layers)]
        else:
            layers = [""]
        for layer in tqdm(layers, desc=f"Iterating over layers for {model_id}"):
            # Load embeddings per layer
            embedding_file = Path(cfg.data.embedding_dir) / f"{model_id.replace('/', '_')}{layer}_embeddings.npy"
            X_embeddings_initial = np.load(embedding_file)
            # Create a mapping: nsdId -> embedding (assumes same order as in nsd_vg_metadata)
            X_embeddings_mapping = {k: v for k, v in zip(nsd_vg_metadata["nsdId"], X_embeddings_initial)}
            X_embeddings = np.array([X_embeddings_mapping[nsd_id] for nsd_id in nsd_vg_metadata["nsdId"]])

            # Aggregate embeddings by unique cocoId
            unique_coco_ids = nsd_vg_metadata["cocoId"].unique()
            X_aggregated_list = []
            targets_aggregated: dict = {tv: [] for tv in cfg.target_variables}
            for c_id in unique_coco_ids:
                idx = nsd_vg_metadata.index[nsd_vg_metadata["cocoId"] == c_id].tolist()
                X_agg = X_embeddings[idx].mean(axis=0)
                X_aggregated_list.append(X_agg)
                for tv in cfg.target_variables:
                    # Take the target variable from the first entry
                    targets_aggregated[tv].append(nsd_vg_metadata.loc[idx[0], tv])
            X_aggregated = np.stack(X_aggregated_list, axis=0)

            for target_variable in tqdm(
                cfg.target_variables,
                desc=f"Iterating over target variables for {model_id} and {layer}",
            ):
                y_agg = np.array(targets_aggregated[target_variable])
                # Binarize target variable if specified
                if cfg.binarize_target:
                    median = np.percentile(y_agg, 50)
                    y_agg = (y_agg > median).astype(int)

                # 5-fold cross-validation with LogisticRegression
                skf = StratifiedKFold(n_splits=cfg.cv, shuffle=True, random_state=cfg.random_state)
                accs = []
                for train_idx, test_idx in skf.split(X_aggregated, y_agg):
                    clf = RidgeClassifierCV() if cfg.ridge_cv else RidgeClassifier()
                    clf.fit(X_aggregated[train_idx], y_agg[train_idx])
                    y_pred = clf.predict(X_aggregated[test_idx])
                    accs.append(scoring_functions[cfg.scoring](y_agg[test_idx], y_pred))

                results = pd.concat(
                    [
                        results,
                        pd.DataFrame(
                            [
                                {
                                    "model_id": model_id,
                                    "layer": layer,
                                    "target_variable": target_variable,
                                    f"{cfg.scoring}_mean": np.mean(accs),
                                    f"{cfg.scoring}_std": np.std(accs),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    # Save results
    output_path = Path(cfg.data.output_dir) / "probing_graph_measures_results.csv"
    results.to_csv(output_path, index=False)
    logger.info("Graph measures probing complete.")


if __name__ == "__main__":
    run_probing_graph_measures()
