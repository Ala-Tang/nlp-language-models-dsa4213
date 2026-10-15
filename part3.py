"""Part III: frozen LM representations for SST-2 sentiment classification."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import part1
from data import ClassificationDataset, load_sst2_splits, prepare_lm_data
from models import build_model
from train import (
    load_checkpoint,
    load_config,
    log_data_summary,
    mode_path,
    save_json,
    select_device,
    set_seed,
    write_markdown_table,
)


@torch.no_grad()
def extract_representations(model, loader, device, include_last: bool = False):
    model.eval()
    mean_features, last_features, labels, sentences = [], [], [], []
    for input_ids, mask, batch_labels, batch_sentences in loader:
        input_ids, mask = input_ids.to(device), mask.to(device)
        states = model.encode(input_ids)
        # CHECK: 平均池化时用 mask 将 padding 状态乘零，并只除以真实 token 数。
        # 因而不同长度样本不会因补齐数量不同而改变表示尺度。
        weights = mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        mean_features.append(pooled.cpu().numpy())
        if include_last:
            indices = mask.sum(dim=1).sub(1).clamp_min(0)
            batch_indices = torch.arange(states.size(0), device=device)
            last_features.append(states[batch_indices, indices].cpu().numpy())
        labels.extend(batch_labels.numpy().tolist())
        sentences.extend(list(batch_sentences))
    result = {"mean": np.concatenate(mean_features), "labels": np.asarray(labels), "sentences": sentences}
    if include_last:
        result["last"] = np.concatenate(last_features)
    return result


def run(quick: bool = False) -> dict:
    config = load_config()
    set_seed(config["seed"])
    device = select_device(config["device"])
    output_dir = mode_path(str(Path(config["paths"]["results"]) / "part3"), quick)
    summary_path = output_dir / "summary.json"
    required = [summary_path, output_dir / "summary_table.md", output_dir / "errors.csv"]
    if all(path.exists() for path in required):
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if "data_source" in cached and "split_sizes" in cached and "vocab_size" in cached:
            log_data_summary(
                "Part III", cached["data_source"], cached["split_sizes"], cached["vocab_size"], "sample"
            )
            print(f"Part III 已完成，跳过：{output_dir}")
            return cached

    part1_summary = part1.run(quick)
    _, _, vocab, _, wikitext_source = prepare_lm_data(config, quick)
    rows, sst2_source = load_sst2_splits(config, quick)
    split_sizes = {split: len(split_rows) for split, split_rows in rows.items()}
    data_source = {"wikitext_vocabulary": wikitext_source, "sst2": sst2_source}
    log_data_summary("Part III", data_source, split_sizes, len(vocab.itos), "sample")
    datasets = {
        split: ClassificationDataset(split_rows, vocab, config["part3"]["max_length"])
        for split, split_rows in rows.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config["part3"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
        )
        for split, dataset in datasets.items()
    }
    checkpoint_dir = mode_path(str(Path(config["paths"]["checkpoints"]) / "part1"), quick)
    models = {}
    for name in ("lstm", "transformer"):
        model = build_model(name, len(vocab.itos), vocab.pad_id, config)
        load_checkpoint(model, checkpoint_dir / f"{name}.pt", device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        models[name] = model

    features = {}
    for model_name, model in models.items():
        for split, loader in loaders.items():
            extracted = extract_representations(
                model, loader, device, include_last=(model_name == "lstm")
            )
            features[(model_name, split, "mean")] = extracted["mean"]
            if model_name == "lstm":
                features[(model_name, split, "last")] = extracted["last"]

    labels = {split: np.asarray([row["label"] for row in split_rows]) for split, split_rows in rows.items()}
    predictions = {}
    metrics = {}
    for representation in ("lstm_mean", "transformer_mean", "lstm_last"):
        model_name, pooling = representation.rsplit("_", 1)
        train_x = features[(model_name, "train", pooling)]
        val_x = features[(model_name, "validation", pooling)]
        test_x = features[(model_name, "test", pooling)]
        best_classifier = None
        best_score = -1.0
        best_c = None
        # CHECK: 每种表示独立地只在 validation 上选择 C，test 保持封存。
        for c_value in config["part3"]["c_values"]:
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    max_iter=config["part3"]["logistic_max_iter"],
                    random_state=config["seed"],
                ),
            )
            classifier.fit(train_x, labels["train"])
            score = accuracy_score(labels["validation"], classifier.predict(val_x))
            if score > best_score:
                best_score, best_c, best_classifier = score, c_value, classifier
        prediction = best_classifier.predict(test_x)
        predictions[representation] = prediction
        metrics[representation] = {
            "best_c": best_c,
            "validation_accuracy": best_score,
            "test_accuracy": accuracy_score(labels["test"], prediction),
            "test_macro_f1": f1_score(labels["test"], prediction, average="macro"),
        }

    lm_names = ("lstm", "transformer")
    perplexities = {
        name: part1_summary["models"][name]["test"]["perplexity"] for name in lm_names
    }
    accuracies = {name: metrics[f"{name}_mean"]["test_accuracy"] for name in lm_names}
    ranking = [
        {
            "model": name,
            "part1_perplexity": perplexities[name],
            "perplexity_rank": int(
                1 + sum(value < perplexities[name] for value in perplexities.values())
            ),
            "classification_accuracy": accuracies[name],
            "classification_rank": int(
                1 + sum(value > accuracies[name] for value in accuracies.values())
            ),
        }
        for name in lm_names
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    error_path = output_dir / "errors.csv"
    with error_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sentence",
            "true_label",
            "lstm_mean_prediction",
            "transformer_mean_prediction",
            "lstm_last_prediction",
            "lstm_transformer_disagree",
            "any_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows["test"]):
            lstm_prediction = int(predictions["lstm_mean"][index])
            transformer_prediction = int(predictions["transformer_mean"][index])
            last_prediction = int(predictions["lstm_last"][index])
            truth = int(row["label"])
            writer.writerow(
                {
                    "sentence": row["sentence"],
                    "true_label": truth,
                    "lstm_mean_prediction": lstm_prediction,
                    "transformer_mean_prediction": transformer_prediction,
                    "lstm_last_prediction": last_prediction,
                    "lstm_transformer_disagree": lstm_prediction != transformer_prediction,
                    "any_error": any(
                        prediction != truth
                        for prediction in (lstm_prediction, transformer_prediction, last_prediction)
                    ),
                }
            )

    table_rows = [
        {"representation": name, **values} for name, values in metrics.items()
    ]
    write_markdown_table(
        table_rows,
        [
            "representation",
            "best_c",
            "validation_accuracy",
            "test_accuracy",
            "test_macro_f1",
        ],
        output_dir / "summary_table.md",
    )
    summary = {
        "quick": quick,
        "seed": config["seed"],
        "dataset": "nyu-mll/glue:sst2",
        "data_source": data_source,
        "split_sizes": split_sizes,
        "vocab_size": len(vocab.itos),
        "oov_ratio": {split: dataset.oov_ratio for split, dataset in datasets.items()},
        "metrics": metrics,
        "rank_comparison": ranking,
        "language_models_frozen": True,
    }
    save_json(summary, summary_path)
    print(f"Part III 完成：{output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.quick)
