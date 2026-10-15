"""Part II: LSTM language model with three embedding settings."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import part1
from data import Vocabulary, prepare_lm_data
from models import build_model
from train import (
    evaluate,
    generate_neural,
    inference_throughput,
    log_data_summary,
    load_checkpoint,
    load_config,
    mode_path,
    parameter_counts,
    plot_histories,
    save_json,
    select_device,
    set_seed,
    train_model,
    write_markdown_table,
)


def _matrix_from_vectors(vocab: Vocabulary, vectors, dimension: int, seed: int):
    covered = [word for word in vocab.itos if word in vectors]
    standard_deviation = float(np.asarray(vectors.vectors).std())
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, standard_deviation, (len(vocab.itos), dimension)).astype("float32")
    for index, word in enumerate(vocab.itos):
        if word in vectors:
            matrix[index] = vectors[word]
    matrix[vocab.pad_id] = 0.0
    return torch.from_numpy(matrix), {
        "covered": len(covered),
        "vocabulary_size": len(vocab.itos),
        "coverage": len(covered) / len(vocab.itos),
        "oov": len(vocab.itos) - len(covered),
        "oov_initialization_std": standard_deviation,
    }


def _quick_embedding(vocab: Vocabulary, dimension: int, seed: int, fraction: float):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.1, (len(vocab.itos), dimension)).astype("float32")
    matrix[vocab.pad_id] = 0.0
    covered = int((len(vocab.itos) - 1) * fraction)
    return torch.from_numpy(matrix), {
        "covered": covered,
        "vocabulary_size": len(vocab.itos),
        "coverage": covered / len(vocab.itos),
        "oov": len(vocab.itos) - covered,
        "oov_initialization_std": 0.1,
        "quick_mock": True,
    }


def train_word2vec(lines, vocab, config, quick: bool):
    dimension = config["model"]["embedding_dim"]
    if quick:
        start = time.perf_counter()
        matrix, coverage = _quick_embedding(vocab, dimension, config["seed"], 0.9)
        return matrix, coverage, time.perf_counter() - start, "quick_mock:word2vec"
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise RuntimeError("Part II 需要 gensim；请先安装 requirements.txt。") from exc
    sentences = [line.lower().split() + ["<eos>"] for line in lines if line.strip()]
    start = time.perf_counter()
    model = Word2Vec(
        sentences=sentences,
        vector_size=dimension,
        window=config["word2vec"]["window"],
        min_count=1,
        sg=1,
        negative=config["word2vec"]["negative"],
        epochs=config["word2vec"]["epochs"],
        workers=1,
        seed=config["seed"],
    )
    elapsed = time.perf_counter() - start
    matrix, coverage = _matrix_from_vectors(vocab, model.wv, dimension, config["seed"])
    return matrix, coverage, elapsed, "trained:gensim_skipgram:wikitext_train"


def load_glove(vocab, config, quick: bool):
    dimension = config["model"]["embedding_dim"]
    if quick:
        matrix, coverage = _quick_embedding(vocab, dimension, config["seed"] + 1, 0.65)
        return matrix, coverage, "quick_mock:glove-wiki-gigaword-100"
    try:
        import gensim.downloader as api
        vectors = api.load("glove-wiki-gigaword-100")
    except Exception as exc:
        raise RuntimeError(
            "无法加载真实 GloVe glove-wiki-gigaword-100；"
            "非 quick 模式禁止使用后备 embedding。"
        ) from exc
    matrix, coverage = _matrix_from_vectors(vocab, vectors, dimension, config["seed"])
    return matrix, coverage, "gensim:glove-wiki-gigaword-100"


def run(quick: bool = False) -> dict:
    config = load_config()
    set_seed(config["seed"])
    device = select_device(config["device"])
    output_dir = mode_path(str(Path(config["paths"]["results"]) / "part2"), quick)
    checkpoint_dir = mode_path(str(Path(config["paths"]["checkpoints"]) / "part2"), quick)
    summary_path = output_dir / "summary.json"
    required = [
        summary_path,
        output_dir / "summary_table.md",
        output_dir / "loss_curves.png",
        output_dir / "generations.md",
        output_dir / "coverage.json",
    ]
    if all(path.exists() for path in required):
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if "data_source" in cached and "token_counts" in cached:
            log_data_summary(
                "Part II", cached["data_source"], cached["token_counts"], cached["vocab_size"], "token"
            )
            print(f"Part II 已完成，跳过：{output_dir}")
            return cached

    part1_summary = part1.run(quick)
    lines, streams, vocab, loaders, wikitext_source = prepare_lm_data(config, quick)
    token_counts = {split: len(tokens) for split, tokens in streams.items()}
    config["_pad_id"] = vocab.pad_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    part1_checkpoint_dir = mode_path(str(Path(config["paths"]["checkpoints"]) / "part1"), quick)
    random_model = build_model("lstm", len(vocab.itos), vocab.pad_id, config)
    random_history = load_checkpoint(random_model, part1_checkpoint_dir / "lstm.pt", device)

    word2vec_matrix, word2vec_coverage, word2vec_seconds, word2vec_source = train_word2vec(
        lines["train"], vocab, config, quick
    )
    glove_matrix, glove_coverage, glove_source = load_glove(vocab, config, quick)
    data_source = {
        "wikitext": wikitext_source,
        "word2vec": word2vec_source,
        "glove": glove_source,
    }
    log_data_summary("Part II", data_source, token_counts, len(vocab.itos), "token")
    coverage = {"word2vec": word2vec_coverage, "glove": glove_coverage}
    save_json(coverage, output_dir / "coverage.json")

    models = {"random_trainable": random_model}
    histories = {"random_trainable": random_history}
    summaries = {
        "random_trainable": {
            **part1_summary["models"]["lstm"],
            "reused_part1": True,
            "embedding_frozen_test": "N/A",
        }
    }
    for name, matrix in (("word2vec_frozen", word2vec_matrix), ("glove_frozen", glove_matrix)):
        set_seed(config["seed"])
        model = build_model(
            "lstm",
            len(vocab.itos),
            vocab.pad_id,
            config,
            embedding_weights=matrix,
            freeze_embedding=True,
        )
        # CHECK: 固定 embedding 的整张权重表 requires_grad=False，包括随机 OOV 行。
        # 训练前后逐元素比较，确认优化器没有改变任何 embedding 权重。
        before = model.embedding.weight.detach().cpu().clone()
        checkpoint = checkpoint_dir / f"{name}.pt"
        if checkpoint.exists():
            history = load_checkpoint(model, checkpoint, device)
        else:
            history = train_model(model, loaders, device, config, checkpoint, quick)
        frozen = bool(torch.equal(before, model.embedding.weight.detach().cpu()))
        print(f"{name} embedding freeze test: {'PASSED' if frozen else 'FAILED'}")
        if not frozen:
            raise AssertionError(f"{name} embedding 被意外更新")
        models[name] = model
        histories[name] = history
        summaries[name] = {
            "test": evaluate(model, loaders["test"], device, vocab.pad_id),
            "parameters": parameter_counts(model),
            "train_seconds": history.get("total_train_seconds"),
            "epoch_seconds": history.get("epoch_seconds", []),
            "best_validation_loss": history.get("best_validation_loss"),
            "inference_tokens_per_second": inference_throughput(
                model, loaders["test"], device, vocab.pad_id, config["train"]["warmup_batches"]
            ),
            "embedding_frozen_test": "PASSED",
        }

    summaries["word2vec_frozen"]["embedding_training_seconds"] = word2vec_seconds
    generation_cfg = config["train"]["generation"]
    generation_lines = ["# Part II generations", ""]
    for prompt_index, prompt in enumerate(generation_cfg["prompts"]):
        generation_lines.extend([f"## Prompt: `{prompt}`", ""])
        prompt_ids = vocab.encode(prompt.lower().split())
        for name, model in models.items():
            ids = generate_neural(
                model,
                prompt_ids,
                device,
                config["data"]["context_length"],
                generation_cfg["max_new_tokens"],
                generation_cfg["temperature"],
                generation_cfg["top_k"],
                config["seed"] + prompt_index,
            )
            generation_lines.extend([f"### {name}", " ".join(vocab.decode(ids)), ""])
    (output_dir / "generations.md").write_text("\n".join(generation_lines), encoding="utf-8")

    plot_histories(histories, output_dir / "loss_curves.png", "Part II embedding comparison")
    rows = []
    for name, values in summaries.items():
        rows.append(
            {
                "setting": name,
                "test_loss": values["test"]["loss"],
                "test_perplexity": values["test"]["perplexity"],
                "train_seconds": values.get("train_seconds", ""),
                "embedding_train_seconds": values.get("embedding_training_seconds", 0.0),
                "frozen": values["embedding_frozen_test"],
            }
        )
    write_markdown_table(
        rows,
        [
            "setting",
            "test_loss",
            "test_perplexity",
            "train_seconds",
            "embedding_train_seconds",
            "frozen",
        ],
        output_dir / "summary_table.md",
    )
    summary = {
        "quick": quick,
        "seed": config["seed"],
        "architecture": "lstm",
        "data_source": data_source,
        "token_counts": token_counts,
        "vocab_size": len(vocab.itos),
        "models": summaries,
        "coverage_file": "coverage.json",
    }
    save_json(summary, summary_path)
    print(f"Part II 完成：{output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.quick)
