"""Part I: train and compare n-gram, RNN, LSTM, and Transformer LMs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data import prepare_lm_data
from models import build_model
from ngram import InterpolatedTrigram
from train import (
    generate_neural,
    hardware_info,
    inference_throughput,
    evaluate,
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
    transformer_causality_test,
    write_markdown_table,
)


def run(quick: bool = False) -> dict:
    config = load_config()
    set_seed(config["seed"])
    device = select_device(config["device"])
    output_dir = mode_path(str(Path(config["paths"]["results"]) / "part1"), quick)
    checkpoint_dir = mode_path(str(Path(config["paths"]["checkpoints"]) / "part1"), quick)
    summary_path = output_dir / "summary.json"
    required = [
        summary_path,
        output_dir / "summary_table.md",
        output_dir / "loss_curves.png",
        output_dir / "generations.md",
        output_dir / "vocab.json",
        checkpoint_dir / "rnn.pt",
        checkpoint_dir / "lstm.pt",
        checkpoint_dir / "transformer.pt",
    ]
    if all(path.exists() for path in required):
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if "data_source" in cached and "token_counts" in cached:
            log_data_summary(
                "Part I", cached["data_source"], cached["token_counts"], cached["vocab_size"], "token"
            )
            print(f"Part I 已完成，跳过：{output_dir}")
            return cached

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lines, streams, vocab, loaders, data_source = prepare_lm_data(config, quick)
    token_counts = {split: len(tokens) for split, tokens in streams.items()}
    log_data_summary("Part I", data_source, token_counts, len(vocab.itos), "token")
    vocab.save(output_dir / "vocab.json")
    config["_pad_id"] = vocab.pad_id
    generation_cfg = config["train"]["generation"]
    prompts = generation_cfg["prompts"]
    generations = {prompt: {} for prompt in prompts}
    summary = {
        "quick": quick,
        "seed": config["seed"],
        "corpus": "Salesforce/wikitext:wikitext-2-v1",
        "data_source": data_source,
        "token_counts": token_counts,
        "vocab_size": len(vocab.itos),
        "hardware": hardware_info(device),
        "models": {},
    }

    ngram = InterpolatedTrigram(len(vocab.itos), config["ngram"]["min_probability"])
    start = time.perf_counter()
    ngram.fit(streams["train"])
    validation = ngram.tune(streams["validation"], config["ngram"]["grid_step"])
    ngram_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    ngram_test = ngram.evaluate(streams["test"])
    ngram_inference_seconds = time.perf_counter() - start
    summary["models"]["ngram"] = {
        "validation": validation,
        "test": ngram_test,
        "parameters": 0,
        "count_entries": len(ngram.unigrams) + len(ngram.bigrams) + len(ngram.trigrams),
        "train_seconds": ngram_train_seconds,
        "inference_tokens_per_second": ngram_test["tokens"] / max(ngram_inference_seconds, 1e-9),
    }
    for index, prompt in enumerate(prompts):
        prompt_ids = vocab.encode(prompt.lower().split())
        ids = ngram.generate(
            prompt_ids,
            generation_cfg["max_new_tokens"],
            generation_cfg["temperature"],
            generation_cfg["top_k"],
            config["seed"] + index,
        )
        generations[prompt]["ngram"] = " ".join(vocab.decode(ids))

    histories = {}
    for name in ("rnn", "lstm", "transformer"):
        set_seed(config["seed"])
        model = build_model(name, len(vocab.itos), vocab.pad_id, config)
        checkpoint = checkpoint_dir / f"{name}.pt"
        if checkpoint.exists():
            print(f"加载已有 checkpoint：{checkpoint}")
            history = load_checkpoint(model, checkpoint, device)
        else:
            history = train_model(model, loaders, device, config, checkpoint, quick)
        histories[name] = history
        test_metrics = evaluate(model, loaders["test"], device, vocab.pad_id)
        model_summary = {
            "test": test_metrics,
            "parameters": parameter_counts(model),
            "train_seconds": history.get("total_train_seconds"),
            "epoch_seconds": history.get("epoch_seconds", []),
            "best_validation_loss": history.get("best_validation_loss"),
            "inference_tokens_per_second": inference_throughput(
                model,
                loaders["test"],
                device,
                vocab.pad_id,
                config["train"]["warmup_batches"],
            ),
        }
        if name == "transformer":
            passed = transformer_causality_test(model, len(vocab.itos), device)
            model_summary["causality_test"] = "PASSED" if passed else "FAILED"
            if not passed:
                raise AssertionError("Transformer 因果性测试失败")
        summary["models"][name] = model_summary
        for index, prompt in enumerate(prompts):
            prompt_ids = vocab.encode(prompt.lower().split())
            ids = generate_neural(
                model,
                prompt_ids,
                device,
                config["data"]["context_length"],
                generation_cfg["max_new_tokens"],
                generation_cfg["temperature"],
                generation_cfg["top_k"],
                config["seed"] + index,
            )
            generations[prompt][name] = " ".join(vocab.decode(ids))

    plot_histories(histories, output_dir / "loss_curves.png", "Part I learning curves")
    rows = []
    for name, values in summary["models"].items():
        params = values.get("parameters", 0)
        rows.append(
            {
                "model": name,
                "test_loss": values["test"]["loss"],
                "test_perplexity": values["test"]["perplexity"],
                "parameters": params.get("total", 0) if isinstance(params, dict) else params,
                "train_seconds": values.get("train_seconds", ""),
                "tokens_per_second": values["inference_tokens_per_second"],
            }
        )
    write_markdown_table(
        rows,
        [
            "model",
            "test_loss",
            "test_perplexity",
            "parameters",
            "train_seconds",
            "tokens_per_second",
        ],
        output_dir / "summary_table.md",
    )
    generation_lines = ["# Part I generations", ""]
    for prompt, outputs in generations.items():
        generation_lines.extend([f"## Prompt: `{prompt}`", ""])
        for name, text in outputs.items():
            generation_lines.extend([f"### {name}", text, ""])
    (output_dir / "generations.md").write_text("\n".join(generation_lines), encoding="utf-8")
    save_json(summary, summary_path)
    print(f"Part I 完成：{output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.quick)
