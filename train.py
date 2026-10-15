"""Shared training, evaluation, timing, generation, and result utilities."""
from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(choice: str = "auto") -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def mode_path(base: str, quick: bool) -> Path:
    path = Path(base)
    return path / "quick" if quick else path


def save_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def log_data_summary(
    part: str, source, split_sizes: dict, vocab_size: int, unit: str
) -> None:
    print(f"[{part}] data source: {json.dumps(source, ensure_ascii=False)}")
    print(
        f"[{part}] {unit} counts: train={split_sizes.get('train', 0)}, "
        f"validation={split_sizes.get('validation', 0)}, test={split_sizes.get('test', 0)}"
    )
    print(f"[{part}] vocabulary size: {vocab_size}")


def hardware_info(device: torch.device) -> dict:
    name = platform.processor() or "unknown CPU"
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    return {
        "device": str(device),
        "name": name,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def parameter_counts(model: nn.Module) -> dict:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def _loss_sum(
    logits: torch.Tensor, targets: torch.Tensor, criterion: nn.CrossEntropyLoss
) -> torch.Tensor:
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def evaluate(model, loader, device, pad_id: int) -> dict:
    model.eval()
    # CHECK: reduction="sum" 配合 ignore_index 会完全排除 padding 的损失。
    # 最后除以真实 target token 数，避免不同 batch padding 比例影响平均值。
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss, total_tokens = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        total_loss += float(_loss_sum(model(inputs), targets, criterion).item())
        total_tokens += int(targets.ne(pad_id).sum().item())
    average = total_loss / max(1, total_tokens)
    # CHECK: perplexity 定义为测试集逐 token 平均交叉熵的指数。
    # 先在整个语料上累加 loss 再取指数，而不是平均各 batch 的 perplexity。
    return {"loss": average, "perplexity": math.exp(min(average, 50.0)), "tokens": total_tokens}


def train_model(
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    config: dict,
    checkpoint_path: Path,
    quick: bool = False,
) -> dict:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    epochs = config["train"]["quick_epochs"] if quick else config["train"]["epochs"]
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config["_pad_id"], reduction="sum")
    history = {"train_loss": [], "validation_loss": [], "epoch_seconds": []}
    best_loss = float("inf")
    total_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        synchronize(device)
        epoch_start = time.perf_counter()
        total_loss, total_tokens = 0.0, 0
        for inputs, targets in loaders["train"]:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = _loss_sum(model(inputs), targets, criterion)
            tokens = targets.ne(config["_pad_id"]).sum()
            loss = loss_sum / tokens.clamp_min(1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip"])
            optimizer.step()
            total_loss += float(loss_sum.detach().item())
            total_tokens += int(tokens.item())
        synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        validation = evaluate(model, loaders["validation"], device, config["_pad_id"])
        train_loss = total_loss / max(1, total_tokens)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation["loss"])
        history["epoch_seconds"].append(elapsed)
        print(
            f"epoch {epoch + 1}/{epochs}: train={train_loss:.4f}, "
            f"val={validation['loss']:.4f}, time={elapsed:.2f}s"
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_path)

    total_seconds = time.perf_counter() - total_start
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    history = saved["history"]
    history["total_train_seconds"] = total_seconds
    history["best_validation_loss"] = best_loss
    torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_path)
    return history


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    saved = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    model.to(device)
    return saved.get("history", {})


@torch.no_grad()
def inference_throughput(model, loader, device, pad_id: int, warmup_batches: int = 2) -> float:
    model.eval()
    batches = list(loader)
    if not batches:
        return 0.0
    for inputs, _ in batches[:warmup_batches]:
        model(inputs.to(device))
    synchronize(device)
    total_tokens = 0
    start = time.perf_counter()
    for inputs, targets in batches:
        model(inputs.to(device))
        total_tokens += int(targets.ne(pad_id).sum().item())
    synchronize(device)
    elapsed = time.perf_counter() - start
    return total_tokens / max(elapsed, 1e-9)


@torch.no_grad()
def generate_neural(
    model,
    prompt_ids: List[int],
    device: torch.device,
    max_context: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> List[int]:
    generator = None
    if device.type in {"cpu", "cuda"}:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        torch.manual_seed(seed)
    generated = list(prompt_ids)
    model.eval()
    for _ in range(max_new_tokens):
        context = generated[-max_context:]
        inputs = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1] / temperature
        k = min(top_k, logits.numel())
        values, indices = torch.topk(logits, k)
        probabilities = torch.softmax(values, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=generator)
        generated.append(int(indices[sampled].item()))
    return generated


def transformer_causality_test(model, vocab_size: int, device: torch.device) -> bool:
    model.eval()
    length, changed_at = min(12, model.max_length), 7
    original = torch.randint(1, vocab_size, (1, length), device=device)
    changed = original.clone()
    changed[0, changed_at] = (changed[0, changed_at] + 1) % vocab_size
    with torch.no_grad():
        before = model(original)[0, :changed_at]
        after = model(changed)[0, :changed_at]
    passed = bool(torch.allclose(before, after, atol=1e-6, rtol=1e-5))
    print(f"Transformer causality test: {'PASSED' if passed else 'FAILED'}")
    return passed


def plot_histories(histories: Dict[str, dict], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs, history["train_loss"], marker="o", label=f"{name} train")
        plt.plot(epochs, history["validation_loss"], marker="s", linestyle="--", label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_markdown_table(rows: List[dict], columns: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
