"""Shared data loading and preprocessing for all three assignment parts."""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

SPECIAL_TOKENS = ("<pad>", "<unk>", "<eos>")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize_sst(text: str) -> List[str]:
    """Lowercase regular-expression tokenization used for SST-2."""
    return TOKEN_RE.findall(text.lower())


@dataclass
class Vocabulary:
    itos: List[str]

    def __post_init__(self) -> None:
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    def encode(self, tokens: Iterable[str]) -> List[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self.itos[int(index)] for index in ids]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def build_vocabulary(
    train_lines: Sequence[str], min_freq: int = 2, max_vocab: int = 30000
) -> Vocabulary:
    counter: Counter = Counter()
    for line in train_lines:
        if line.strip():
            counter.update(line.lower().split())
    words = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
    kept = [word for word, count in words if count >= min_freq and word not in SPECIAL_TOKENS]
    return Vocabulary(list(SPECIAL_TOKENS) + kept[: max_vocab - len(SPECIAL_TOKENS)])


def lines_to_ids(lines: Sequence[str], vocab: Vocabulary) -> List[int]:
    ids: List[int] = []
    for line in lines:
        tokens = line.lower().split()
        if tokens:
            ids.extend(vocab.encode(tokens))
            ids.append(vocab.eos_id)
    return ids


class LanguageModelDataset(Dataset):
    def __init__(self, token_ids: Sequence[int], context_length: int, pad_id: int):
        # CHECK: 每个样本先取 context_length + 1 个连续 token。
        # 输入去掉最后一个，目标去掉第一个，因此每个目标恰好比输入错开一位。
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        step = context_length
        for start in range(0, max(1, len(token_ids) - 1), step):
            chunk = list(token_ids[start : start + context_length + 1])
            if len(chunk) < 2:
                continue
            chunk += [pad_id] * (context_length + 1 - len(chunk))
            self.samples.append(
                (torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long))
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


def _quick_wikitext() -> Dict[str, List[str]]:
    base = [
        "The history of natural language processing began in the nineteen fifties .",
        "A language model assigns probabilities to sequences of words .",
        "Neural networks learn representations from examples in a training corpus .",
        "The recurrent model processes one token at a time and keeps a hidden state .",
        "Attention allows a model to combine information from earlier positions .",
        "Singapore is an island country and city state in Southeast Asia .",
        "Scientists compare models using held out validation and test data .",
        "The university library contains books journals and digital collections .",
    ]
    return {
        "train": base * 20,
        "validation": base[1:] * 5,
        "test": list(reversed(base)) * 5,
    }


def load_wikitext_lines(
    config: dict, quick: bool = False
) -> Tuple[Dict[str, List[str]], str]:
    try:
        from datasets import load_dataset

        dataset = load_dataset(config["dataset"], config["subset"])
        result = {split: list(dataset[split]["text"]) for split in ("train", "validation", "test")}
        source = f"huggingface:{config['dataset']}:{config['subset']}"
    except Exception as exc:
        if not quick:
            raise RuntimeError(
                "无法加载 HuggingFace Salesforce/wikitext（wikitext-2-v1）；"
                "非 quick 模式禁止使用后备数据。"
            ) from exc
        print(f"[quick] WikiText 不可用，使用内置微型语料检查流程：{exc}")
        result = _quick_wikitext()
        source = "quick_fallback:built_in_wikitext"
    if quick:
        limits = {
            "train": config["quick_train_lines"],
            "validation": config["quick_val_lines"],
            "test": config["quick_test_lines"],
        }
        result = {split: lines[: limits[split]] for split, lines in result.items()}
    return result, source


def prepare_lm_data(config: dict, quick: bool = False):
    lines, source = load_wikitext_lines(config["data"], quick)
    vocab = build_vocabulary(
        lines["train"], config["data"]["min_freq"], config["data"]["max_vocab"]
    )
    streams = {split: lines_to_ids(split_lines, vocab) for split, split_lines in lines.items()}
    loaders = {}
    for split, ids in streams.items():
        dataset = LanguageModelDataset(ids, config["data"]["context_length"], vocab.pad_id)
        generator = torch.Generator().manual_seed(config["seed"])
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=config["data"]["num_workers"],
            generator=generator,
        )
    return lines, streams, vocab, loaders, source


def _quick_sst2() -> Dict[str, List[dict]]:
    positive = ["a warm and delightful film", "excellent acting and a moving story"]
    negative = ["a dull and predictable film", "poor acting and a tedious story"]
    rows = [{"sentence": text, "label": label} for label, texts in enumerate((negative, positive)) for text in texts]
    train = rows * 100
    random.Random(42).shuffle(train)
    return {"train": train[:280], "validation": rows * 25}


def load_sst2_splits(
    config: dict, quick: bool = False
) -> Tuple[Dict[str, List[dict]], str]:
    try:
        from datasets import load_dataset

        raw = load_dataset("nyu-mll/glue", "sst2")
        train_rows = [dict(row) for row in raw["train"]]
        test_rows = [dict(row) for row in raw["validation"]]
        source = "huggingface:nyu-mll/glue:sst2"
    except Exception as exc:
        if not quick:
            raise RuntimeError(
                "无法加载 HuggingFace nyu-mll/glue（sst2）；"
                "非 quick 模式禁止使用后备数据。"
            ) from exc
        print(f"[quick] SST-2 不可用，使用内置微型数据检查流程：{exc}")
        fallback = _quick_sst2()
        train_rows, test_rows = fallback["train"], fallback["validation"]
        source = "quick_fallback:built_in_sst2"

    rng = random.Random(config["seed"])
    rng.shuffle(train_rows)
    part = config["part3"]
    train_size = part["quick_train_size"] if quick else part["train_size"]
    val_size = part["quick_validation_size"] if quick else part["validation_size"]
    test_size = part["quick_test_size"] if quick else len(test_rows)
    if len(train_rows) < train_size + val_size:
        repeats = (train_size + val_size + len(train_rows) - 1) // len(train_rows)
        train_rows = (train_rows * repeats)[: train_size + val_size]
    return (
        {
            "train": train_rows[:train_size],
            "validation": train_rows[train_size : train_size + val_size],
            "test": test_rows[:test_size],
        },
        source,
    )


class ClassificationDataset(Dataset):
    def __init__(self, rows: Sequence[dict], vocab: Vocabulary, max_length: int):
        self.items = []
        self.unknown = 0
        self.total = 0
        for row in rows:
            tokens = tokenize_sst(row["sentence"])[:max_length]
            ids = vocab.encode(tokens)
            self.unknown += sum(index == vocab.unk_id for index in ids)
            self.total += len(ids)
            ids = ids or [vocab.unk_id]
            mask = [1] * len(ids)
            padding = max_length - len(ids)
            self.items.append(
                (
                    torch.tensor(ids + [vocab.pad_id] * padding, dtype=torch.long),
                    torch.tensor(mask + [0] * padding, dtype=torch.bool),
                    int(row["label"]),
                    row["sentence"],
                )
            )

    @property
    def oov_ratio(self) -> float:
        return self.unknown / max(1, self.total)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]
