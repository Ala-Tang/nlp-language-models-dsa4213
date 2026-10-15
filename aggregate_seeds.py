"""Aggregate the four completed 20-epoch runs (seeds 42, 1, 2, and 3)."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


RUNS = {
    "42": Path("results_e20"),
    "1": Path("results_seed1"),
    "2": Path("results_seed2"),
    "3": Path("results_seed3"),
}
PART1_MODELS = ("ngram", "rnn", "lstm", "transformer")
PART2_SETTINGS = ("random_trainable", "word2vec_frozen", "glove_frozen")
PART3_REPRESENTATIONS = ("lstm_mean", "transformer_mean", "lstm_last")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: Iterable[float]) -> dict:
    numbers = [float(value) for value in values]
    return {
        "values": numbers,
        "mean": statistics.mean(numbers),
        "sample_std": statistics.stdev(numbers),
        "n": len(numbers),
    }


def rank(values: Dict[str, float], lower_is_better: bool) -> List[str]:
    return sorted(values, key=values.get, reverse=not lower_is_better)


def pairwise(mean_a: float, std_a: float, mean_b: float, std_b: float) -> dict:
    gap = abs(mean_a - mean_b)
    combined = math.sqrt(std_a**2 + std_b**2)
    return {
        "absolute_mean_gap": gap,
        "sample_std_a": std_a,
        "sample_std_b": std_b,
        "combined_sample_std": combined,
        "gap_exceeds_each_std": gap > std_a and gap > std_b,
        "gap_exceeds_combined_std": gap > combined,
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    loaded = {}
    sources = {}
    for seed, root in RUNS.items():
        paths = {
            "part1": root / "part1" / "summary.json",
            "part2": root / "part2" / "summary.json",
            "part3": root / "part3" / "summary.json",
        }
        loaded[seed] = {part: load_json(path) for part, path in paths.items()}
        sources[seed] = {part: path.as_posix() for part, path in paths.items()}
        actual_seed = str(loaded[seed]["part1"]["seed"])
        if actual_seed != seed:
            raise ValueError(f"{paths['part1']} records seed {actual_seed}, expected {seed}")

    output = {
        "seeds": [42, 1, 2, 3],
        "standard_deviation": "sample (ddof=1)",
        "sources": sources,
        "part1_test_perplexity": {},
        "part2_test_perplexity": {},
        "part3": {},
        "rankings": {"part1": {}, "part2": {}, "part3_accuracy": {}, "part3_macro_f1": {}},
        "pairwise_gap_checks": {},
    }

    for model in PART1_MODELS:
        per_seed = {
            seed: loaded[seed]["part1"]["models"][model]["test"]["perplexity"]
            for seed in RUNS
        }
        output["part1_test_perplexity"][model] = {
            "per_seed": per_seed,
            **summarize(per_seed.values()),
            "stochastic": model != "ngram",
        }

    for setting in PART2_SETTINGS:
        per_seed = {
            seed: loaded[seed]["part2"]["models"][setting]["test"]["perplexity"]
            for seed in RUNS
        }
        output["part2_test_perplexity"][setting] = {
            "per_seed": per_seed,
            **summarize(per_seed.values()),
        }

    for representation in PART3_REPRESENTATIONS:
        output["part3"][representation] = {}
        for metric in ("test_accuracy", "test_macro_f1"):
            per_seed = {
                seed: loaded[seed]["part3"]["metrics"][representation][metric]
                for seed in RUNS
            }
            output["part3"][representation][metric] = {
                "per_seed": per_seed,
                **summarize(per_seed.values()),
            }
        output["part3"][representation]["best_c"] = {
            seed: loaded[seed]["part3"]["metrics"][representation]["best_c"]
            for seed in RUNS
        }

    for seed in RUNS:
        p1 = {
            model: output["part1_test_perplexity"][model]["per_seed"][seed]
            for model in PART1_MODELS
        }
        p2 = {
            setting: output["part2_test_perplexity"][setting]["per_seed"][seed]
            for setting in PART2_SETTINGS
        }
        p3_acc = {
            representation: output["part3"][representation]["test_accuracy"]["per_seed"][seed]
            for representation in PART3_REPRESENTATIONS
        }
        p3_f1 = {
            representation: output["part3"][representation]["test_macro_f1"]["per_seed"][seed]
            for representation in PART3_REPRESENTATIONS
        }
        output["rankings"]["part1"][seed] = rank(p1, lower_is_better=True)
        output["rankings"]["part2"][seed] = rank(p2, lower_is_better=True)
        output["rankings"]["part3_accuracy"][seed] = rank(p3_acc, lower_is_better=False)
        output["rankings"]["part3_macro_f1"][seed] = rank(p3_f1, lower_is_better=False)

    p1_lstm = output["part1_test_perplexity"]["lstm"]
    p1_transformer = output["part1_test_perplexity"]["transformer"]
    output["pairwise_gap_checks"]["part1_lstm_vs_transformer"] = pairwise(
        p1_lstm["mean"],
        p1_lstm["sample_std"],
        p1_transformer["mean"],
        p1_transformer["sample_std"],
    )
    p2_w2v = output["part2_test_perplexity"]["word2vec_frozen"]
    p2_glove = output["part2_test_perplexity"]["glove_frozen"]
    output["pairwise_gap_checks"]["part2_word2vec_vs_glove"] = pairwise(
        p2_w2v["mean"], p2_w2v["sample_std"], p2_glove["mean"], p2_glove["sample_std"]
    )
    for metric in ("test_accuracy", "test_macro_f1"):
        for left, right in (
            ("transformer_mean", "lstm_mean"),
            ("lstm_mean", "lstm_last"),
            ("transformer_mean", "lstm_last"),
        ):
            left_values = output["part3"][left][metric]
            right_values = output["part3"][right][metric]
            output["pairwise_gap_checks"][f"part3_{metric}_{left}_vs_{right}"] = pairwise(
                left_values["mean"],
                left_values["sample_std"],
                right_values["mean"],
                right_values["sample_std"],
            )

    error_sources = {
        seed: root / "part3" / "errors.csv" for seed, root in RUNS.items()
    }
    labels_by_seed = {}
    for seed, error_path in error_sources.items():
        with error_path.open("r", encoding="utf-8", newline="") as handle:
            labels_by_seed[seed] = [row["true_label"] for row in csv.DictReader(handle)]
    if any(labels != labels_by_seed["42"] for labels in labels_by_seed.values()):
        raise ValueError("SST-2 test labels differ across seed result folders")
    label_counts = {
        "0": labels_by_seed["42"].count("0"),
        "1": labels_by_seed["42"].count("1"),
    }
    total = sum(label_counts.values())
    output["sst2_test_majority_baseline"] = {
        "sources": {seed: path.as_posix() for seed, path in error_sources.items()},
        "labels_identical_across_seeds": True,
        "label_counts": label_counts,
        "total": total,
        "majority_label": max(label_counts, key=label_counts.get),
        "accuracy": max(label_counts.values()) / total,
    }

    output_dir = Path("analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "multiseed_summary.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 四种子汇总（20 epochs）",
        "",
        "种子为 42、1、2、3。均值和标准差直接由四个 `summary.json` 计算；标准差为样本标准差（ddof=1）。",
        "",
        "## Part I：测试 perplexity",
        "",
        "| 模型 | seed 42 | seed 1 | seed 2 | seed 3 | 均值 ± 标准差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in PART1_MODELS:
        item = output["part1_test_perplexity"][model]
        std_text = "无随机性" if model == "ngram" else fmt(item["sample_std"])
        lines.append(
            f"| {model} | {fmt(item['per_seed']['42'])} | {fmt(item['per_seed']['1'])} | "
            f"{fmt(item['per_seed']['2'])} | {fmt(item['per_seed']['3'])} | "
            f"{fmt(item['mean'])} ± {std_text} |"
        )
    lines.extend(
        [
            "",
            "来源：`results_e20/part1/summary.json`、`results_seed1/part1/summary.json`、"
            "`results_seed2/part1/summary.json`、`results_seed3/part1/summary.json`。",
            "",
            "## Part II：测试 perplexity",
            "",
            "| 设置 | seed 42 | seed 1 | seed 2 | seed 3 | 均值 ± 标准差 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in PART2_SETTINGS:
        item = output["part2_test_perplexity"][setting]
        lines.append(
            f"| {setting} | {fmt(item['per_seed']['42'])} | {fmt(item['per_seed']['1'])} | "
            f"{fmt(item['per_seed']['2'])} | {fmt(item['per_seed']['3'])} | "
            f"{fmt(item['mean'])} ± {fmt(item['sample_std'])} |"
        )
    lines.extend(
        [
            "",
            "来源：`results_e20/part2/summary.json`、`results_seed1/part2/summary.json`、"
            "`results_seed2/part2/summary.json`、`results_seed3/part2/summary.json`。",
            "",
            "## Part III：SST-2",
            "",
            "| 表示 | 指标 | seed 42 | seed 1 | seed 2 | seed 3 | 均值 ± 标准差 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for representation in PART3_REPRESENTATIONS:
        for metric, label in (("test_accuracy", "accuracy"), ("test_macro_f1", "macro-F1")):
            item = output["part3"][representation][metric]
            lines.append(
                f"| {representation} | {label} | {fmt(item['per_seed']['42'])} | "
                f"{fmt(item['per_seed']['1'])} | {fmt(item['per_seed']['2'])} | "
                f"{fmt(item['per_seed']['3'])} | {fmt(item['mean'])} ± "
                f"{fmt(item['sample_std'])} |"
            )
    lines.extend(
        [
            "",
            "来源：`results_e20/part3/summary.json`、`results_seed1/part3/summary.json`、"
            "`results_seed2/part3/summary.json`、`results_seed3/part3/summary.json`。",
            "",
            "### 每个种子的排名",
            "",
        ]
    )
    for seed in RUNS:
        lines.append(
            f"- seed {seed}：Part I `{' < '.join(output['rankings']['part1'][seed])}`；"
            f"Part II `{' < '.join(output['rankings']['part2'][seed])}`；"
            f"Part III accuracy `{' > '.join(output['rankings']['part3_accuracy'][seed])}`。"
        )
    lines.extend(
        [
            "",
            "完整精度、各表示选择的 C、两两均值差与组合标准差比较见 "
            "`analysis/multiseed_summary.json`。",
            "",
            "### 稳定性结论",
            "",
            "- Part I 四个种子的完整排名一致。LSTM 与 Transformer 的平均 PPL 差为 "
            f"{fmt(output['pairwise_gap_checks']['part1_lstm_vs_transformer']['absolute_mean_gap'])}，"
            "大于组合标准差 "
            f"{fmt(output['pairwise_gap_checks']['part1_lstm_vs_transformer']['combined_sample_std'])}。",
            "- Part II 的 Word2Vec 与 GloVe 排名随种子互换；平均 PPL 差为 "
            f"{fmt(output['pairwise_gap_checks']['part2_word2vec_vs_glove']['absolute_mean_gap'])}，"
            "小于组合标准差 "
            f"{fmt(output['pairwise_gap_checks']['part2_word2vec_vs_glove']['combined_sample_std'])}。",
            "- Part III 的 Transformer mean 与 LSTM mean accuracy 排名在 seed 2 反转；"
            "平均差为 "
            f"{fmt(output['pairwise_gap_checks']['part3_test_accuracy_transformer_mean_vs_lstm_mean']['absolute_mean_gap'])}，"
            "小于组合标准差 "
            f"{fmt(output['pairwise_gap_checks']['part3_test_accuracy_transformer_mean_vs_lstm_mean']['combined_sample_std'])}。"
            "两种 mean 表示相对 LSTM last 的平均差均大于对应组合标准差。",
            "",
            "## SST-2 多数类基线",
            "",
            f"测试集标签 0/1 分别为 {label_counts['0']}/{label_counts['1']}，多数类为 "
            f"{output['sst2_test_majority_baseline']['majority_label']}，全部预测为多数类的 "
            f"accuracy 为 {fmt(output['sst2_test_majority_baseline']['accuracy'])}。"
            "四套 `part3/errors.csv` 的测试标签顺序已核对为一致；来源路径见 JSON。",
            "",
        ]
    )
    (output_dir / "multiseed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path} and {output_dir / 'multiseed_summary.md'}")


if __name__ == "__main__":
    main()
