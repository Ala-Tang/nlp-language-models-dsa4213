"""Interpolated trigram language model."""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple


class InterpolatedTrigram:
    def __init__(self, vocab_size: int, min_probability: float = 1e-12):
        self.vocab_size = vocab_size
        self.min_probability = min_probability
        self.unigrams: Counter = Counter()
        self.bigrams: Counter = Counter()
        self.trigrams: Counter = Counter()
        self.bigram_contexts: Counter = Counter()
        self.trigram_contexts: Counter = Counter()
        self.total = 0
        self.weights = (0.25, 0.25, 0.25, 0.25)

    def fit(self, tokens: Sequence[int]) -> None:
        self.unigrams.update(tokens)
        self.total = len(tokens)
        for previous, token in zip(tokens, tokens[1:]):
            self.bigrams[(previous, token)] += 1
            self.bigram_contexts[previous] += 1
        for first, second, token in zip(tokens, tokens[1:], tokens[2:]):
            self.trigrams[(first, second, token)] += 1
            self.trigram_contexts[(first, second)] += 1

    def components(self, context: Sequence[int], token: int) -> Tuple[float, float, float, float]:
        unigram = self.unigrams[token] / max(1, self.total)
        bigram = 0.0
        trigram = 0.0
        if context:
            denominator = self.bigram_contexts[context[-1]]
            bigram = self.bigrams[(context[-1], token)] / denominator if denominator else 0.0
        if len(context) >= 2:
            key = (context[-2], context[-1])
            denominator = self.trigram_contexts[key]
            trigram = self.trigrams[(key[0], key[1], token)] / denominator if denominator else 0.0
        return trigram, bigram, unigram, 1.0 / self.vocab_size

    def probability(self, context: Sequence[int], token: int, weights=None) -> float:
        weights = weights or self.weights
        value = sum(weight * part for weight, part in zip(weights, self.components(context, token)))
        return max(value, self.min_probability)

    def evaluate(self, tokens: Sequence[int], weights=None) -> dict:
        negative_log_likelihood = 0.0
        # 与神经模型一致，语料第一个 token 只作为上下文，不作为无条件预测目标。
        for index, token in enumerate(tokens[1:], start=1):
            negative_log_likelihood -= math.log(self.probability(tokens[max(0, index - 2) : index], token, weights))
        target_count = max(0, len(tokens) - 1)
        loss = negative_log_likelihood / max(1, target_count)
        return {"loss": loss, "perplexity": math.exp(min(loss, 50.0)), "tokens": target_count}

    def tune(self, validation_tokens: Sequence[int], step: float = 0.25) -> dict:
        units = round(1.0 / step)
        candidates = []
        for a in range(units + 1):
            for b in range(units - a + 1):
                for c in range(units - a - b + 1):
                    d = units - a - b - c
                    candidates.append(tuple(value / units for value in (a, b, c, d)))
        best_metrics = {"loss": float("inf")}
        for weights in candidates:
            metrics = self.evaluate(validation_tokens, weights)
            if metrics["loss"] < best_metrics["loss"]:
                best_metrics = metrics
                self.weights = weights
        return {"weights": self.weights, **best_metrics}

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        seed: int,
    ) -> List[int]:
        rng = random.Random(seed)
        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            context = generated[-2:]
            scores = [
                self.probability(context, token) ** (1.0 / temperature)
                for token in range(self.vocab_size)
            ]
            candidates = sorted(range(self.vocab_size), key=scores.__getitem__, reverse=True)[
                : min(top_k, self.vocab_size)
            ]
            candidate_scores = [scores[token] for token in candidates]
            generated.append(rng.choices(candidates, weights=candidate_scores, k=1)[0])
        return generated
