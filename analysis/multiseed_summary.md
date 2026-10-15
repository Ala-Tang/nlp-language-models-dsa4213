# 四种子汇总（20 epochs）

种子为 42、1、2、3。均值和标准差直接由四个 `summary.json` 计算；标准差为样本标准差（ddof=1）。

## Part I：测试 perplexity

| 模型 | seed 42 | seed 1 | seed 2 | seed 3 | 均值 ± 标准差 |
|---|---:|---:|---:|---:|---:|
| ngram | 264.6483 | 264.6483 | 264.6483 | 264.6483 | 264.6483 ± 无随机性 |
| rnn | 175.8897 | 177.2848 | 178.8812 | 176.0868 | 177.0356 ± 1.3762 |
| lstm | 154.7271 | 153.3955 | 155.7498 | 152.6938 | 154.1416 ± 1.3640 |
| transformer | 163.0557 | 162.2067 | 162.0160 | 162.5470 | 162.4563 ± 0.4559 |

来源：`results_e20/part1/summary.json`、`results_seed1/part1/summary.json`、`results_seed2/part1/summary.json`、`results_seed3/part1/summary.json`。

## Part II：测试 perplexity

| 设置 | seed 42 | seed 1 | seed 2 | seed 3 | 均值 ± 标准差 |
|---|---:|---:|---:|---:|---:|
| random_trainable | 154.7271 | 153.3955 | 155.7498 | 152.6938 | 154.1416 ± 1.3640 |
| word2vec_frozen | 146.5231 | 147.3072 | 145.9058 | 146.6443 | 146.5951 ± 0.5744 |
| glove_frozen | 147.1188 | 143.1957 | 144.8447 | 149.0574 | 146.0542 ± 2.5682 |

来源：`results_e20/part2/summary.json`、`results_seed1/part2/summary.json`、`results_seed2/part2/summary.json`、`results_seed3/part2/summary.json`。

## Part III：SST-2

| 表示 | 指标 | seed 42 | seed 1 | seed 2 | seed 3 | 均值 ± 标准差 |
|---|---|---:|---:|---:|---:|---:|
| lstm_mean | accuracy | 0.6456 | 0.6193 | 0.6376 | 0.6353 | 0.6345 ± 0.0111 |
| lstm_mean | macro-F1 | 0.6442 | 0.6192 | 0.6373 | 0.6335 | 0.6336 ± 0.0105 |
| transformer_mean | accuracy | 0.6640 | 0.6261 | 0.6296 | 0.6468 | 0.6416 ± 0.0174 |
| transformer_mean | macro-F1 | 0.6625 | 0.6245 | 0.6291 | 0.6457 | 0.6405 ± 0.0173 |
| lstm_last | accuracy | 0.6261 | 0.5963 | 0.6032 | 0.6124 | 0.6095 ± 0.0129 |
| lstm_last | macro-F1 | 0.6234 | 0.5945 | 0.6017 | 0.6068 | 0.6066 ± 0.0123 |

来源：`results_e20/part3/summary.json`、`results_seed1/part3/summary.json`、`results_seed2/part3/summary.json`、`results_seed3/part3/summary.json`。

### 每个种子的排名

- seed 42：Part I `lstm < transformer < rnn < ngram`；Part II `word2vec_frozen < glove_frozen < random_trainable`；Part III accuracy `transformer_mean > lstm_mean > lstm_last`。
- seed 1：Part I `lstm < transformer < rnn < ngram`；Part II `glove_frozen < word2vec_frozen < random_trainable`；Part III accuracy `transformer_mean > lstm_mean > lstm_last`。
- seed 2：Part I `lstm < transformer < rnn < ngram`；Part II `glove_frozen < word2vec_frozen < random_trainable`；Part III accuracy `lstm_mean > transformer_mean > lstm_last`。
- seed 3：Part I `lstm < transformer < rnn < ngram`；Part II `word2vec_frozen < glove_frozen < random_trainable`；Part III accuracy `transformer_mean > lstm_mean > lstm_last`。

完整精度、各表示选择的 C、两两均值差与组合标准差比较见 `analysis/multiseed_summary.json`。

### 稳定性结论

- Part I 四个种子的完整排名一致。LSTM 与 Transformer 的平均 PPL 差为 8.3148，大于组合标准差 1.4382。
- Part II 的 Word2Vec 与 GloVe 排名随种子互换；平均 PPL 差为 0.5409，小于组合标准差 2.6316。
- Part III 的 Transformer mean 与 LSTM mean accuracy 排名在 seed 2 反转；平均差为 0.0072，小于组合标准差 0.0206。两种 mean 表示相对 LSTM last 的平均差均大于对应组合标准差。

## SST-2 多数类基线

测试集标签 0/1 分别为 428/444，多数类为 1，全部预测为多数类的 accuracy 为 0.5092。四套 `part3/errors.csv` 的测试标签顺序已核对为一致；来源路径见 JSON。
