# 实验结果总览（seed 42）

本文件只汇总正式结果：`results_e20/` 为主实验（20 epochs），`results_e8/` 为补充实验（8 epochs）。`results_e8/*/quick/` 是流程检查产物，不用于任何结论。所有“最终 loss”均指日志中的最后一轮，而测试指标来自最佳验证 checkpoint。差距统一定义为 `val loss - train loss`。

## 数据与环境

- WikiText-2 训练、验证、测试流分别含 2,075,677、216,347、244,102 个 token，训练词表为 28,913；来源：`results_e20/part1/summary.json`。
- SST-2 使用 10,000/2,000/872 个训练/验证/测试样本；来源：`results_e20/part3/summary.json`。
- 主实验硬件为 Tesla T4，Python 3.13.15，PyTorch 2.11.0+cu130，Linux；来源：`results_e20/part1/summary.json`。

## Part I：模型结构比较

### 20-epoch 主结果

| 模型 | 测试 perplexity | 参数量 | 训练时间 | 推理吞吐量（token/s） | 来源 |
|---|---:|---:|---:|---:|---|
| trigram | 264.6483 | 0（计数表 1,993,038 项） | 26.36 s | 351,650 | `results_e20/part1/summary.json` |
| RNN | 175.8897 | 10,545,173 | 843.26 s | 191,976 | `results_e20/part1/summary.json` |
| LSTM | **154.7271** | 11,214,869 | 920.69 s | 166,892 | `results_e20/part1/summary.json` |
| Transformer | 163.0557 | 7,237,013 | 637.81 s | 290,536 | `results_e20/part1/summary.json` |

神经模型按测试 perplexity 的排名为 LSTM < Transformer < RNN。Transformer 参数最少、训练最快且神经模型中吞吐量最高，但 perplexity 略高于 LSTM。这里“结构导致差异”只能作为可能解释，因为参数量也不同。

### 学习过程

| 模型 | 最佳 epoch / val loss | 最后一轮 train | 最后一轮 val | 最终差距（val-train） | 来源 |
|---|---:|---:|---:|---:|---|
| RNN | 13 / 5.2463 | 4.4541 | 5.2715 | 0.8174 | `results_e20/part1/log.txt` |
| LSTM | 20 / 5.1091 | 4.5723 | 5.1091 | 0.5368 | `results_e20/part1/log.txt` |
| Transformer | 11 / 5.1609 | 4.1414 | 5.2029 | 1.0615 | `results_e20/part1/log.txt` |

观察：RNN 和 Transformer 的验证 loss 分别在第 13、11 轮后不再持续改善，而训练 loss 继续下降；Transformer 的最终差距最大。LSTM 到第 20 轮仍取得新的最佳验证 loss，因此不能断言它已完全收敛。

### 8 vs 20 epochs

| 预算 | 测试 perplexity 排名（低到高） | 来源 |
|---|---|---|
| 8 epochs | Transformer 164.5489 < RNN 180.6489 < LSTM 211.2915 < trigram 264.6483 | `results_e8/part1/summary.json` |
| 20 epochs | LSTM 154.7271 < Transformer 163.0557 < RNN 175.8897 < trigram 264.6483 | `results_e20/part1/summary.json` |

排名随训练预算变化。尤其 LSTM 从 8 轮的最差神经模型变为 20 轮的最佳模型；这与其验证 loss 在第 20 轮仍改善的轨迹一致，但不证明更长训练对任何 LSTM 都有同样效果。

8-epoch 的最终学习状态为：RNN 第 8 轮最佳，train/val=4.9650/5.2776，差距 0.3126；LSTM 第 8 轮最佳，5.5185/5.4208，差距 -0.0977；Transformer 第 8 轮最佳，4.6603/5.1726，差距 0.5123。来源：`results_e8/part1/log.txt`。

生成样例中，trigram 对 “the meaning of life” 产生连续 `= = = =`，并较频繁跳入标题标记；三个神经模型能形成更长的局部句法片段，但主题一致性仍弱。例如 Transformer 将 “once upon a time” 延续为飓风条目。来源：`results_e20/part1/generations.md`。这些是少量定性样例，不能替代整体指标。

## Part II：embedding 设置

### 20-epoch 主结果

| 设置 | 测试 perplexity | 最佳 epoch / val | 最终 train/val | 差距 | LM 训练时间 | 来源 |
|---|---:|---:|---:|---:|---:|---|
| 随机可训练 | 154.7271 | 20 / 5.1091 | 4.5723 / 5.1091 | 0.5368 | 920.69 s | `results_e20/part2/summary.json`; `results_e20/part1/log.txt` |
| Word2Vec 冻结 | **146.5231** | 20 / 5.0478 | 4.5476 / 5.0478 | 0.5002 | 907.08 s | `results_e20/part2/summary.json`; `results_e20/part2/log.txt` |
| GloVe 冻结 | 147.1188 | 20 / 5.0503 | 4.6827 / 5.0503 | **0.3676** | 904.14 s | `results_e20/part2/summary.json`; `results_e20/part2/log.txt` |

- GloVe 覆盖 27,809/28,913 个词，即 96.1816%，OOV 1,104；来源：`results_e20/part2/coverage.json`。
- 自训练 Word2Vec 覆盖 28,912/28,913 个词，即 99.9965%，单独训练时间 64.9973 s；来源：`results_e20/part2/coverage.json` 与 `results_e20/part2/summary.json`。
- 两种冻结设置的日志冻结测试均为 `PASSED`；来源：`results_e20/part2/log.txt`。

观察：两个冻结设置的 perplexity 都低于随机可训练设置，且 Word2Vec 与 GloVe 只相差 0.5957 perplexity。GloVe 的最终 train/val 差距最小。一个可能解释是冻结表限制了可训练自由度，产生正则化效果；另一个可能解释是外部 GloVe 数据带来有用先验。然而，自训练 Word2Vec 略优于 GloVe，说明“外部数据更多”在本实验中并未自动转化为最佳 perplexity。

8 epochs 时排序为 Word2Vec 180.8879 < 随机可训练 211.2915 < GloVe 229.7323；来源：`results_e8/part2/summary.json`。到 20 epochs，三个设置都继续改善，GloVe 的变化尤其大，因此短预算会改变 embedding ablation 的结论。

## Part III：SST-2 表示

### 主结果

| 表示 | 选择的 C | 测试 accuracy | 测试 macro-F1 | 来源 |
|---|---:|---:|---:|---|
| LSTM masked mean | 0.01 | 0.6456 | 0.6442 | `results_e20/part3/summary.json` |
| Transformer masked mean | 0.01 | **0.6640** | **0.6625** | `results_e20/part3/summary.json` |
| LSTM last state | 0.01 | 0.6261 | 0.6234 | `results_e20/part3/summary.json` |

SST-2 OOV 比例为训练 7.6802%、验证 7.3689%、测试 6.9213%；来源：`results_e20/part3/summary.json`。

在语言建模中，LSTM/Transformer perplexity 为 154.7271/163.0557，排名 1/2；在平均池化分类中，accuracy 为 0.6456/0.6640，排名 2/1。来源：`results_e20/part3/summary.json`。因此主实验的 perplexity 排名没有迁移到下游排名。一个可能解释是平均池化的 Transformer 状态更适合此分类器，也可能是 WikiText 与电影评论的领域差异；当前实验不能分离这些因素。

8-epoch 补充结果中，Transformer/LSTM 的 perplexity 排名为 1/2，分类 accuracy 排名也为 1/2；Transformer mean accuracy 0.6479，LSTM mean 0.6216。来源：`results_e8/part3/summary.json`。这进一步说明跨任务排名也依赖训练预算。

### 代表性错误样本

以下均来自 `results_e20/part3/errors.csv`。预测顺序为“LSTM mean / Transformer mean / LSTM last”。

**转折、让步或混合情感**

1. “we root for ... even like them, though perhaps ... pity.”：真值 1，预测 0/1/1。Transformer 捕捉到整体正类，而 LSTM mean 被后半句影响。
2. “there is nothing outstanding ... but it is good enough ...”：真值 1，预测 0/1/1。前负后正的转折造成表示差异。
3. “it 's a scattershot affair, but when it hits its mark it 's brilliant.”：真值 1，预测 0/0/1。两个平均池化表示都误判。
4. “if ... solaris is a failure it is a glorious failure.”：真值 1，预测 0/0/0。表面负词与整体标签冲突。

**讽刺、隐喻或依赖常识**

5. “or doing last year 's taxes with your ex-wife.”：真值 0，预测 0/1/1。句子没有直接评价词。
6. “holden caulfield did it better.”：真值 0，预测 1/0/1。理解需要比较对象与语用信息。
7. “the cold turkey would 've been a far better title.”：真值 0，预测 1/0/1。负面评价通过改标题间接表达。
8. “why make a documentary about these marginal historical figures?”：真值 0，预测 1/1/1。修辞问句被三种表示共同误判。

**简短或隐含的正面评价**

9. “the emotions are raw and will strike a nerve ...”：真值 1，预测 0/1/0。Transformer mean 是唯一正确表示。
10. “has a lot of the virtues of eastwood at his best.”：真值 1，预测 0/1/0。短句依赖 “virtues” 与人物比较。

这些类别是对表面语言现象的人工归纳，不是经控制实验确认的错误原因。

## 对初步观察的核实

1. **已确认**：20 epochs 的最佳验证 loss 排名为 LSTM 5.1091（epoch 20）< Transformer 5.1609（epoch 11）< RNN 5.2463（epoch 13）；来源：`results_e20/part1/log.txt` 与 `results_e20/part1/summary.json`。Transformer 在 epoch 11 后总体上升，但 epoch 14、17 有小幅回落，因此更准确的说法是“未再优于 epoch 11”，不是逐轮单调上升。
2. **已确认**：8 epochs 的验证 loss 排名为 Transformer 5.1726 < RNN 5.2776 < LSTM 5.4208，和 20 epochs 不同；来源：`results_e8/part1/log.txt`。
3. **部分确认**：Word2Vec 5.0478 ≈ GloVe 5.0503 < 随机 5.1091，以及 GloVe 最终 train/val 差距最小，均已确认；来源：`results_e20/part1/log.txt` 与 `results_e20/part2/log.txt`。GloVe 在 epochs 1–3 的 val loss 下降 0.4375，慢于随机设置的 0.5819 和 Word2Vec 的 0.6649；但若把“前几个”扩展到 epoch 5，随机设置下降 0.8387，反而小于 GloVe 的 0.9610。因此“前几个 epoch 下降最慢”只在明确采用前三轮时成立。
4. **部分确认**：两次运行中 RNN 前 8 轮的 train/val 数值在日志显示的四位小数上完全相同；LSTM 和 Transformer 每轮存在细微差别。来源：`results_e20/part1/log.txt` 与 `results_e8/part1/log.txt`。日志没有保存更高精度 loss，因此不能确认底层浮点值“完全相同”；GPU 非确定性是合理假设，但本文件没有直接证据证明原因。

## 文件缺失与可解释性边界

- `results_e20/part1/summary.json` 中 RNN 和 Transformer 的 `epoch_seconds` 数组分别只到最佳 checkpoint（13 和 11 项），但 `log.txt` 各有完整 20 轮；因此最佳 epoch 与最终 loss 使用日志，`train_seconds` 使用 summary 中记录的完整训练总时间。
- 正式结果中没有逐样本 token OOV 标记，因此无法把单个分类错误直接归因于 OOV；只能报告整体 OOV 比例。来源：`results_e20/part3/summary.json` 与 `results_e20/part3/errors.csv`。
- 本文件保留 seed 42 的逐轮分析；四种子（42、1、2、3）汇总另见 `analysis/multiseed_summary.md`。
- GloVe 下载或预训练本身的时间未单独记录；仅 LM 训练时间可用。来源：`results_e20/part2/summary.json`。
