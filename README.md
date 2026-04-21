# TrustLoRA: 基于超网络的语言模型校准

> **范彦松 & 肖智豪** — 北京航空航天大学

![teaser.png](teaser.png)

## 摘要

基于Transformer的现代模型经常遭受校准偏差的影响，产生过于自信的预测，这些预测不能反映真实的经验频率。本研究探讨了LoRA（低秩适应）和一种新颖的超网络基自适应框架在RoBERTa参数高效替代全量微调方面的校准动态。通过在GLUE基准测试中评估，我们证明基于LoRA的适应在所有任务中都能实现与全量微调相当的校准性能——在某些特定任务中甚至超过全量微调，同时保持显著更高的参数效率。我们进一步探索了一种动态方法，其中共享超网络生成LoRA因子（A和B），以在所有层之间诱导结构耦合。我们的研究揭示了一个关键权衡：约束适应空间（例如，冻结矩阵A）作为强大的正则化器能够提升ECE指标，但需要在下游任务准确性方面进行精心平衡的牺牲。

---

## 概述

标准的LoRA适配器是静态的——一旦训练完成，每个层都使用一对固定的低秩矩阵**A**和**B**。本项目将每层的静态**B**矩阵（以及可选的**A**矩阵）替换为由紧凑共享**超网络**（`LoRAHyperNet`）生成的输出。该超网络以每个Transformer层的学得嵌入为条件，并在单次前向传播中生成跨所有层的协调适配器权重，而所有主干参数保持冻结状态。

### 核心思想

- **动态权重生成** — 适配器权重由超网络在每个前向传播中生成，实现了跨层的参数共享并减少了总的适配器参数数量。
- **两种超网络架构** — 4层MLP（隐藏层大小2048，GELU激活函数）或2层Transformer编码器（256维，16个注意力头），两者都以128维学得层嵌入为条件。
- **固定-A vs. 生成-A变体** — 矩阵A可以固定（Kaiming均匀初始化），仅生成B矩阵，这充当正则化器改善校准性能，但会牺牲一定的任务表现。
- **噪声混合** — 生成的矩阵可以通过`add` / `multiply` / `replace`模式与初始随机矩阵混合；混合系数`noise_alpha`在训练过程中线性退火至0。
- **校准感知评估** — 每次评估步骤都会计算ECE、类间ECE、MCE、ACE、阈值ACE和Brier分数，同时还包括GLUE任务指标。

---

## 主要发现

1. **LoRA ≈ 全量微调在校准方面** — LoRA在各种GLUE任务中提供与全量微调相当的可比校准性能，同时在参数效率上显著优于全量微调。

2. **超网络并非普遍改善校准** — 通过超网络完全生成A和B矩阵得到的指标与标准LoRA大致相似，这表明仅靠跨层结构耦合不会产生系统性的置信度校正。

3. **固定矩阵A对模型起正则化作用** — 在生成B矩阵的同时冻结A矩阵会引入结构化扰动，适度降低ECE指标。这种方法的代价是在任务性能上的持续下降（CoLA的MCC、SST-2的准确率）。

4. **延长训练会恶化校准** — 在所有方法中，更长的训练时间会渐进式地过拟合分布并削弱不确定性估计。

---

## 项目结构

```
.
├── run_experiment.py               # 主要训练入口点
├── calibration_metrics.py          # ECE, CECE, MCE, ACE, TACE, Brier Score
├── requirements.txt
│
├── models/
│   ├── hypernet.py                 # LoRAHyperNet (MLP) & LoRAHyperNetTransformer
│   ├── dynamic_lora_layer.py       # DynamicLoRALayer – 应用超网络输出作为LoRA
│   └── get_roberta.py              # 模型构建器（基线 & 超网络变体）
│
├── data_loading/
│   └── get_datasets.py             # GLUE数据集加载和分词
│
├── utils/
│   ├── alpha_callback.py           # 在训练过程中线性退火noise_alpha
│   ├── batch_generation_trainer.py # 自定义Trainer：每批预生成B矩阵
│   ├── forward_pass_repetition_data_collator.py  # 通过重复前向传播实现梯度累积
│   ├── lr_scheduler_callback.py    # LR调度工具
│   ├── metrics.py                  # B矩阵统计信息（跨层的均值/标准差）
│   ├── metrics_trainer_callback.py # 保存每轮指标的CSV文件
│   └── one_hot_encoding.py         # 独热编码（学得嵌入的替代方案）
│
├── params/
│   ├── example_config_hypernet.py  # 模板配置 — 超网络模式
│   ├── example_config_no_hypernet.py # 模板配置 — LoRA / FT基线
│   ├── ft_baselines/               # 每个GLUE任务的完整微调配置
│   ├── lora_baselines/             # 每个GLUE任务的LoRA基线配置
│   ├── hypernet_mlp/               # MLP超网络实验（fixed_A / gen_A）
│   ├── transformer/                # Transformer超网络实验
│   └── roberta_large_baselines/    # RoBERTa-large FT & LoRA配置
│
├── pretrained_models/              # 保存的检查点
└── results/                        # 每次运行的CSV指标日志
```

---

## 安装

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

**要求:** Python ≥ 3.9, PyTorch, Transformers, PEFT, Datasets, WandB, scikit-learn, accelerate。

---

## 运行实验

所有实验都由单个入口点驱动，该入口点接受Python配置文件：

```bash
python run_experiment.py --params <path/to/config.py>
```

脚本加载数据集，构建模型，运行`num_runs`次独立种子实验，记录到**WandB**并将指标保存到`results/`目录。

### 基线实验

```bash
# 完整微调
python run_experiment.py --params params/ft_baselines/cola.py

# LoRA基线
python run_experiment.py --params params/lora_baselines/cola.py
```

### 超网络实验

```bash
# MLP超网络，固定A矩阵
python run_experiment.py --params params/hypernet_mlp/fixed_A/cola.py

# MLP超网络，生成A矩阵
python run_experiment.py --params params/hypernet_mlp/gen_A/cola.py

# Transformer超网络
python run_experiment.py --params params/transformer/fixed_A/cola.py
```

---

## 配置参考

配置文件是分配给`params`变量的普通Python字典。关键参数：

| 参数 | 描述 |
|---|---|
| `glue_dataset_name` | GLUE任务：`cola`, `sst2`, `mrpc`, `qqp`, `mnli`, `qnli`, `rte`, `stsb` |
| `model_name` | HuggingFace模型ID或本地检查点路径 |
| `use_hypernet` | `True`表示使用通过超网络的动态LoRA；`False`表示基线模式 |
| `use_peft` | 用PEFT LoRA配置包装模型 |
| `lora_r` | LoRA秩（默认值：8） |
| `lora_alpha` | LoRA缩放因子 |
| `layers_to_transform` | 应用LoRA的编码器层（默认值：全部12层） |
| `layers_to_use_hypernet` | 由其LoRA权重由超网络生成的层子集 |
| `hypernet_use_transformer` | `True`表示Transformer超网络；`False`表示MLP |
| `hypernet_transformer_nhead` | 注意力头数（Transformer超网络） |
| `hypernet_transformer_num_layers` | 超网络中Transformer层数 |
| `hypernet_hidden_dim` | MLP超网络的隐藏层维度 |
| `hypernet_embeddings_dim` | 学得层嵌入的维度（默认值：128） |
| `hypernet_A_matrix` | A矩阵的处理方式：`"random"`、`"fixed"`或`"generated"` |
| `hypernet_noise_type_A/B` | A/B的混合模式：`"replace"`、`"add"`、`"multiply"` |
| `hypernet_noise_alpha` | 初始混合权重；当`hypernet_reduce_noise_alpha=True`时退火至0 |
| `hypernet_large_model` | `True`表示4层MLP；`False`表示2层MLP |
| `hypernet_use_batches` | 每批一次预生成B矩阵 |
| `forward_pass_reps` | 每批重复前向传播N次 |
| `num_runs` | 独立运行次数（不同种子） |

---

## 校准指标

每次评估步骤都会计算以下指标并记录到WandB和结果CSV文件中：

| 指标 | 公式/描述 |
|---|---|
| **ECE** | 每个bin的\|accuracy − confidence\|差距的加权平均值 |
| **类间ECE (CECE)** | 按类别计算的ECE，对所有类别取平均 |
| **MCE** | 每个bin的最大校准误差（最坏情况bin） |
| **ACE** | 等人口bin的ECE，按类别取平均 |
| **TACE** | 限制于置信度阈值ε ∈ {0.01, 0.001, 0.0001}以上的ACE |
| **Brier分数** | 预测概率向量与独热标签之间的均方误差 |

---

## 实验跟踪

所有运行都记录到[Weights & Biases](https://wandb.ai)。每次运行都被标记为：
- `hypernet`或`baseline`
- GLUE数据集名称
- 运行索引

指标也会作为CSV文件本地保存在`results/`目录中用于离线分析。

---

## 支持的GLUE任务

| 任务 | 指标 |
|---|---|
| CoLA | Matthews相关系数(MCC) |
| SST-2 | 准确率 |
| MRPC | F1分数 |
| QQP | F1分数/准确率 |
| MNLI | 准确率 |
| QNLI | 准确率 |
| RTE | 准确率 |
| STS-B | Pearson/Spearman相关系数 |