# Faster-YOCO

[English](README.md) | [简体中文](README.zh-CN.md)

**AI agent 优化的 YOCO-3B 预训练：在 8 张 Hopper GPU 上每次更新加速 2.12×。**

![Faster-YOCO：8 张 Hopper GPU 上每次更新加速 2.12 倍](docs/performance-views/01-benchmark.svg)

| 实现 | 每次更新耗时（毫秒）↓ | 每次更新耗时中位数（毫秒）↓ | 加速比 ↑ |
|---|---:|---:|---:|
| 基线 | 251.9 | 249.7 | 1.00× |
| **Faster-YOCO** | **117.9** | **117.5** | **2.12×** |

在 **8 张 Hopper GPU（80 GB）** 上用完整检查测得：YOCO-3B（`yoco_3b`，34.5 亿参数）按上游 `scripts/train.sh` 配方训练，每卡一条 2,048 token 的序列（每次更新 16,384 个 token），BF16，主权重和 Adam 状态为 FP32，ZeRO 第一阶段优化器分片（`--zero-sharding os`），**不使用 FP8**。测速直接运行仓库的训练入口 `train.py`，优化版通过了检查中的数值关。加速比取每次更新中位耗时之比和总耗时之比（2.1369×）中较小的一个：**2.1247×**。

在同一台机器上训练 20,000 次更新，Faster-YOCO 用时 **2,799 秒**，基线 **5,345 秒**：端到端 **1.91×**，含启动和保存模型的时间（每次更新 0.1180 秒对 0.2497 秒）。显存用得更多：这次训练中每卡峰值分配 55.4 GiB，基线 48.3 GiB（**多 7.1 GiB**）。

## 快速开始

先准备[环境和训练数据](docs/training.md)。优化以环境变量开关的形式加在上游训练代码上；启动脚本会加载这些开关，再在 8 张 GPU 上运行上游的 YOCO-3B 命令：

```bash
export DATA_DIR=/path/to/Data        # json/train.json、json/<来源>.json、shard/<来源>/*.jsonl
export OUTPUT_DIR=/path/to/output
bash scripts/train_faster_yoco_3b.sh --max-update 20000
```

也可以先 `source scripts/configs/faster.env`，再在 `yoco/` 下用上游的 `torchrun ... train.py` 命令，加上[记录在案的改动项](docs/training.md#launch)。只在 **PyTorch 2.3.1 + CUDA 12.1（Triton 2.3.1）、apex、flash-attn 2.6.3 和上游指定的 fairseq 分支**上测试过，8 张 GPU，`--zero-sharding os`。所有开关默认关闭；一个都不设时，代码走上游路径。

## Agent 优化过程

图中每个点都是同一台机器上的一次完整检查，8 张 Hopper GPU，每卡 2,048 个 token。

![YOCO 优化过程：1.39 倍到 2.12 倍](docs/performance-views/02-history.svg)

## 主要改动

- **ZeRO-1 优化器路径：** 上游在每张卡上都对完整的 FP32 梯度做拷贝、缩放和求范数，再广播完整的 FP32 主权重；这里每张卡只对自己负责的分片做 FP32 计算，更新后的 BF16 参数用一次原地 all-gather 交换，参数本身就放在这块收集缓冲区里。Adam 直接写出 BF16 副本（Triton）。
- **梯度规约：** 不再用 DDP 分桶 all-reduce，每个梯度只送到持有其优化器状态的那张卡（ZeRO-2 式），并通过 GPU 的拷贝引擎推送，规约 kernel 不再和反向的矩阵乘抢计算单元；持有者按卡号顺序用 FP32 把 8 份相加。词嵌入梯度在持有它的卡上由各卡的 token 行构建。
- **kernel 发射与主机同步：** 26 个解码层（前向和反向）以 CUDA graph 运行；fairseq 的 `train_step` 和每次更新末尾的日志不再等待 GPU。
- **kernel：** Triton RMSNorm（以及与转置融合的逐头归一化），门控保持（gated retention）递推周围的逐元素运算用 Inductor 编译，在 BF16 logits 上直接算交叉熵的 Triton kernel，q/k/v/g 四个投影合成一次矩阵乘。

![优化模块图：ZeRO-1 路径、梯度规约、发射与同步、kernel](docs/change-views/module-map.svg)

详见[开关表](docs/training.md#switches)和[测速配置](scripts/configs/faster.env)。这些改动会改变浮点舍入和求和顺序，不是无损加速。

## 正确性验证

**数值检查。** 最终配置通过了[冻结的判卷](docs/judge.md)：结构检查和精确项（配方、数据、随机数指纹、调用次数、学习率、Adam 超参数、数据类型）没有任何失败；记录的 38,464 个张量中有 5,123 个逐位一致而不是全部一致，因此由统计关裁决。统计关从头训练 200 次更新，使用三个配对种子；阈值 δ 是基线自身种子间标准差的 2 倍，由六个基线种子事先冻结。

| 损失窗口（第几次更新） | 均值差（nats） | 均值差 / 基线标准差 | 在 δ = 2σ 内 |
|---|---:|---:|:---:|
| 5–15 | +0.00003 | +0.03 | ✓ |
| 15–25 | −0.00000 | −0.00 | ✓ |
| 25–35 | −0.00004 | −0.08 | ✓ |
| 35–50 | +0.00001 | +0.03 | ✓ |
| 50–100 | −0.00095 | −0.22 | ✓ |
| 100–200 | −0.00713 | −0.75 | ✓ |

**长训练。** Faster-YOCO 和基线按同一配方各用两个种子训练 20,000 次更新（3.28 亿 token）；每 2,000 次更新存一次检查点，用基线的模型代码在留出集上评测（164 万 fineweb-edu token 和 41 万代码 token，取自未参与训练的文件，并去掉了与训练文档完全相同的文档）。所有运行都没有出现非有限的 loss 或梯度范数。

![四组运行的训练 loss、留出集 loss 及其差值](docs/performance-views/04-validation.svg)

尺子是基线自己换种子的差，Y = ref<sub>s2</sub> − ref<sub>s1</sub>。若 |E1|、|E2|、|S| 都不超过 |Y|，该检查点算一致，其中 E1 = opt<sub>s1</sub> − ref<sub>s1</sub>，E2 = opt<sub>s2</sub> − ref<sub>s2</sub>，S = opt<sub>s2</sub> − opt<sub>s1</sub>。

| 更新次数 | Y | E1 | E2 | S | 一致 |
|---:|---:|---:|---:|---:|:---:|
| 2,000 | −0.1593 | +0.0018 | +0.0130 | −0.1481 | ✓ |
| 4,000 | −0.1466 | +0.0034 | +0.0169 | −0.1332 | ✓ |
| 6,000 | −0.1100 | −0.0023 | +0.0029 | −0.1048 | ✓ |
| 8,000 | −0.0642 | +0.0018 | +0.0031 | −0.0629 | ✓ |
| 10,000 | −0.0289 | +0.0052 | +0.0085 | −0.0256 | ✓ |
| 12,000 | −0.0261 | +0.0046 | +0.0085 | −0.0222 | ✓ |
| 14,000 | −0.0582 | +0.0043 | +0.0068 | −0.0558 | ✓ |
| 16,000 | −0.0665 | +0.0033 | +0.0046 | −0.0653 | ✓ |
| 18,000 | −0.0769 | −0.0027 | +0.0076 | −0.0666 | ✓ |
| 20,000 | −0.0255 | +0.0047 | +0.0100 | −0.0202 | ✓ |

表中是 fineweb-edu 0.8、代码 0.2 混合留出集的 loss 差，单位 bits/token。

| 指标 | 一致的检查点 | 最后一个检查点 |
|---|---:|:---:|
| 留出混合集 | 10 / 10 | ✓ |
| 留出 fineweb-edu | 10 / 10 | ✓ |
| 留出代码集（41 万 token） | 5 / 10 | ✓ |
| 训练 loss（1,000 次更新均值） | 7 / 10 | ✓ |

八处不一致里，七处只卡在 S 上；第八处（代码集 14,000）S 和 E1 都超线（0.0179，|Y| 为 0.0160）。按定义 S − Y = E2 − E1；同一个种子下两种实现的数据顺序相同，所以 S 紧跟 Y，只要 E2 − E1 与 Y 同号，S 就会超过 |Y|。同种子的训练曲线相关系数高于 0.9999；按每 100 次更新的均值比，差的绝对值平均为 0.004（种子 1）和 0.007（种子 2）bits，而基线两个种子之间是 0.13 bits。

**需要说明的一点。** Faster-YOCO 的留出集 loss 略高但方向一致：E1 平均 +0.0024 bits（10 个检查点中 8 个为正），E2 平均 +0.0082 bits（10 个全为正）。这最多占 loss 的 0.2%，在每个检查点都小于换种子的差（至少小 2.6 倍，2.6 倍出现在最后一个检查点的 E2），所以按规则通过；但只有两个种子，分不清这样大小的系统偏差是否真实存在。Faster-YOCO 的种子 2 是在另一台同型号机器上跑的，所以 E2 里也包含两台机器之间可能存在的差别。没有做下游评测。

详见[验证细节](docs/training.md#validation)和[复现脚本](scripts/validate/)。

## 为什么停在 2.12×

3× 要求每次更新不超过 84 ms。最终配置里，矩阵乘约占每次更新的 65 ms；即使按本机实测的 BF16 峰值算，这部分计算也需要 53.6 ms。每次更新之后，每张 GPU 还必须在下一次更新读取参数之前收到 6.0 GB 更新后的 BF16 参数，走 NVLink 至少 13.4 ms（现在用 NCCL 是 17.4 ms）。仅这两项下限就有 67–76 ms，所有组件的估计下限合计 80–91 ms，而这要求每个组件同时达到硬件极限。FP8 不在允许范围内。优化在第 28 轮后停止；第 26–28 轮未完成的工作不包含在内。

![最终配置的时间预算](docs/performance-views/03-time-budget.svg)

详见[为什么停在这里](docs/training.md#why-it-stopped-here)以及尝试过但没有采用的方向。

## AI4AI-for-Anything 的其他项目

[Faster-JiT](https://github.com/AI4AI-for-Anything/Faster-JiT) · [Faster-RAE](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Faster-LLaVA](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Faster-DMD](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Faster-SelfForcing](https://github.com/AI4AI-for-Anything/Faster-SelfForcing)

## 上游 YOCO 文档

基于 [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/YOCO) 在 `f4f7b27` 的 `YOCO/` 目录，即论文 [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/abs/2405.05254) 的代码。[MIT 许可证](LICENSE)（microsoft/unilm 的许可证）。`upstream` 分支保存未修改的代码。原 README 保存在 [README.upstream.md](README.upstream.md)，也附在下面。

<details>
<summary>原 YOCO README：方法、结果、安装、评测和预训练</summary>

# You Only Cache Once: Decoder-Decoder Architectures for Large Language Models

## Approach
<div align="center">
  <img src="./imgs/arch.png" width=60%/>
</div>

<div align="center">
  <img src="./imgs/inference.png" width=50%/>
</div>

## Performance
### Harness Eval
Training with 1T Tokens: 
| **Model**                  | **Arc-c** | **Arc-e** | **BoolQ** | **Hellaswag**$^*$ | **OBQA** | **PIQA** | **Winogrande** | **SciQ** | **Avg** |  
|----------------------------|-----------|-----------|-----------|-------------------|----------|----------|----------------|----------|---------|  
| OpenLLaMA-3B-v2            | 0.339     | 0.676     | 0.657     | **0.700**         | 0.260    | 0.767    | 0.629          | 0.924    | 0.619   |  
| StableLM-base-alpha-3B-v2  | 0.324     | 0.673     | 0.646     | 0.686             | 0.264    | 0.760    | 0.621          | 0.921    | 0.612   |  
| StableLM-3B-4E1T        | ---       | 0.666     | ---       | ---               | ---      | **0.768**| 0.632          | 0.914    | ---     |  
| YOCO-3B                 | **0.379** | **0.731** | 0.645     | 0.689             | **0.298**| 0.763    | 0.639          | 0.924    | **0.634**| 

Training with 1.6T Tokens: 
| **Model**                  | **Arc-c** | **Arc-e** | **BoolQ** | **Hellaswag**$^*$ | **OBQA** | **PIQA** | **Winogrande** | **SciQ** | **Avg** |  
|----------------------------|-----------|-----------|-----------|-------------------|----------|----------|----------------|----------|---------|  
| StableLM-3B-4E1T      | ---       | 0.688     | ---       | ---               | ---      | 0.762    | 0.627          | 0.913    | ---     | 
| YOCO-3B               | 0.396     | 0.733     | **0.644** | 0.698             | 0.300    | 0.764    | 0.631          | 0.921    | 0.636   |  
| YOCO-3B-1M            | **0.413** | **0.747** | 0.638     | **0.705**         | 0.300    | **0.773**| **0.651**      | **0.932**| **0.645**| 
### Needle In A Haystack
<div align="center">
  <img src="./imgs/1m_retrieval.png"/>
</div>

### Multi-Needle Eval
| **Model**               | **Size** | **N=1** | **N=2** | **N=4** | **N=8** |  
|-------------------------|----------|---------|---------|---------|---------|  
| GPT-4-128K              | --       | 1.00    | 1.00    | 0.98    | 1.00    |  
| MiniCPM-128K            | 2.4B     | 1.00    | 1.00    | 0.54    | 0.56    |  
| ChatGLM3-128K           | 6B       | 0.94    | 0.72    | 0.52    | 0.44    |  
| YaRN-Mistral-128K       | 7B       | 0.02    | 0.12    | 0.08    | 0.20    |  
| LWM-1M-text             | 7B       | 1.00    | 0.90    | 0.76    | 0.62    |  
| YOCO-3B-1M              | 3B       | 0.98    | 0.98    | 0.84    | 0.56    |  

## Setup
  
To install the required packages, use the following command:  
  
```bash  
pip install -r requirements.txt
```

Besides normal packages, [Apex](https://github.com/NVIDIA/apex) and [Flash-Attention](https://github.com/Dao-AILab/flash-attention) should be installed seperately following their offcial guidences.

## Harness Eval

To evaluate models in Harness-Eval, the script is as follows in ```scripts/eval_task.sh```:
```bash
cd fairseq/
TASK='harness_boolq'

torchrun --master-port=29505 --nproc_per_node=1 validate.py \
    --data-dir ../harness_data/ \
    --criterion harness_eval \
    --task harness_eval \
    --batch-size 4 \
    --eval-data ${TASK}  \
    --log-format simple  --log-interval 10 \
    --bf16 \
    --tokenizer-pad-to-multiple 8 \
    --arch yoco_3b_new --tiktoken-model cl100k_base --load-ckpt /path_to_ckpt/YOCO-3B-1M/checkpoint.pth --yoco-model /path_to_ckpt/YOCO-3B-1M  --tokens-per-sample 4096
```

## Needle In A Haystack Evaluation
Our model uses city-number pairs for long sequence evaluation. To get the results at a certain maximal length, the script is as follows in ```scripts/eval_needle.sh```:
```bash
cd fairseq/
torchrun --master-port=29504 --nproc_per_node=1 validate.py \
    --task pseudo \
    --criterion needle_haystack \
    --batch-size 1 \
    --max-epoch 1 \
    --no-save \
    --tiktoken-model cl100k_base \
    --bf16 \
    --arch yoco_3b_new --tiktoken-model cl100k_base --load-ckpt /path_to_ckpt/YOCO-3B-1M/checkpoint.pth --yoco-model /path_to_ckpt/YOCO-3B-1M --tokens-per-sample 1048576 --interval 1048576
```

To run Multi-Needle experiments, replace ```--criterion needle_haystack``` with ```--criterion multi_needle --needle-num {num}```.

## Pretraining From Scratch
To support distributed training, our implementation is based on infinibatch to read data iteratively. The overall data directory should be organized as follows:
```
Data/
├── json/
│   ├── train.json
│   └── CC.json
│   └── StarCoder.json
│   └── ...
├── shard/
│   ├── CC/
│   │   ├── 00000.jsonl
│   │   ├── 00001.jsonl
│   │   └── ...
│   └── StarCoder/
│       ├── 00000.jsonl
│       ├── 00001.jsonl
│       └── ...
```

We recommend that each sharded data files contains no more than 10K lines with one json dict per line, and jsonl file, such as ```Data/shard/CC/00000.jsonl```, should be in the format like this:
```json
{"text": "File 1 is here..."}
{"text": "File 2 is here..."}
...
```

Then, for each source, a JSON file preserves all the paths of the jsonl files. Take ```Data/json/CC.json``` for example:
```json
[
    "/path_to_data/Data/shard/CC/00000.jsonl",
    "/path_to_data/Data/shard/CC/00001.jsonl",
    ...
]
```

Finally, ```train.json``` records all sources' information and sampling ratio:
```json
[
    {
        "name": "CC",
        "weight": 0.5
    },
    {
        "name": "StarCoder",
        "weight": 0.2
    },
    ...
]
```

 ```scripts/train.sh```:
```bash
cd fairseq/
torchrun --nproc-per-node=1 train.py /path_to_data \
    --save-interval-updates 5000 \
    --no-epoch-checkpoints \
    --arch yoco_base \
    --criterion cross_entropy \
    --task gpt \
    --tokens-per-sample 2048 \
    --tokenizer-pad-to-multiple 8 \
    --pad-to-max-len \
    --optimizer adam --adam-betas "(0.9, 0.95)" \
    --adam-eps 1e-06 \
    --clip-norm 2.0 \
    --lr 0.00015 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 50 \
    --weight-decay 0.05 \
    --batch-size 1  \
    --model-parallel-size 1 \
    --update-freq 1 \
    --batch-read-ahead 1000 \
    --total-num-update 300000 \
    --log-format simple      --log-interval 10    --disable-validation \
    --tiktoken-model cl100k_base \
    --save-interval-updates 5000 \
    --bf16 # bf16 is encouraged in pre-training
```

</details>
