# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

exmaples文件夹下提供了哪些训练示例脚本，支持哪几类算法的训练？请逐个列举

*Listed directory [examples](file:///Users/linus/Desktop/2025/newborn/verl/examples) *

### Planner Response

`examples` 文件夹下提供了丰富的训练示例脚本，涵盖多种算法类别。以下是逐个列举：

---

## 一、强化学习（RL）训练算法

| # | 目录 | 算法 | 说明 |
|---|------|------|------|
| 1 | `ppo_trainer/` | **PPO** (Proximal Policy Optimization) | 近端策略优化，经典 RLHF 算法，需要 Actor + Critic + Reward Model + Ref Model |
| 2 | `grpo_trainer/` | **GRPO** (Group Relative Policy Optimization) | 分组相对策略优化，DeepSeek 提出，免 Critic |
| 3 | `rloo_trainer/` | **RLOO** (REINFORCE Leave-One-Out) | 基于 leave-one-out 基线的 REINFORCE 变体 |
| 4 | `reinforce_plus_plus_trainer/` | **REINFORCE++** | REINFORCE 的改进版本 |
| 5 | `remax_trainer/` | **ReMax** | 另一种 RL 训练算法变体 |
| 6 | `gpg_trainer/` | **GPG** (Generalized Policy Gradient) | 广义策略梯度 |
| 7 | `gmpo_trainer/` | **GMPO** (Generalized Min-Max Policy Optimization) | 广义极小极大策略优化 |
| 8 | `gspo_trainer/` | **GSPO** (Group Score Policy Optimization) | 分组分数策略优化 |
| 9 | `cispo_trainer/` | **CISPO** | 一种策略优化变体 |
| 10 | `sapo_trainer/` | **SAPO** (Self-Adaptive Policy Optimization) | 自适应策略优化 |

---

## 二、监督微调（SFT）

| # | 目录 | 说明 |
|---|------|------|
| 11 | `sft/` | **SFT** (Supervised Fine-Tuning) 监督微调训练 |

---

## 三、特殊训练技术 / 工具

| # | 目录 | 说明 |
|---|------|------|
| 12 | `mtp_trainer/` | **MTP** (Multi-Token Prediction) 多 Token 预测训练 |
| 13 | `otb_trainer/` | **OTB** (On-The-Board) 训练 |
| 14 | `rollout_correction/` | **Rollout Correction** 采样校正技术 |

---

## 四、基础设施 / 辅助工具

| # | 目录 | 说明 |
|---|------|------|
| 15 | `data_preprocess/` | 数据预处理脚本 |
| 16 | `generation/` | 推理/生成示例 |
| 17 | `sglang_multiturn/` | 基于 SGLang 的多轮对话示例 |
| 18 | `split_placement/` | 模型分片放置策略示例 |
| 19 | `prefix_grouper/` | 前缀分组相关 |
| 20 | `router_replay/` | 路由重放相关 |
| 21 | `tuning/` | 超参调优相关 |
| 22 | `tutorial/` | 教程 |
| 23 | `ray/` | Ray 分布式相关示例 |
| 24 | `skypilot/` | SkyPilot 云端调度示例 |
| 25 | `slurm/` | Slurm 集群调度示例 |

---

### 总结

`examples` 文件夹主要支持 **三大类算法训练**：

1. **强化学习算法（10 种）**：PPO、GRPO、RLOO、REINFORCE++、ReMax、GPG、GMPO、GSPO、CISPO、SAPO
2. **监督微调（SFT）**
3. **特殊训练技术**：MTP（多 Token 预测）、OTB、Rollout Correction

其中 RL 算法最为丰富，`grpo_trainer`（60 个子文件）和 `ppo_trainer`（23 个子文件）提供了最多的配置示例，覆盖不同模型规模和数据集的组合。

### User Input

我发现RL算法的各种变体的入口文件都是main_ppo.py，这是为何？

### Planner Response

Let我来验证一下这个现象，并看看 `main_ppo.py` 的设计。

*Grep searched codebase*

*Searched filesystem*

### Planner Response

确认了：几乎所有 RL 变体都调用 `python3 -m verl.trainer.main_ppo`。让我看看这个文件的内容来解释原因：

*Analyzed [main_ppo.py](file:///Users/linus/Desktop/2025/newborn/verl/verl/trainer/main_ppo.py) *

### Planner Response

Now let me look at the key part — how the algorithm is selected:

*Grep searched codebase*

*Viewed [main_ppo.py](file:///Users/linus/Desktop/2025/newborn/verl/verl/trainer/main_ppo.py) *

### Planner Response

Now let me also quickly check how different algorithms differentiate themselves—look at one example script to see how the algorithm is configured:

*Viewed [run_deepseek7b_llm.sh](file:///Users/linus/Desktop/2025/newborn/verl/examples/grpo_trainer/run_deepseek7b_llm.sh) *

*Viewed [run_cispo_qwen2_5_0_5b_gsm8k.sh](file:///Users/linus/Desktop/2025/newborn/verl/examples/cispo_trainer/run_cispo_qwen2_5_0_5b_gsm8k.sh) *

*Grep searched codebase*

### Planner Response

Let me also check the `adv_estimator` config to see what options are available:

*Grep searched codebase*

*Grep searched codebase*

*Viewed code item*

### Planner Response

现在真相大白了！这是一个非常好的问题。以下是详细解释：

## 为什么所有 RL 变体都用 [main_ppo.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/verl/trainer/main_ppo.py:0:0-0:0) 作为入口？

### 核心原因：**统一的训练框架 + 配置驱动的算法切换**

verl 框架采用的是 **"一个入口，配置决定一切"** 的设计理念。所有 RL 算法变体（PPO、GRPO、RLOO、REINFORCE++ 等）共享**完全相同的训练流程骨架**，它们之间的差异仅体现在一个配置参数上：

```
algorithm.adv_estimator=xxx
```

### 各算法是如何区分的？

对比两个脚本就能看出差别：

| 算法 | 脚本关键配置行 |
|------|-------------|
| **GRPO** | `algorithm.adv_estimator=grpo` |
| **CISPO** | `algorithm.adv_estimator=grpo` + `actor_rollout_ref.actor.policy_loss.loss_mode=cispo` |
| **PPO** | `algorithm.adv_estimator=gae`（默认值，需要 Critic） |
| **RLOO** | `algorithm.adv_estimator=rloo` |
| **REINFORCE++** | `algorithm.adv_estimator=reinforce_plus_plus` |
| **ReMax** | `algorithm.adv_estimator=remax` |
| **GPG** | `algorithm.adv_estimator=gpg` |

在 [core_algos.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/verl/trainer/ppo/core_algos.py:0:0-0:0) 中，定义了一个 [AdvantageEstimator](cci:2://file:///Users/linus/Desktop/2025/newborn/verl/verl/trainer/ppo/core_algos.py:87:0-108:61) 枚举类，列出了所有支持的优势估计器：

```python
class AdvantageEstimator(str, Enum):
    GAE = "gae"                        # PPO 使用
    GRPO = "grpo"                      # GRPO/CISPO/GMPO/GSPO/SAPO 使用
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"
    GPG = "gpg"
    # ... 更多变体
```

### 为什么这样设计？

这些 RL 算法**本质上共享同一套训练循环**：

```
采样(Rollout) → 计算奖励(Reward) → 计算优势(Advantage) → 更新策略(Policy Update)
```

它们之间的**唯一核心区别**就是**第三步"计算优势"使用的估计器不同**，以及**第四步"更新策略"使用的 loss 函数可能不同**（如 CISPO 通过 `loss_mode=cispo` 切换）。

既然 95% 的代码逻辑完全一样，verl 就没有为每个算法写一个独立的入口文件，而是：

1. **[main_ppo.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/verl/trainer/main_ppo.py:0:0-0:0)** → 统一入口，负责初始化 Ray、加载模型、构建数据集
2. **`RayPPOTrainer`** → 统一的训练循环
3. **`algorithm.adv_estimator`** → 通过配置注入不同的优势估计函数
4. **`policy_loss.loss_mode`** → 通过配置注入不同的策略损失函数

> 💡 所以文件虽然叫 [main_ppo.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/verl/trainer/main_ppo.py:0:0-0:0)，但更准确地说它是 **"main_rl.py"** —— 一个通用的 RL 训练入口。名字沿用 PPO 只是因为 PPO 是最早实现的算法，属于历史命名惯例。

### User Input

scripts文件夹下的文件有什么作用？请逐步解释

*Listed directory [scripts](file:///Users/linus/Desktop/2025/newborn/verl/scripts) *

*Analyzed [converter_hf_to_mcore.py](file:///Users/linus/Desktop/2025/newborn/verl/scripts/converter_hf_to_mcore.py) *

*Analyzed [diagnose.py](file:///Users/linus/Desktop/2025/newborn/verl/scripts/diagnose.py) *

*Analyzed [init_random_model.py](file:///Users/linus/Desktop/2025/newborn/verl/scripts/init_random_model.py) *

*Analyzed [legacy_model_merger.py](file:///Users/linus/Desktop/2025/newborn/verl/scripts/legacy_model_merger.py) *

*Analyzed [megatron_merge_lora.py](file:///Users/linus/Desktop/2025/newborn/verl/scripts/megatron_merge_lora.py) *

*Analyzed [rollout_viewer.py](file:///Users/linus/Desktop/2025/newborn/verl/scripts/rollout_viewer.py) *

*Analyzed [print_cfg.py](file:///Users/linus/Desktop/2025/newborn/verl/scripts/print_cfg.py) *

*Viewed [generate_trainer_config.sh](file:///Users/linus/Desktop/2025/newborn/verl/scripts/generate_trainer_config.sh) *

*Viewed [install_vllm_sglang_mcore.sh](file:///Users/linus/Desktop/2025/newborn/verl/scripts/install_vllm_sglang_mcore.sh) *

*Viewed [install_sglang_mcore_npu.sh](file:///Users/linus/Desktop/2025/newborn/verl/scripts/install_sglang_mcore_npu.sh) *

*Listed directory [veomni](file:///Users/linus/Desktop/2025/newborn/verl/scripts/veomni) *

### Planner Response

以下是 `scripts` 文件夹下各文件的逐个解释：

---

## 📁 `scripts/` 文件功能一览

### 1. 🔧 环境安装脚本

| 文件 | 作用 |
|------|------|
| **[install_vllm_sglang_mcore.sh](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/install_vllm_sglang_mcore.sh:0:0-0:0)** | **CUDA GPU 环境一键安装脚本**。按顺序安装：① vLLM + SGLang 推理框架 ② transformers/ray 等基础包 ③ FlashAttention + FlashInfer ④ TransformerEngine + Megatron-LM ⑤ OpenCV 修复 ⑥ cuDNN |
| **[install_sglang_mcore_npu.sh](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/install_sglang_mcore_npu.sh:0:0-0:0)** | **华为 NPU (昇腾) 环境安装脚本**。针对 A2/A3 芯片，安装 SGLang NPU 版、torch_npu、sgl-kernel-npu、MindSpeed + Megatron 等 |

---

### 2. 🔄 模型格式转换工具

| 文件 | 作用 |
|------|------|
| **[converter_hf_to_mcore.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/converter_hf_to_mcore.py:0:0-0:0)** | **HuggingFace → Megatron-Core 格式转换器**（611行）。支持单卡和分布式转换，适配 Qwen2.5-VL、DeepSeek-V3 等模型。Megatron 训练前的必备步骤 |
| **[legacy_model_merger.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/legacy_model_merger.py:0:0-0:0)** | **分布式训练 checkpoint → HuggingFace 格式合并器**（805行）。将 FSDP 或 Megatron 分片保存的检查点合并回标准 HF 格式，支持上传到 HuggingFace Hub |
| **[megatron_merge_lora.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/megatron_merge_lora.py:0:0-0:0)** | **Megatron LoRA 权重合并工具**。将 Megatron 训练产出的 LoRA adapter 权重合并到基础模型中，然后保存为完整的 HF 模型 |

---

### 3. 🛠️ 调试与诊断工具

| 文件 | 作用 |
|------|------|
| **[diagnose.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/diagnose.py:0:0-0:0)** | **环境诊断工具**（313行）。一键检查：Python 版本、pip 包版本、verl 安装状态、操作系统、GPU 硬件信息、CUDA 版本、网络连通性等。排错时非常实用 |
| **[init_random_model.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/init_random_model.py:0:0-0:0)** | **随机权重模型初始化工具**。用自定义 config（如缩小层数、隐藏维度）创建一个小型随机权重模型，**主要用于调试**——不需要下载完整大模型就能跑通训练流程 |
| **[print_cfg.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/print_cfg.py:0:0-0:0)** | **配置打印工具**。读取 Hydra 配置并将完整解析后的配置打印出来，方便检查配置是否正确 |

---

### 4. 📊 可视化工具

| 文件 | 作用 |
|------|------|
| **[rollout_viewer.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/rollout_viewer.py:0:0-0:0)** | **Rollout 结果可视化查看器**（566行）。基于 [Textual](https://textual.textualize.io/) 的终端 TUI 应用，可以交互式浏览训练过程中每一步的 rollout 结果（prompt/response/reward 等），支持搜索、翻页、排序和字段过滤 |

---

### 5. ⚙️ 配置生成

| 文件 | 作用 |
|------|------|
| **[generate_trainer_config.sh](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/generate_trainer_config.sh:0:0-0:0)** | **自动生成参考配置文件**。调用 [print_cfg.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/print_cfg.py:0:0-0:0) 将 Hydra 多层配置展平为一个完整的 YAML 文件（如 `_generated_ppo_trainer.yaml`），仅供参考，不直接使用。同时会检查生成结果是否与仓库中已有的一致（CI 校验用） |

---

### 6. 📁 `veomni/` 子目录

| 文件 | 作用 |
|------|------|
| **[veomni/moe_split.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/veomni/moe_split.py:0:0-0:0)** | MoE（Mixture of Experts）模型的权重**拆分**工具，将合并的专家权重拆分为独立存储 |
| **[veomni/moe_merge.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/scripts/veomni/moe_merge.py:0:0-0:0)** | MoE 模型的权重**合并**工具，将拆分后的专家权重合并回来 |

---

### 总结

`scripts/` 文件夹是 verl 的**工具箱**，按功能分为 5 大类：

```
scripts/
├── 🏗️ 环境安装    → install_*.sh（一键搭建训练环境）
├── 🔄 格式转换    → converter_hf_to_mcore.py / legacy_model_merger.py / megatron_merge_lora.py
├── 🛠️ 调试诊断    → diagnose.py / init_random_model.py / print_cfg.py
├── 📊 可视化      → rollout_viewer.py
├── ⚙️ 配置生成    → generate_trainer_config.sh
└── veomni/       → MoE 模型拆分/合并工具
```

### User Input

docs下的文件是不是就是verl的官方文档？

*Listed directory [docs](file:///Users/linus/Desktop/2025/newborn/verl/docs) *

### Planner Response

是的，`docs/` 就是 verl 的**官方文档源文件**，使用 [Sphinx](https://www.sphinx-doc.org/) 构建，最终会发布到 [verl.readthedocs.io](https://verl.readthedocs.io)。

主要结构如下：

| 目录/文件 | 内容 |
|-----------|------|
| [conf.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/docs/conf.py:0:0-0:0) | Sphinx 配置文件（定义文档主题、扩展等） |
| [index.rst](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/docs/index.rst:0:0-0:0) | 文档首页/目录结构 |
| `start/` | **快速开始**指南（安装、首次运行等） |
| `preparation/` | **数据准备**教程 |
| `algo/` | **算法说明**（PPO、GRPO 等各 RL 算法的原理与配置） |
| `examples/` | **示例教程**（各种训练场景的具体指导） |
| `advance/` | **进阶用法**（checkpoint 转换、自定义 reward 等） |
| `workers/` | **Worker 架构**文档 |
| `perf/` | **性能优化**相关 |
| `api/` | **API 参考** |
| `faq/` | **常见问题** |
| `ascend_tutorial/` | **华为昇腾 NPU** 使用教程 |
| `amd_tutorial/` | **AMD GPU** 使用教程 |
| `sglang_multiturn/` | **多轮对话**训练教程 |
| [hybrid_flow.rst](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/docs/hybrid_flow.rst:0:0-0:0) | HybridFlow 架构设计说明 |
| [single_controller.rst](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/docs/single_controller.rst:0:0-0:0) | SingleController 架构设计说明 |
| `_static/` | 静态资源（图片等） |
| [Makefile](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/docs/Makefile:0:0-0:0) | 构建命令（`make html` 即可在本地生成文档） |

本地构建文档只需：
```bash
cd docs
pip install -r requirements-docs.txt
make html
```
然后在 `_build/html/index.html` 就能查看完整的渲染后文档。

### User Input

verl文件夹下的主要文件夹是不是trainer？

### Planner Response

`trainer` 是重要文件夹之一，但不是唯一的"主要"文件夹。让我看看完整结构：

*Listed directory [verl](file:///Users/linus/Desktop/2025/newborn/verl/verl) *

### Planner Response

`trainer` 确实很核心，但 `verl/verl/` 下有多个同样重要的模块，它们共同组成了完整的训练框架。以下是各文件夹的职责：

| 目录 | 子文件数 | 职责 | 重要性 |
|------|---------|------|--------|
| **`trainer/`** | 62 | **训练主循环**（main_ppo 入口、RayPPOTrainer、核心算法、配置） | ⭐⭐⭐ 核心 |
| **`workers/`** | 72 | **分布式 Worker 实现**（Actor/Critic/Reward/Ref 的 FSDP/Megatron worker） | ⭐⭐⭐ 核心 |
| **`models/`** | 65 | **模型适配层**（各种 HF 模型到 FSDP/Megatron 的桥接） | ⭐⭐⭐ 核心 |
| **`utils/`** | 105 | **通用工具集**（数据集处理、tokenizer、checkpoint、日志等） | ⭐⭐⭐ 核心 |
| **`experimental/`** | 138 | **实验性功能**（新特性孵化区，文件最多） | ⭐⭐ 重要 |
| **`single_controller/`** | 7 | **分布式控制器**（SingleController 架构，协调所有 Worker） | ⭐⭐ 重要 |
| **`checkpoint_engine/`** | 7 | **检查点引擎**（异步保存/加载 checkpoint） | ⭐⭐ 重要 |
| **`tools/`** | 15 | **工具函数**（沙盒执行、代码评测等） | ⭐ 辅助 |
| **`interactions/`** | 6 | **多轮交互**相关 | ⭐ 辅助 |
| **`model_merger/`** | 5 | **模型合并**逻辑 | ⭐ 辅助 |
| **`third_party/`** | 7 | **第三方代码** | ⭐ 辅助 |
| **[version/](cci:1://file:///Users/linus/Desktop/2025/newborn/verl/scripts/converter_hf_to_mcore.py:72:0-117:36)** | 1 | 版本信息 | — |
| [protocol.py](cci:7://file:///Users/linus/Desktop/2025/newborn/verl/verl/protocol.py:0:0-0:0) | — | **数据协议**（DataProto 等核心数据结构，48KB 大文件） | ⭐⭐⭐ 核心 |

所以更准确地说，verl 的**四大核心模块**是：

```
trainer/          → 训练"怎么跑"（流程编排）
workers/          → 训练"谁来干"（分布式执行者）
models/           → 训练"用什么模型"（模型适配）
utils/            → 训练"靠什么支撑"（基础设施）
```

`trainer` 是**调度中枢**，但它离不开 `workers`（实际执行）和 `models`（模型加载与适配）的配合。