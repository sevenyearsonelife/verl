# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**verl (Volcano Engine Reinforcement Learning for LLMs)** 是字节跳动 Seed 团队开源的 LLM 强化学习训练库，基于 HybridFlow 论文（EuroSys 2025）。这是一个生产就绪的 RLHF 框架，支持从数十亿到数千亿参数规模的大模型训练。

## 核心架构

### 混合控制器编程模型

verl 采用 **Hybrid Controller** 架构，解耦计算与数据依赖：

- **Single Controller**: 基于 Ray 的统一分布式控制器（`verl/trainer/main_ppo.py`）
- **Workers**: 执行具体计算的 Worker 进程（`verl/workers/`）
- **DataProto**: 基于 TensorDict 的统一数据传输协议（`verl/protocol.py`）

### 核心目录结构

```
verl/
├── trainer/              # 训练器入口和配置
│   ├── main_ppo.py       # PPO 训练主入口
│   ├── ppo/              # PPO 算法实现
│   ├── config/           # Hydra 配置文件
│   └── *.py              # SFT、Eval 等其他训练器
├── workers/              # 分布式 Workers
│   ├── fsdp_workers.py   # FSDP/FSDP2 后端实现
│   ├── megatron_workers.py  # Megatron-LM 后端实现
│   ├── engine_workers.py     # 推理引擎 Worker
│   ├── actor/                # Actor 模型
│   ├── critic/               # Critic 模型
│   ├── rollout/              # Rollout 采样
│   └── reward_manager/       # 奖励函数管理
├── protocol.py           # DataProto 数据协议
├── base_config.py        # 配置基类
└── utils/                # 工具库（50+ 模块）
```

### Worker 架构模式

所有 Worker 继承自 `verl/workers/base_worker.py`，使用 `@register` 装饰器定义远程方法：

```python
from verl.protocol import DataProto
from verl.utils.py_functional import register_dispatch_mode

class MyWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def remote_method(self, data: DataProto) -> DataProto:
        # 分布式执行逻辑
        pass
```

**主要 Worker 类型**：
- `ActorRolloutRefWorker`: Actor + Rollout + Reference 混合引擎
- `FSDPTrainingWorker`: FSDP 训练后端
- `MegatronTrainingWorker`: Megatron 训练后端
- `RewardModelWorker`: 奖励模型 Worker

### 配置系统

使用 **Hydra** 进行配置管理，配置文件位于 `verl/trainer/config/`：

```bash
# 生成 _generated_*.yaml 配置文件
scripts/generate_trainer_config.sh
```

配置层级结构（默认组合模式）：
```yaml
defaults:
  - actor@actor_rollout_ref.actor: dp_actor
  - critic@critic: dp_critic
  - rollout@actor_rollout_ref.rollout: rollout
  - _self_
```

配置覆盖：
```bash
python -m verl.trainer.main_ppo \
  trainer.n_gpus_per_node=8 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
  data.train_batch_size=256
```

## 开发工作流

### 环境安装

```bash
# 基础安装（用于纯 Python 开发）
pip install -e ".[test]"

# vLLM 推理引擎
pip install -e ".[test,vllm]"

# SGLang 推理引擎
pip install -e ".[test,sglang]"

# Megatron 后端
pip install -e ".[mcore]"

# 数学奖励验证
pip install -e ".[math]"
```

### 代码规范检查

**Pre-commit 安装**：
```bash
pip install pre-commit
pre-commit install
```

**运行检查**：
```bash
# 检查暂存文件
pre-commit run

# 检查所有文件
pre-commit run --all-files

# 运行特定检查
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

**检查项**：
- `ruff`: 代码格式化和 Lint（line_length=120）
- `mypy`: 类型检查（仅特定模块启用）
- `autogen-trainer-cfg`: 配置文件自动生成验证
- `check-docstrings`: 文档字符串覆盖率
- `check-license`: Apache 2.0 许可证头检查
- `compileall`: Python 语法编译验证

### 测试

**测试目录**：
```
tests/
├── trainer/              # Trainer 相关测试
├── workers/              # Worker 相关测试
├── special_distributed/  # 多 GPU 测试
├── special_e2e/          # 端到端训练测试
├── special_npu/          # NPU 设备测试
├── special_sanity/       # 快速完整性检查
└── **/test_*_on_cpu.py   # CPU 专用测试
```

**运行测试**：
```bash
# 所有测试
pytest -q

# 特定类别
pytest tests/trainer -q
pytest tests/workers -q

# CPU 测试（快速）
pytest tests/**/test_*_on_cpu.py
```

**单个测试文件**：
```bash
pytest tests/workers/actor/test_dataproto.py -v
```

### 文档构建

```bash
cd docs
pip install -r requirements-docs.txt
make clean && make html
python -m http.server -d _build/html/
# 访问 http://localhost:8000
```

## 关键开发约定

### 命名约定

- 模块/函数：`snake_case`
- 类：`PascalCase`
- 测试文件：`test_*.py`
- CPU 测试：`test_*_on_cpu.py`

### PR 标题格式

```
[{modules}] {type}: {description}

# 示例
[fsdp, megatron] fix: handle gradient accumulation
[trainer] feat: add new reward function
[doc] chore: update installation guide
```

**类型**：`feat` / `fix` / `refactor` / `chore` / `test`

### YAML 配置规范

在 `verl/trainer/config/` 中：
- YAML 字段上方必须有注释
- 字段间必须有空行
- 禁止行内注释

### DataProto 使用

DataProto 是模块间数据传输的核心协议：

```python
from verl.protocol import DataProto

# 创建 DataProto
data = DataProto.from_dict(tensors={
    'input_ids': tensor,
    'attention_mask': tensor
})

# 批处理和切片
batch = data.batch(batch_size)
subset = data[:10]

# 分布式收集
gathered = data.all_gather()
```

**典型字段**：
- `input_ids`, `attention_mask`: 输入数据
- `responses`, `log_probs`: 生成响应
- `token_level_scores`, `token_level_rewards`: 奖励信号
- `old_log_probs`, `ref_log_prob`: PPO 相关

### Ray 资源池配置

灵活的设备映射用于不同部署场景：

```python
resource_pool_spec = {
    "global_pool": [0, 1, 2, 3, 4, 5, 6, 7]  # GPU 列表
}

mapping = {
    Role.ActorRollout: "global_pool",
    Role.Critic: "global_pool",
    Role.Rollout: "global_pool"
}
```

## 支持的训练后端

### FSDP/FSDP2

推荐使用 FSDP2（PyTorch 官方推荐）：
```yaml
actor_rollout_ref.actor.strategy: fsdp2
critic.strategy: fsdp2
```

启用 CPU Offload：
```yaml
actor_rollout_ref.actor.fsdp_config.offload_policy: true
```

### Megatron-LM

用于超大规模模型（如 DeepSeek-671B）：
```bash
pip install -e ".[mcore]"
```

## 支持的推理引擎

### vLLM

```bash
pip install -e ".[test,vllm]"
```

**注意**：避免 vllm 0.7.x（有 OOM bug），推荐 >= 0.8.2

### SGLang

```bash
pip install -e ".[test,sglang]"
```

SGLang 提供独特功能：
- 多轮对话 RL
- Agent RL
- VLM RLHF
- Server-based RL

## RL 算法支持

### 主流算法（内置）

- **PPO**: Proximal Policy Optimization（需要 Critic）
- **GRPO**: Group Relative Policy Optimization（无需 Critic）
- **RLOO**: REINFORCE with Leave-One-Out
- **ReMax**: 直接优化最大奖励
- **REINFORCE++**: 增强版 REINFORCE

### 前沿算法（recipe 子模块）

- **DAPO**: SOTA 数学推理 RL 算法
- **PF-PPO**: Policy Filtration PPO（ICML 2025）
- **VAPO**: Value-based Augmented PPO
- **PRIME**: Process Reinforcement through Implicit Rewards
- **DrGRPO**: Direct GRPO

## 性能优化特性

### 3D-HybridEngine

消除训练-推理转换的内存冗余，显著减少通信开销。

### 序列长度平衡

动态 workload 分配，减少 padding 浪费：
```bash
examples/ppo_trainer/run_qwen2-7b_seq_balance.sh
```

### 高级特性

- Sequence Parallelism（Ulysses）
- Flash Attention 2
- Liger Kernel
- FP8 训练
- LoRA 支持（节省显存）

## 常见开发任务

### 添加新的 RL 算法

1. 在 `verl/trainer/ppo/` 中添加算法实现
2. 在 `verl/trainer/config/algorithm/` 中添加配置
3. 在 `examples/` 中添加示例脚本

### 添加新的奖励函数

1. 在 `verl/workers/reward_manager/` 中实现
2. 在配置中指定 `reward_manager.reward_manager.name`

### 扩展到新模型

- FSDP 后端：参考 [FSDP 扩展指南](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
- Megatron 后端：参考 [Megatron 扩展指南](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)

## 重要资源

- **官方文档**: https://verl.readthedocs.io/
- **GitHub**: https://github.com/volcengine/verl
- **Slack 社区**: https://verl-project.slack.com
- **Breaking Changes**: https://github.com/volcengine/verl/discussions/2270
- **Q3 Roadmap**: https://github.com/volcengine/verl/issues/2388
