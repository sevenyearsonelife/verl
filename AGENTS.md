# Repository Guidelines

## 项目结构与模块组织

- `verl/`：核心 Python 包（训练器、workers、models、`experimental/` 等）。
- `tests/`：pytest 测试；按子命名空间分目录（如 `tests/trainer/` 对应 `verl/trainer/`）。`special_*` 用于分布式、多 GPU、端到端与快速校验等场景（见 `tests/README.md`）。
- `docs/`：文档与构建脚本（ReadTheDocs/Sphinx）。
- `examples/`：可运行示例与训练脚本。
- `scripts/`：维护与开发脚本（例如生成 `verl/trainer/config/_generated_*.yaml`）。
- `docker/`：容器化/环境相关文件；CI 定义在 `.github/workflows/`。
- `recipe/`：配方仓库的 git submodule；首次使用需初始化：
  ```bash
  git submodule update --init --recursive recipe
  ```

## 构建、测试与本地开发命令

- 安装（Python >= 3.10）：
  ```bash
  pip install -e ".[test]"
  # 可选：按后端安装
  pip install -e ".[test,vllm]"   # vLLM
  pip install -e ".[test,sglang]" # SGLang
  ```
- 代码规范（必跑）：本仓库使用 `pre-commit`，其中包含 `ruff`/`ruff-format`、`mypy`、配置自动生成、docstring/license/compileall 校验：
  ```bash
  pre-commit install
  pre-commit run --all-files --show-diff-on-failure --color=always
  ```
- 单元测试（pytest）：
  ```bash
  pytest -q
  # 仅跑某一类/单测文件
  pytest tests/trainer -q
  ```
- 构建文档：
  ```bash
  pip install -e ".[test]"
  cd docs && pip install -r requirements-docs.txt
  make clean && make html
  ```

## 代码风格与命名约定

- Python：4 空格缩进；格式化交给 `ruff-format`，lint 规则由 `ruff` 管理（`pyproject.toml` 中 `line-length=120`）。
- 命名：模块/函数使用 `snake_case`，类使用 `PascalCase`；测试文件遵循 `test_*.py`。

## 测试指南

- 优先在对应子目录新增测试（例如改动 `verl/interactions/`，测试放到 `tests/interactions/`）。
- CPU 专用测试文件以 `*_on_cpu.py` 结尾；其余测试默认按 GPU 资源执行（CI 也按此区分）。

## Commit 与 Pull Request 指南

- PR 标题需符合模板（CI 会检查）：`[{modules}] {type}: {description}`，例如 `[megatron, fsdp] fix: handle ...`（详见 `.github/PULL_REQUEST_TEMPLATE.md`）。
- Git 历史中常见提交信息风格为 `feat/fix/chore/test` 等类型前缀（例如 `fix(sft_trainer): ...` 或 `[misc] fix: ...`）；建议保持一致且描述具体。
- PR 需包含：变更动机/影响范围、可复现步骤或结果（无法走 CI 的改动需给出实验/指标）、必要时更新 `docs/` 与相关 workflow 覆盖。

## 配置与安全提示

- 依赖与硬件环境差异较大：按需参考 `requirements.txt`、`requirements-test.txt`、`requirements-cuda.txt`、`requirements-npu.txt` 等文件，避免在通用路径中硬编码设备/驱动假设。
- 不要提交密钥与账号信息；涉及第三方服务（如 wandb 等）的 token 使用环境变量或本地配置注入，并确保 PR 描述中不包含敏感数据。
