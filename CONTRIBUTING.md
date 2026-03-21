# Contributing to Doc Thinker

欢迎为 Doc Thinker 贡献代码或文档。

## 开发环境

```bash
git clone https://github.com/Yang-Jiashu/doc-thinker.git
cd doc-thinker
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e ".[all]"
pip install pytest pytest-asyncio
```

## 运行测试

```bash
pytest tests/ -v
```

## 提交流程

1. Fork 本仓库，在本地创建分支（如 `feature/xxx` 或 `fix/xxx`）。
2. 修改后运行测试，确认通过。
3. 提交 commit，推送到你 fork 的仓库，再打开 Pull Request 到主仓库的 `main` 分支。
4. 维护者 review 通过后会合并。

## 代码风格

- 推荐使用 [ruff](https://github.com/astral-sh/ruff) 做格式与简单 lint（项目已配置 `pyproject.toml` 中的 `[tool.ruff]`）。

如有问题可在 GitHub Issues 提出。
