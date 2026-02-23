# 将 Doc Thinker 上传到 GitHub

按下面步骤即可把本仓库打包并上传为开源项目 **doc-thinker**。

## 1. 替换仓库链接（可选）

上传前请把 `pyproject.toml` 里 `[project.urls]` 中的 `your-username` 改成你的 GitHub 用户名或组织名，例如：

```ini
Homepage = "https://github.com/你的用户名/doc-thinker"
Repository = "https://github.com/你的用户名/doc-thinker"
Issues = "https://github.com/你的用户名/doc-thinker/issues"
```

## 2. 在 GitHub 上新建仓库

1. 登录 [GitHub](https://github.com)，点击右上角 **+** → **New repository**。
2. **Repository name** 填：`doc-thinker`（或你喜欢的名字）。
3. 选 **Public**，**不要**勾选 “Add a README” / “Add .gitignore”（本地已有）。
4. 点 **Create repository**。

## 3. 本地打包并推送到 GitHub

在项目根目录（`doc`）打开终端，执行：

```bash
# 若尚未初始化 git
git init

# 添加所有文件（.gitignore 会排除 .venv、rag_storage_api 等）
git add .
git commit -m "chore: Doc Thinker open source release"

# 添加远程仓库（把 你的用户名 换成你的 GitHub 用户名）
git remote add origin https://github.com/你的用户名/doc-thinker.git

# 推送（若已存在 main 分支可直接 push）
git branch -M main
git push -u origin main
```

若仓库已存在且已有历史，可先 `git remote add origin ...`，再 `git push -u origin main`。

## 4. 打标签发版（可选）

```bash
git tag -a v1.0.0 -m "Doc Thinker v1.0.0"
git push origin v1.0.0
```

## 5. 打包成安装包（可选）

需要发 PyPI 或给别人离线安装时：

```bash
pip install build
python -m build
```

会在 `dist/` 下生成 `*.whl` 和 `*.tar.gz`。若仅上传 GitHub，可跳过此步。

---

上传完成后，在仓库 **Settings → General** 里可填写 Description 和 Website，并可在 **About** 中上传 `logo.png` 作为仓库头像（与 README 中的 logo 一致）。
