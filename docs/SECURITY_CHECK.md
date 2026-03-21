# SECURITY_CHECK

## 当前状态

- 仓库中不应包含真实 API Key
- `.env` 仅用于本地，不应提交
- `env.example` 保留占位符

## 提交前检查

1. 检查敏感信息：

```bash
rg -n "sk-|api_key|secret|token" .
```

2. 确认 `.env` 未被跟踪：

```bash
git status --short
```

3. 如误提交 `.env`，执行：

```bash
git rm --cached .env
git commit -m "chore: stop tracking .env"
```

## 建议

- 使用环境变量注入密钥
- 在 CI 中增加 secret scan（如 gitleaks）
- 禁止在日志里输出完整凭据
