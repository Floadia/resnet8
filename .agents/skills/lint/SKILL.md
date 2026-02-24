---
name: lint
description: Run lint checks exactly as GitHub Actions
---

Run these commands in order:

```bash
uv sync --group dev
uv run ruff check --fix .
uv run ruff format .
```
