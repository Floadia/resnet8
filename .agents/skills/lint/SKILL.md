---
name: lint
description: Lint this project 
---

## Steps

Fix until pass all of these

```bash
uv run ruff check .
uv run ruff format --check .
uv run ruff check --fix .
uv run ruff format .
```
