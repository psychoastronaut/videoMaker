# Contributing

```bash
uv venv && uv pip install -e '.[dev,semantic,upscale]'
python -m pytest            # unit tests + a short end-to-end render smoke test (needs ffmpeg)
ruff check src tests
```

- Python 3.11+ syntax, line length 100 (`pyproject.toml`), ruff-clean; type hints on public functions.
- Add or update tests for behaviour changes. Tests must not need a GPU, Ollama or network.
- Config keys go in `config/default.yaml` with a comment, and in `docs/CONFIG.md`.
- Keep everything local: no cloud APIs. New stock sources must be CC0/public domain (or clearly flagged) and audited.
- Do not commit media, `.env`, `work/` or `out/`, or absolute personal paths.
- Check new model/weight licenses before adding them (see the credits in the README).
- Small focused PRs; describe how you tested.
