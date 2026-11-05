.PHONY: reproduce test

# Rebuild runs/results/metrics.json from the aggregate run outputs in runs/ and check it against the committed copy
# (no GPU, no patient data).
reproduce:
	scripts/repro/reproduce.sh

# Unit tests on synthetic fixtures (needs `uv sync` and the submodules).
test:
	uv run pytest -q
