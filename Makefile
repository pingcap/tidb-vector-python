GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

format:	## Run code autoformatters (black).
	pre-commit install
	pre-commit run black --all-files

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	pre-commit install && pre-commit run --all-files --show-diff-on-failure

test:	## Run tests via pytest.
	pytest tests
