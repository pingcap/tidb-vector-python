GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

format:	## Run code autoformatters (black).
	pre-commit install
	pre-commit run black --all-files

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	tox -e lint

test:
	tox
