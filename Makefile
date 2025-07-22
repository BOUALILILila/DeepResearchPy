SHELL=/bin/bash

export PYTHONPATH:=${PYTHONPATH}:${PWD}/src

COVERAGE_THRESHOLD ?= 90

REPORT_OUTPUT_DIRECTORY = "reports"
COVERAGE_HTML_DIR = ./${REPORT_OUTPUT_DIRECTORY}/html_coverage
TESTS_HTML_DIR = ./${REPORT_OUTPUT_DIRECTORY}/html_unit_tests


.PHONY: uv
uv:  ## Install uv if it's not present.
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: install
install:
	@uv sync --frozen

.PHONY: install-dev
install-dev:
	@uv sync --dev

.PHONY: format
format:
	@uv run ruff format src/ tests/

.PHONY: lint
lint:
	@uv run ruff check src/ tests/

.PHONY: pre-commit
pre-commit:
	@uv run pre-commit run --all-files

.PHONY: sort-imports
sort-imports:
	@uv run ruff check --fix --select I src/ tests/

.PHONY: coverage
coverage:
	@uv run python -m coverage run -m pytest tests --html=${TESTS_HTML_DIR}/html_report.html --self-contained-html

.PHONY: coverage-report-html
coverage-report-html:
	@uv run python -m coverage html -d ${COVERAGE_HTML_DIR}

.PHONY: reports
reports: coverage coverage-report-html
	@uv run python -m coverage report --fail-under=${COVERAGE_THRESHOLD}
