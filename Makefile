ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: help install uninstall reinstall checks build clean publish publish-test

help:
	@echo "torch-wheel-index"
	@echo ""
	@echo "Available targets:"
	@echo "  make install      - Install torch-wheel-index using uv tool"
	@echo "  make uninstall    - Remove torch-wheel-index from uv tools"
	@echo "  make reinstall    - Reinstall torch-wheel-index (useful during development)"
	@echo "  make checks       - Run mypy and ruff"
	@echo "  make build        - Build sdist and wheel into dist/"
	@echo "  make clean        - Remove dist/"
	@echo "  make publish      - Publish to PyPI (requires .env with UV_PUBLISH_TOKEN)"
	@echo "  make publish-test - Publish to TestPyPI (requires .env with UV_PUBLISH_TOKEN_TESTPYPI)"

install:
	cd $(ROOT_DIR) && uv tool install .

uninstall:
	uv tool uninstall torch-wheel-index

reinstall:
	cd $(ROOT_DIR) && uv tool install . --reinstall

checks:
	mypy src
	ruff format src
	ruff check src

build:
	uv build

clean:
	rm -rf dist/

publish: clean build
	@set -a && . ./.env && set +a && uv publish

publish-test: clean build
	@set -a && . ./.env && set +a && uv publish \
		--publish-url https://test.pypi.org/legacy/ \
		--token "$$UV_PUBLISH_TOKEN_TESTPYPI"
