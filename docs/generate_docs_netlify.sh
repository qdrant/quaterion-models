#!/usr/bin/env sh

set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

# install CPU torch, cause it is smaller
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

pip install poetry
poetry build -f wheel
pip install dist/$(ls -1 dist | grep .whl)
poetry install --extras "fasttext"

pip install sphinx>=4.4.0
pip install "git+https://github.com/qdrant/qdrant_sphinx_theme.git@master#egg=qdrant-sphinx-theme"

sphinx-build docs/source docs/html
