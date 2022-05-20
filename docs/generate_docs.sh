#!/usr/bin/env sh

set -e

# Ensure current path is project root
cd "$(dirname "$0")/../"

pip install poetry
poetry build -f wheel
pip install dist/$(ls -1 dist | grep .whl)

poetry run sphinx-apidoc -f -e -o docs/source quaterion_models
poetry run sphinx-build docs/source docs/html
