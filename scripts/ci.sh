#!/usr/bin/env bash
# CI entry point, same on a GitHub runner and a local Ubuntu VM: install, fetch the one Mojo
# dependency (mojo-json v0.1.0) into lib/, then the gate.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "FAIL setup: uv not installed"; exit 1; }
uv sync || { echo "FAIL setup: uv sync"; exit 1; }
if [[ ! -e lib/json ]]; then
  git clone -q --depth 1 --branch v0.1.0 https://github.com/amarbaro/mojo-json .work/deps/mojo-json \
    || { echo "FAIL setup: clone mojo-json v0.1.0"; exit 1; }
  mkdir -p lib && ln -s ../.work/deps/mojo-json/src/json lib/json
fi
python3 -c 'import platform; print("oracle python", platform.python_version())'
scripts/check.sh
