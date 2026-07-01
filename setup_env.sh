#!/bin/bash
# Create/activate the braidlab virtualenv and install dependencies.
#   source setup_env.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ ! -d .venv ]; then
    uv venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e . --no-deps
echo "braidlab env ready: $(python --version)"
