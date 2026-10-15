#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  if .venv/bin/python -m notebook --version >/dev/null 2>&1; then
    exec .venv/bin/python -m notebook \
      --ServerApp.root_dir="$PWD" \
      --ServerApp.ip=127.0.0.1 \
      demo.ipynb
  fi
fi

cat <<'EOF'
Jupyter is not available in this demo folder's local .venv.

This machine's bare `python` and `jupyter` commands may point to a broken pyenv
installation. Avoid:

  python demo.ipynb
  jupyter notebook demo.ipynb

Run the notebook-equivalent terminal demo now with:

  /usr/bin/python3 run_demo.py

To enable the browser notebook in this folder, create a local venv and install
Jupyter with:

  /usr/bin/python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install notebook
  ./launch_notebook.sh
EOF
