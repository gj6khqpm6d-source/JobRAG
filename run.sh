#!/bin/sh
set -eu
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "首次运行：正在创建 Python 3.11 虚拟环境…"
  uv venv --python 3.11 .venv
fi

if ! .venv/bin/python -c 'import importlib.util,sys; names=("fastapi","jobspy","uvicorn","sqlalchemy","alembic","pgvector","psycopg","sentence_transformers","torch"); sys.exit(any(importlib.util.find_spec(name) is None for name in names))' 2>/dev/null; then
  echo "正在安装 JobSpy 和本地 Web 服务依赖…"
  uv pip install --python .venv/bin/python -e . fastapi 'uvicorn[standard]' openpyxl sqlalchemy alembic 'psycopg[binary]' pgvector sentence-transformers torch 'transformers>=4.41,<5' 'numpy>=1.26,<2'
fi
exec .venv/bin/uvicorn app.server:app --host 127.0.0.1 --port "${JOBSPY_PORT:-8000}"
