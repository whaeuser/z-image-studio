#!/usr/bin/env bash
# Start Z-Image Studio dev server from local .venv
cd "$(dirname "$0")"
exec .venv/bin/zimg serve "$@"
