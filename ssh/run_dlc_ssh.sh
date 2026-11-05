#!/usr/bin/env bash
# DLC 任务入口：保持 SSH 服务在前台运行，日志进入 DLC 任务日志。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/activate_connection.sh" foreground
