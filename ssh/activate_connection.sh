#!/usr/bin/env bash

# 从 lijingzheng/codex_local/activate_connection.sh 复制并独立定制。
# 在 DLC 上配置本目录的 id_rsa.pub，启动 SSH 服务供 DSW 转发访问。
#   bash activate_connection.sh check
#   bash activate_connection.sh foreground  # DLC 任务入口使用此模式
#   bash activate_connection.sh start
#   bash activate_connection.sh status
#   bash activate_connection.sh stop
#
# 本地 SSH 配置可通过 ProxyJump 连接：本地电脑 -> DSW -> DLC

set -uo pipefail
umask 077

DLC_SSH_PORT="${DLC_SSH_PORT:-2222}"
if [[ ! "$DLC_SSH_PORT" =~ ^[0-9]{1,5}$ ]] || \
   (( 10#$DLC_SSH_PORT < 1 || 10#$DLC_SSH_PORT > 65535 )); then
  echo "DLC_SSH_PORT must be an integer from 1 to 65535." >&2
  exit 2
fi
DLC_SSH_PORT="$((10#$DLC_SSH_PORT))"
DLC_SSH_PID="/tmp/lzx-dlc-sshd-${UID}-${DLC_SSH_PORT}.pid"
DLC_SSH_LOG="/tmp/lzx-dlc-sshd-${UID}-${DLC_SSH_PORT}.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_PUBLIC_KEY="${DLC_SSH_PUBLIC_KEY:-$SCRIPT_DIR/id_rsa.pub}"
AUTHORIZED_KEYS="/root/.ssh/authorized_keys"

read_public_key() {
  local key_line

  if [ ! -r "$CLIENT_PUBLIC_KEY" ] || [ ! -s "$CLIENT_PUBLIC_KEY" ]; then
    echo "DLC SSH failed: public key is missing or unreadable: $CLIENT_PUBLIC_KEY" >&2
    return 1
  fi
  # Only accept a public key. The private key stays on the local computer.
  if ! ssh-keygen -lf "$CLIENT_PUBLIC_KEY" >/dev/null 2>&1; then
    echo "DLC SSH failed: invalid public key: $CLIENT_PUBLIC_KEY" >&2
    return 1
  fi
  key_line="$(awk 'NF >= 2 && $1 ~ /^(ssh-|ecdsa-|sk-)/ { print $1, $2; exit }' "$CLIENT_PUBLIC_KEY")"
  if [ -z "$key_line" ]; then
    echo "DLC SSH failed: expected an OpenSSH public key (.pub), not a private key." >&2
    return 1
  fi
  printf '%s\n' "$key_line"
}

install_client_key() {
  local key_line key_type key_data
  key_line="$(read_public_key)" || return 1
  key_type="${key_line%% *}"
  key_data="${key_line#* }"

  mkdir -p /root/.ssh || return 1
  chmod 700 /root/.ssh || return 1
  touch "$AUTHORIZED_KEYS" || return 1
  chmod 600 "$AUTHORIZED_KEYS" || return 1

  if ! awk -v type="$key_type" -v data="$key_data" \
    '$1 == type && $2 == data { found=1 } END { exit !found }' "$AUTHORIZED_KEYS"; then
    printf '%s %s lzx-dlc-login\n' "$key_type" "$key_data" >> "$AUTHORIZED_KEYS" || return 1
    echo "Added client public key to $AUTHORIZED_KEYS."
  else
    echo "Client public key is already present in $AUTHORIZED_KEYS."
  fi
}

dlc_sshd_pid() {
  local pid executable cmdline

  [ -r "$DLC_SSH_PID" ] || return 1
  IFS= read -r pid < "$DLC_SSH_PID" || return 1
  case "$pid" in
    ''|*[!0-9]*) return 1 ;;
  esac
  kill -0 "$pid" 2>/dev/null || return 1

  executable="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
  case "$executable" in
    */sshd)
      cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" || return 1
      [[ "$cmdline" == *"PidFile=$DLC_SSH_PID "* ]] || return 1
      printf '%s\n' "$pid"
      ;;
    *) return 1 ;;
  esac
}

dlc_sshd_ready() {
  dlc_sshd_pid >/dev/null || return 1
  timeout 2 bash -c "</dev/tcp/127.0.0.1/$DLC_SSH_PORT" 2>/dev/null
}

prepare_dlc_sshd() {
  if [ "$UID" -ne 0 ]; then
    echo "DLC SSH failed: run this script as root inside the DLC container." >&2
    return 1
  fi
  sshd_bin="$(command -v sshd 2>/dev/null || true)"
  if [ -z "$sshd_bin" ]; then
    echo "DLC SSH failed: sshd is not installed." >&2
    return 1
  fi
  if ! install_client_key; then
    return 1
  fi

  mkdir -p /run/sshd || return 1
  ssh-keygen -A >/dev/null 2>&1 || return 1

  sshd_args=(
    -p "$DLC_SSH_PORT"
    -o "PidFile=$DLC_SSH_PID"
    -o ListenAddress=0.0.0.0
    -o AuthorizedKeysFile=/root/.ssh/authorized_keys
    -o PermitRootLogin=prohibit-password
    -o PubkeyAuthentication=yes
    -o PasswordAuthentication=no
    -o KbdInteractiveAuthentication=no
    -o ChallengeResponseAuthentication=no
    -o StrictModes=yes
  )
  "$sshd_bin" -t "${sshd_args[@]}" || return 1
}

show_connection() {
  echo "DLC container IP(s): $(hostname -I 2>/dev/null || true)"
  echo "SSH login: root; port: $DLC_SSH_PORT; connect via ProxyJump AtlantaDSW."
}

start_dlc_sshd() {
  local sshd_bin
  local -a sshd_args

  # Install the key even if this instance is already running.
  prepare_dlc_sshd || return 1
  if dlc_sshd_ready; then
    echo "DLC SSH is already ready on port $DLC_SSH_PORT."
    return 0
  fi
  rm -f "$DLC_SSH_PID"
  if ! "$sshd_bin" "${sshd_args[@]}" -E "$DLC_SSH_LOG"; then
    echo "DLC SSH failed. Log: $DLC_SSH_LOG" >&2
    return 1
  fi

  for _ in 1 2 3 4 5; do
    sleep 1
    if dlc_sshd_ready; then
      echo "DLC SSH is ready on port $DLC_SSH_PORT."
      show_connection
      return 0
    fi
  done

  echo "DLC SSH failed. Log: $DLC_SSH_LOG" >&2
  return 1
}

foreground_dlc_sshd() {
  local sshd_bin
  local -a sshd_args

  prepare_dlc_sshd || return 1
  if dlc_sshd_pid >/dev/null; then
    echo "This SSH instance is already running; stop it before using foreground mode." >&2
    return 1
  fi
  rm -f "$DLC_SSH_PID"
  show_connection
  echo "Starting SSH in foreground; keep this DLC task running."
  # Stay as the DLC task's main process, with SSH logs sent to the task log.
  exec "$sshd_bin" -D -e "${sshd_args[@]}"
}

check_dlc_sshd() {
  command -v sshd >/dev/null 2>&1 || {
    echo "DLC SSH failed: sshd is not installed in this image." >&2
    return 1
  }
  read_public_key >/dev/null || return 1
  echo "Public key is valid: $CLIENT_PUBLIC_KEY"
  ssh-keygen -lf "$CLIENT_PUBLIC_KEY" | awk '{ print "Key:", $1, $2, $NF }'
  echo "SSH server is installed; configured port: $DLC_SSH_PORT."
}

stop_dlc_sshd() {
  local pid

  if pid="$(dlc_sshd_pid)"; then
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$DLC_SSH_PID"
  echo "DLC SSH is stopped."
}

case "${1:-start}" in
  check)
    check_dlc_sshd
    ;;
  foreground)
    foreground_dlc_sshd
    ;;
  start)
    start_dlc_sshd
    ;;
  status)
    if dlc_sshd_ready; then
      echo "DLC SSH: ready on port $DLC_SSH_PORT"
    else
      echo "DLC SSH: not ready"
      exit 1
    fi
    ;;
  stop)
    stop_dlc_sshd
    ;;
  *)
    echo "Usage: $0 [check|foreground|start|status|stop]" >&2
    exit 2
    ;;
esac
