#!/usr/bin/env bash
# stop.sh — Stop the Bonsai 27B server
set -euo pipefail

PID_FILE="${HOME}/.bonsai/llama-server.pid"

if [[ -f "${PID_FILE}" ]]; then
  PID=$(cat "${PID_FILE}")
  if kill "${PID}" 2>/dev/null; then
    echo "✔  Stopped llama-server (PID ${PID})"
  else
    echo "   (Process ${PID} not running — cleaning up pid file)"
  fi
  rm -f "${PID_FILE}"
else
  # Fallback: find and kill any llama-server process
  PIDS=$(pgrep -f llama-server 2>/dev/null || true)
  if [[ -n "${PIDS}" ]]; then
    echo "   Stopping llama-server (PID(s): ${PIDS})..."
    kill ${PIDS} 2>/dev/null || true
    echo "✔  Stopped"
  else
    echo "   No llama-server process found."
  fi
fi
