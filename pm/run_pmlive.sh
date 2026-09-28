#!/usr/bin/env bash
# Run the Polymarket capture and its dashboard as one foreground process tree,
# suitable for `systemd Type=simple`.
#
#   collector : restarted every CYCLE_H hours.  The restart is not a workaround --
#               it is how a newly-discovered event joins the capture, because a
#               websocket subscription is fixed for the life of the connection.
#   dashboard : long-lived on PORT, restarted if it ever dies.
#
# SIGTERM/SIGINT are forwarded to the collector so it flushes its in-memory bars
# before exit; anything buffered since the last hourly write would otherwise be
# lost.  systemd's default KillMode sends TERM to the whole group, so the unit
# below sets KillMode=mixed and a TimeoutStopSec long enough for that flush.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
CYCLE_H=${CYCLE_H:-6}
TOP=${TOP:-150}
SPORTS_CAP=${SPORTS_CAP:-40}
PORT=${PORT:-2217}
HOST=${HOST:-0.0.0.0}
LOG=data/pmlive/run.log

mkdir -p data/pmlive
[ -x "$PY" ] || { echo "no interpreter at $PY (set PY=...)" >&2; exit 1; }

log(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

DASH_PID=""; COLL_PID=""; STOPPING=0

start_dash(){
  "$PY" pm/pmlive_dash.py --port "$PORT" --host "$HOST" >>"$LOG" 2>&1 &
  DASH_PID=$!
  log "dashboard up on http://${HOST}:${PORT} (pid $DASH_PID)"
}

shutdown(){
  [ "$STOPPING" = 1 ] && return
  STOPPING=1
  log "shutdown requested; flushing"
  if [ -n "$COLL_PID" ] && kill -0 "$COLL_PID" 2>/dev/null; then
    kill -TERM "$COLL_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$COLL_PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$COLL_PID" 2>/dev/null && { log "collector slow; KILL"; kill -KILL "$COLL_PID" 2>/dev/null; }
  fi
  [ -n "$DASH_PID" ] && kill -TERM "$DASH_PID" 2>/dev/null
  log "stopped"
  exit 0
}
trap shutdown TERM INT

start_dash
log "starting capture: top-$TOP, sports cap $SPORTS_CAP, ${CYCLE_H}h cycles"

while [ "$STOPPING" = 0 ]; do
  if ! kill -0 "$DASH_PID" 2>/dev/null; then
    log "dashboard died; restarting"
    start_dash
  fi

  if ! "$PY" pm/pmlive_universe.py --top "$TOP" --sports-cap "$SPORTS_CAP" >>"$LOG" 2>&1; then
    log "universe refresh failed; continuing with the existing universe"
  fi

  "$PY" pm/pmlive_collect.py --minutes $((CYCLE_H * 60)) >>"$LOG" 2>&1 &
  COLL_PID=$!
  log "collector cycle started (pid $COLL_PID, ${CYCLE_H}h)"
  wait "$COLL_PID"; rc=$?
  COLL_PID=""
  [ "$STOPPING" = 1 ] && break
  log "collector cycle ended rc=$rc"
  [ "$rc" -ne 0 ] && sleep 10
done
