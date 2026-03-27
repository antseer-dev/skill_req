#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.server.pid"
LOG_FILE="$SCRIPT_DIR/server.log"

# Load .env early so HOST/PORT are available to all subcommands
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"

load_env() {
    if [ -z "$OPENROUTER_KEY" ]; then
        echo "ERROR: OPENROUTER_KEY is not set. Copy .env.example to .env and fill in values."
        exit 1
    fi
}

cmd_start() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Service already running (PID $PID)"
            exit 0
        else
            rm -f "$PID_FILE"
        fi
    fi

    load_env
    echo "Starting Skill Req API on ${HOST}:${PORT} ..."
    nohup uvicorn api:app --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started (PID $(cat $PID_FILE)). Logs: $LOG_FILE"
}

cmd_stop() {
    # Kill by PID file
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Stopped (PID $PID)"
        else
            echo "Process $PID not found, cleaning up PID file"
        fi
        rm -f "$PID_FILE"
    fi
    # Also kill any stale processes still holding the port
    STALE=$(lsof -ti:"$PORT" 2>/dev/null)
    if [ -n "$STALE" ]; then
        echo "Killing stale process(es) on port $PORT: $STALE"
        echo "$STALE" | xargs kill -9 2>/dev/null || true
    fi
    # Wait up to 5s for the port to be released
    for i in $(seq 1 10); do
        lsof -ti:"$PORT" > /dev/null 2>&1 || break
        sleep 0.5
    done
}

cmd_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Running (PID $PID) on ${HOST}:${PORT}"
        else
            echo "Dead (stale PID file)"
        fi
    else
        echo "Stopped"
    fi
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

case "${1:-start}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    status)  cmd_status ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
