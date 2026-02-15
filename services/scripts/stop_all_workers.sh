#!/bin/bash
# Stop all Genesis worker services

echo "Stopping all Genesis workers..."

if [ -f "logs/workers/pids.txt" ]; then
    while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Stopping worker (PID: $pid)..."
            kill $pid
        fi
    done < logs/workers/pids.txt

    rm logs/workers/pids.txt
    echo "✓ All workers stopped"
else
    echo "No PID file found. Workers may not be running."
    echo "Trying to kill by port..."

    for port in 8080 8081 8082 8083; do
        pid=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$pid" ]; then
            echo "  Killing process on port $port (PID: $pid)..."
            kill $pid 2>/dev/null || true
        fi
    done
fi

echo "✓ Cleanup complete"
