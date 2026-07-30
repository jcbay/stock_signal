#!/bin/bash
# 股票信号系统 - 停止脚本
# 用法: ./stop.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$DIR/.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "ℹ️  服务未在运行"
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    sleep 1
    # 如果还没死，强杀
    if kill -0 "$PID" 2>/dev/null; then
        kill -9 "$PID"
        sleep 1
    fi
    if kill -0 "$PID" 2>/dev/null; then
        echo "❌ 无法停止进程 (PID: $PID)，请手动处理: kill -9 $PID"
        exit 1
    else
        echo "✅ 服务已停止 (PID: $PID)"
    fi
else
    echo "ℹ️  服务未在运行"
fi

rm -f "$PID_FILE"
