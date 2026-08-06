#!/bin/bash
# 股票信号系统 - 启动脚本
# 用法: ./start.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
PID_FILE="$DIR/.pid"
LOG_FILE="$DIR/server.log"
PYTHON="$DIR/venv/bin/python"

# 检查是否已在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "✅ 服务已在运行中 (PID: $OLD_PID)"
        echo "🌐 访问: http://localhost:8080"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

# 自动创建 venv（如果不存在）
if [ ! -f "$PYTHON" ]; then
    echo "📦 首次运行，正在创建虚拟环境..."
    python3 -m venv venv
    if [ ! -f "$PYTHON" ]; then
        echo "❌ 创建 venv 失败，请检查 python3 是否已安装"
        exit 1
    fi
    echo "📥 正在安装依赖..."
    "$DIR/venv/bin/pip" install -r requirements.txt -q
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败，请手动执行: venv/bin/pip install -r requirements.txt"
        exit 1
    fi
    echo "✅ 环境准备完成"
fi

# 启动服务
echo "🚀 正在启动..."
nohup "$PYTHON" app.py > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

# 等待启动
sleep 2

# 验证
if kill -0 "$NEW_PID" 2>/dev/null; then
    echo "✅ 启动成功! (PID: $NEW_PID)"
    echo "🌐 访问: http://localhost:8080"
    echo "📄 日志: tail -f $LOG_FILE"
    echo "🛑 停止: ./stop.sh"
else
    echo "❌ 启动失败，请查看日志:"
    tail -20 "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
