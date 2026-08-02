#!/bin/bash
# PingPong AI Analyst - 服务器启动脚本
# 使用: bash start.sh [start|stop|restart|status]

APP_DIR="/opt/pingpong_analyst"
CONDA_ENV="pingpong"
CONDA_SH="/root/anaconda3/etc/profile.d/conda.sh"
PORT=8077
PID_FILE="$APP_DIR/logs/api.pid"
LOG_FILE="$APP_DIR/logs/api.log"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

# 确保 conda 环境的 ffmpeg/bin 在 PATH 前面
export PATH="$CONDA_PREFIX/bin:$PATH"

mkdir -p "$APP_DIR/logs" "$APP_DIR/output/clips" "$APP_DIR/data/uploads"

case "$1" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "服务已在运行, PID: $(cat $PID_FILE)"
            exit 1
        fi
        cd "$APP_DIR"
        nohup python -m pingpong_analyst.api --host 0.0.0.0 --port $PORT > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "服务已启动, PID: $(cat $PID_FILE), 端口: $PORT"
        sleep 2
        curl -s http://127.0.0.1:$PORT/api/health || echo "等待启动中..."
        ;;
    stop)
        if [ -f "$PID_FILE" ]; then
            kill $(cat "$PID_FILE") 2>/dev/null
            rm -f "$PID_FILE"
            echo "服务已停止"
        else
            echo "服务未运行"
        fi
        ;;
    restart)
        $0 stop
        sleep 1
        $0 start
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "运行中, PID: $(cat $PID_FILE)"
            curl -s http://127.0.0.1:$PORT/api/health
            echo ""
        else
            echo "未运行"
        fi
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
