#!/bin/sh
# fake_sleep_serve.sh — 常驻但不提供 /health 的假 serve：
# 触发 startup_timeout，并验证 bench.py 的清理逻辑真的杀掉了它。
echo "no health server here"
while true; do sleep 5; done
