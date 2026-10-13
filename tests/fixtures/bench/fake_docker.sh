#!/bin/sh
# fake_docker.sh — 模拟 docker 客户端（docker 运行时生命周期测试，无真实 docker/GPU）。
# 每次调用把完整命令行追加到 FAKE_DOCKER_STATE_FILE（用于断言 rm 先于 run）；
#   rm -f 容器      → 无操作（幂等）
#   run … serve …  → 打印 KV 校准行并起 /health=200 的服务常驻（docker run attach 语义）
#   run … bench …  → 打印样例 bench 表格并把 saved JSON 写进 -v host:container 的宿主侧目录
#   FAKE_DOCKER_CRASH=1 时 serve 立即 exit 1（crash 路径）

STATE_FILE="${FAKE_DOCKER_STATE_FILE:-/tmp/fake_docker_state.txt}"
mkdir -p "$(dirname "$STATE_FILE")"
printf 'docker %s\n' "$*" >> "$STATE_FILE"

# rm -f：幂等清理，无操作
if [ "$1" = "rm" ]; then
  exit 0
fi

# 解析：--port <n>、-v <host>:<container>（取宿主侧）、--entrypoint vllm <image> <serve|bench>
SUB=""; PORT=""; RDIR_HOST=""; prev=""; entry_seen=""; entry_tok=0
for a in "$@"; do
  if [ "$prev" = "--port" ]; then PORT=$a; fi
  if [ "$prev" = "-v" ]; then RDIR_HOST="${a%%:*}"; fi
  if [ "$a" = "--entrypoint" ]; then
    entry_seen=1
    entry_tok=0
    prev=$a
    continue
  fi
  if [ "$entry_seen" = "1" ]; then
    entry_tok=$((entry_tok+1))
    if [ "$entry_tok" -eq 3 ]; then SUB=$a; fi   # entrypoint值(vllm)=1, image=2, 子命令=3
  fi
  prev=$a
done

if [ "$SUB" = "serve" ]; then
  if [ "${FAKE_DOCKER_CRASH:-0}" = "1" ]; then
    echo "fake docker serve crashing" >&2
    exit 1
  fi
  exec python3 - "$PORT" <<'EOF'
import http.server
import socketserver
import sys

port = int(sys.argv[1])
print("GPU KV cache size: 1,234,567 tokens, Maximum concurrency for 204,800 tokens per request: 6.02x",
      flush=True)


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *a):
        pass


class S(socketserver.TCPServer):
    allow_reuse_address = True


S(("127.0.0.1", port), H).serve_forever()
EOF
fi

# bench：打印样例表格 + 把 saved JSON 写进宿主侧挂载目录（模拟 -v host:/bench_results）
python3 - "$RDIR_HOST" <<'EOF'
import json
import os
import sys
import time

rdir = sys.argv[1]
os.makedirs(rdir, exist_ok=True)
count_file = os.path.join(rdir, ".count")
n = 0
try:
    with open(count_file) as f:
        n = int(f.read().strip())
except OSError:
    pass
with open(count_file, "w") as f:
    f.write(str(n + 1))

goodput = 12.34 + n * 1.0
print("============ BENCHMARK RESULTS ============")
print("Successful requests:                     1000")
print("Output token throughput (tok/s):      12345.6")
print("P99 TTFT (ms):                           620.1")
print("P99 TPOT (ms):                            18.3")
print(f"Request goodput (req/s):                 {goodput:.2f}")

name = os.path.join(rdir, f"result-{int(time.time() * 1000)}-{os.getpid()}.json")
json.dump({"request_goodput": goodput, "output_throughput": 12345.6,
           "p99_ttft_ms": 620.1, "p99_tpot_ms": 18.3}, open(name, "w"))
EOF
