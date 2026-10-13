#!/bin/sh
# fake_serve.sh — 模拟 vllm serve（供 bench.py 生命周期测试）：
# 打印 docs §3 的校准日志行，起一个 /health 返回 200 的 HTTP 服务，然后常驻等待被清理。
# 用法：fake_serve.sh --model <id> --port <port>（--port 缺省 8000）

PORT=8000
prev=""
for a in "$@"; do
  if [ "$prev" = "--port" ]; then PORT=$a; fi
  prev=$a
done

exec python3 - "$PORT" <<'EOF'
import http.server
import socketserver
import sys

port = int(sys.argv[1])
# 校准日志行（vllm/v1/core/kv_cache_utils.py，见 docs/vllm-facts.md §3）
print("GPU KV cache size: 1,234,567 tokens, Maximum concurrency for 204,800 tokens per request: 6.02x",
      flush=True)


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
        else:
            self.send_response(404)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *a):  # 安静，不刷屏 run.log
        pass


class S(socketserver.TCPServer):
    allow_reuse_address = True


S(("127.0.0.1", port), H).serve_forever()
EOF
