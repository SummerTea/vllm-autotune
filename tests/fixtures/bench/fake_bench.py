#!/usr/bin/env python3
"""fake_bench.py — 模拟 `vllm bench serve`（供 bench.py 生命周期测试）：
打印样例 stdout 表格并写入 --save-result 风格 JSON；goodput 随重复次数递增
（12.34 / 13.34 …），让 bench.py 的 repeats 取中位数逻辑真正被检验。"""

import json
import os
import sys
import time

args = sys.argv[1:]
rdir = None
for i, a in enumerate(args):
    if a == "--result-dir" and i + 1 < len(args):
        rdir = args[i + 1]
if not rdir:
    sys.exit("fake_bench.py: 缺少 --result-dir")

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
table = (
    "============ BENCHMARK RESULTS ============\n"
    "Successful requests:                     1000\n"
    "Failed requests:                             0\n"
    "Request throughput (req/s):              100.0\n"
    "Output token throughput (tok/s):      12345.6\n"
    "Mean TTFT (ms):                          420.1\n"
    "P99 TTFT (ms):                           620.1\n"
    "Mean TPOT (ms):                           12.3\n"
    "P99 TPOT (ms):                            18.3\n"
    f"Request goodput (req/s):                 {goodput:.2f}\n"
)
print(table, end="", flush=True)

name = os.path.join(rdir, f"result-{int(time.time() * 1000)}-{os.getpid()}.json")
json.dump(
    {
        "duration": 90.0,
        "completed": 1000,
        "failed": 0,
        "request_throughput": 100.0,
        "request_goodput": goodput,
        "output_throughput": 12345.6,
        "total_output_tokens": 1000000,
        "p99_ttft_ms": 620.1,
        "p99_tpot_ms": 18.3,
    },
    open(name, "w"),
)
