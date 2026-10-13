#!/usr/bin/env bash
# setup_remote.sh — 主机环境校验（docker-only，零破坏）：校验 docker 可用与镜像就绪，
# 不 pull、不安装。幂等：可安全重复运行；仅在 target.yaml 确认的 work_dir 内写入。
# 在编排机（本仓库目录）执行，经 ssh 作用到 head 主机。用法: bash setup_remote.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ── .env：手工解析而非 source（远端目录一律取自 target.yaml 的 work_dir）──
load_env() {
  [ -f .env ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      '' | '#'*) continue ;;
      *=*)
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
          SSH_PORT | SSH_KEY)
            printf -v "$key" '%s' "$value"
            # SSH_KEY 是本地路径：以 ~/ 开头时在本机展开
            case "$key:$value" in
              SSH_KEY:'~/'*) printf -v SSH_KEY '%s' "$HOME${value#\~}" ;;
            esac
            ;;
        esac
        ;;
    esac
  done < .env
}
load_env
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
if [ -n "$SSH_PORT" ]; then SSH_OPTS+=(-p "$SSH_PORT"); fi
if [ -n "$SSH_KEY" ]; then SSH_OPTS+=(-i "$SSH_KEY"); fi

# 解析 yaml 用的 python：优先仓库 venv（pyyaml 是已声明依赖），退化到 python3/python
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python; fi

sh_quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }

# 提取 head 主机信息：ssh 地址（1）/ work_dir（2）/ remote.vllm_python（3）/ docker 镜像（4）
extract_head() {
  "$PY" - <<'PYEOF'
import sys
try:
    import yaml
except ImportError:
    sys.exit("缺少 PyYAML：请先 `uv sync`（或 pip install pyyaml）")
def load(p):
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        sys.exit(f"未找到 {p}")
target = load("target.yaml")
heads = [h for h in target.get("hosts") or [] if h.get("role") == "head"]
if len(heads) != 1:
    sys.exit(f"target.yaml 必须恰好一个 role=head 的主机（当前 {len(heads)} 个）")
ssh_addr = heads[0].get("ssh")
if not ssh_addr:
    sys.exit("target.yaml 的 head 主机缺少 ssh 字段")
harness = load("harness.yaml")
vllm_python = (harness.get("remote") or {}).get("vllm_python") or "python"
runtime = harness.get("runtime") or {}
print(ssh_addr)
print(heads[0].get("work_dir") or "")
print(vllm_python)
print(runtime.get("image") or "")
PYEOF
}

usage() {
  cat <<'EOF'
用法: bash setup_remote.sh
  对 head 主机做零破坏环境校验（docker-only：校验 docker 可用与镜像就绪，不 pull、不安装）
  + 主机工具清单。--help | -h 显示本帮助。
EOF
}

main() {
  case "${1:-}" in
    --help | -h) usage; return 0 ;;
  esac
  local out head_ssh work_dir vllm_python runtime_image
  out="$(extract_head)" || exit 1
  head_ssh="$(printf '%s\n' "$out" | sed -n '1p')"
  work_dir="$(printf '%s\n' "$out" | sed -n '2p')"
  vllm_python="$(printf '%s\n' "$out" | sed -n '3p')"
  runtime_image="$(printf '%s\n' "$out" | sed -n '4p')"

  if [ -z "$work_dir" ]; then
    echo "请在 target.yaml 的 head 主机上提供 work_dir（主机专属目录，需用户确认；模板仅写此目录）" >&2
    exit 1
  fi
  if [ -z "$runtime_image" ]; then
    echo "需要 docker 镜像：在 config.yaml 或 harness.yaml 的 runtime.image 指定（如 vllm/vllm-openai:v0.31.0）" >&2
    exit 1
  fi

  echo "== setup_remote.sh：目标 head 主机 $head_ssh（python=$vllm_python，work_dir=$work_dir） =="
  if ssh "${SSH_OPTS[@]}" "$head_ssh" \
      "bash -s $(sh_quote "$work_dir") $(sh_quote "$vllm_python") $(sh_quote "$runtime_image")" <<'RMT'
# ── 远端脚本（在 head 主机上执行）──────────────────────────────
set -u
WORK_DIR="$1"; VLLM_PYTHON="$2"; RUNTIME_IMAGE="$3"
case "$WORK_DIR" in
  '~/'*) WORK_DIR="$HOME${WORK_DIR#\~}" ;;
esac
PASS=0; FAIL=0
ok()  { printf '  [PASS] %s\n' "$1"; PASS=$((PASS+1)); }
bad() { printf '  [FAIL] %s\n' "$1"; FAIL=$((FAIL+1)); }

echo "== 零破坏横幅 =="
echo "  本脚本对主机零破坏：仅在 $WORK_DIR 内写入；默认不 pull 镜像、不安装环境"

# ── docker-only：零 pull，只用主机上已预先准备的镜像 ──
echo "== [1/5] docker 检查 =="
DOCKER_FAIL=0
if docker --version >/dev/null 2>&1; then
  ok "docker 可用（$(docker --version)）"
else
  bad "docker 不可用（docker --version 失败）"
  DOCKER_FAIL=1
fi

echo "== [2/5] 镜像存在性检查（零 pull） =="
if [ "$DOCKER_FAIL" -eq 0 ] && docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
  ok "镜像已就绪: $RUNTIME_IMAGE"
else
  bad "镜像 $RUNTIME_IMAGE 未在主机上预先准备；本模板默认不 pull（保持主机零新增），请人工准备后再运行"
  DOCKER_FAIL=1
fi

echo "== [3/5] 容器入口验证 =="
if [ "$DOCKER_FAIL" -eq 0 ] && docker run --rm --gpus all --entrypoint vllm "$RUNTIME_IMAGE" --version >/dev/null 2>&1; then
  ok "docker run --gpus all --entrypoint vllm $RUNTIME_IMAGE --version"
else
  bad "容器 entrypoint vllm 不可用（镜像不含 vllm 控制台脚本？）"
  DOCKER_FAIL=1
fi

echo "== [4/5] WORK_DIR =="
if mkdir -p "$WORK_DIR"; then
  ok "WORK_DIR 就绪: $WORK_DIR"
else
  bad "创建 $WORK_DIR 失败"
fi

# ── [5/5] 主机工具清单（只读检测，零安装；信息性，不计入 PASS/FAIL）──────
echo ""
echo "== [5/5] 主机工具清单（只读；缺失不影响本模板默认运行） =="
check_tool() {  # $1 = 工具名  $2 = 检测命令  $3 = 用途
  if eval "$2" >/dev/null 2>&1; then
    printf '   %-14s present   (%s)\n' "$1" "$3"
  else
    printf '   %-14s MISSING  (%s)\n' "$1" "$3"
  fi
}
# 必需（docker-only）
check_tool "nvidia-smi" "command -v nvidia-smi" "必需：GPU 枚举/显存/时钟"
check_tool "python3"    "command -v python3"    "必需：远端运行 bench.py"
check_tool "pyyaml"     "$VLLM_PYTHON -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(\"yaml\") else 1)'" "必需：bench.py 依赖"
check_tool "docker"     "command -v docker"     "必需：容器执行"
# 推荐（v1.1 路线图相关，见 docs/roadmap.md）
check_tool "nsys"            "command -v nsys"            "推荐：Nsight Systems profiling"
check_tool "ib_write_bw"     "command -v ib_write_bw"      "推荐：perftest，RDMA 带宽"
check_tool "nvbandwidth"     "command -v nvbandwidth"       "推荐：NVLink/PCIe 实测带宽"
check_tool "all_reduce_perf" "command -v all_reduce_perf"   "推荐：nccl-tests 集合带宽"
check_tool "dcgmi"           "command -v dcgmi"             "推荐：DCGM 命令行"
check_tool "dcgm-exporter"   "command -v dcgm-exporter"     "推荐：DCGM Prometheus exporter(9400)"
check_tool "nvitop"          "command -v nvitop"            "推荐：GPU 监控 TUI/JSON"
check_tool "gpustat"         "command -v gpustat"           "推荐：GPU 监控（nvitop 替代）"
check_tool "prometheus"      "command -v prometheus"        "推荐：指标存储 + prometheus-mcp"
echo "  缺失工具不影响本模板默认运行；可按 docs/roadmap.md 补齐（推荐项为 v1.1 可选增强）"

printf '\nSUMMARY: %s（%d PASS / %d FAIL）\n' "$([ "$FAIL" -eq 0 ] && echo PASS || echo FAIL)" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
RMT
  then
    echo "== setup_remote.sh: PASS（head 主机环境就绪） =="
  else
    echo "== setup_remote.sh: FAIL（幂等：修复后可安全重跑） ==" >&2
    exit 1
  fi
}
main "$@"
