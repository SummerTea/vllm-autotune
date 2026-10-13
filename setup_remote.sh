#!/usr/bin/env bash
# setup_remote.sh — 针对 head 主机的一次性环境准备：pin vLLM 版本 + 装 uv + 建 REMOTE_DIR。
# 幂等：可安全重复运行；无破坏性操作。在编排机（本仓库目录）执行，经 ssh 作用到 head 主机。
# 用法: bash setup_remote.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ── .env：手工解析而非 source，保留字面值（REMOTE_DIR 的 ~ 留给远端 shell 展开）──
load_env() {
  [ -f .env ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      '' | '#'*) continue ;;
      *=*)
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
          SSH_PORT | SSH_KEY | REMOTE_DIR)
            printf -v "$key" '%s' "$value"
            # SSH_KEY 是本地路径：以 ~/ 开头时在本机展开（REMOTE_DIR 不做此处理）
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
REMOTE_DIR="${REMOTE_DIR:-~/vllm-autotune}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
if [ -n "$SSH_PORT" ]; then SSH_OPTS+=(-p "$SSH_PORT"); fi
if [ -n "$SSH_KEY" ]; then SSH_OPTS+=(-i "$SSH_KEY"); fi

# 解析 yaml 用的 python：优先仓库 venv（pyyaml 是已声明依赖），退化到 python3/python
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python; fi

sh_quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }

# 提取 head 主机 ssh 地址（第 1 行）与 remote.vllm_python（第 2 行）
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
print(ssh_addr)
print(vllm_python)
PYEOF
}

main() {
  local out head_ssh vllm_python
  out="$(extract_head)" || exit 1
  head_ssh="$(printf '%s\n' "$out" | sed -n '1p')"
  vllm_python="$(printf '%s\n' "$out" | sed -n '2p')"

  echo "== setup_remote.sh：目标 head 主机 $head_ssh（python=$vllm_python） =="
  if ssh "${SSH_OPTS[@]}" "$head_ssh" \
      "bash -s $(sh_quote "$REMOTE_DIR") $(sh_quote "$vllm_python")" <<'RMT'
# ── 远端脚本（在 head 主机上执行）──────────────────────────────
set -u
REMOTE_DIR="$1"; VLLM_PYTHON="$2"
case "$REMOTE_DIR" in
  '~/'*) REMOTE_DIR="$HOME${REMOTE_DIR#\~}" ;;
esac
PASS=0; FAIL=0
ok()  { printf '  [PASS] %s\n' "$1"; PASS=$((PASS+1)); }
bad() { printf '  [FAIL] %s\n' "$1"; FAIL=$((FAIL+1)); }

echo "== [1/4] CUDA 检测（决定 vLLM wheel） =="
CUDA_MAJ="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\.[0-9]*.*/\1/p' | head -1)"
if [ -n "$CUDA_MAJ" ]; then
  ok "nvidia-smi 可用；CUDA major=$CUDA_MAJ"
else
  bad "nvidia-smi 不可用或读不到 CUDA 版本（vLLM 需要 NVIDIA GPU + 驱动）"
fi
# 只认明确可判定的 CUDA 主版本：12 → +cu129 wheel；13 → 默认 wheel；
# 其余（11/10/空/未知）→ 记 FAIL、跳过安装，绝不猜测 wheel 导致假 PASS
case "$CUDA_MAJ" in
  12) WHEEL="vllm==0.31.0+cu129" ;;
  13) WHEEL="vllm==0.31.0" ;;
  *)  bad "无法确定/不支持的 CUDA 主版本（'$CUDA_MAJ'）；支持 12.x / 13.x"
      SKIP_INSTALL=1 ;;
esac
echo "    选用 wheel: ${WHEEL:-<跳过>}（docs/vllm-facts.md §7）"

echo "== [2/4] uv =="
if command -v uv >/dev/null 2>&1; then
  ok "uv 已存在（$(uv --version 2>/dev/null)）"
elif curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  ok "uv 已安装（$(uv --version 2>/dev/null)）"
else
  bad "uv 安装失败（head 主机需可访问网络）"
fi

echo "== [3/4] uv pip install ${WHEEL:-<未定>}（--python $VLLM_PYTHON） =="
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
  echo "    CUDA 主版本无法确定/不支持：跳过安装（修复驱动后重跑 setup_remote.sh）"
elif LOG="$(uv pip install --python "$VLLM_PYTHON" "$WHEEL" 2>&1)"; then
  ok "vllm $WHEEL 已就绪"
else
  printf '%s\n' "$LOG" >&2
  bad "uv pip install 失败"
fi

echo "== [4/4] 验证 + REMOTE_DIR =="
if "$VLLM_PYTHON" --version >/dev/null 2>&1; then
  ok "python: $("$VLLM_PYTHON" --version 2>&1)"
else
  bad "python（$VLLM_PYTHON）不可用"
fi
VLLM_VER="$("$VLLM_PYTHON" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
if [ -n "$VLLM_VER" ]; then
  ok "vllm --version: $VLLM_VER"
else
  bad "vllm import 失败（$VLLM_PYTHON -c 'import vllm'）"
fi
# vllm 控制台脚本可达性：command -v vllm 或 python 同目录下的 vllm，且 --version 能跑
VLLM_BIN="$(command -v vllm 2>/dev/null)"
if [ -z "$VLLM_BIN" ] && [ -x "$(dirname "$VLLM_PYTHON")/vllm" ]; then
  VLLM_BIN="$(dirname "$VLLM_PYTHON")/vllm"
fi
if [ -n "$VLLM_BIN" ] && "$VLLM_BIN" --version >/dev/null 2>&1; then
  ok "vllm CLI 可达（$VLLM_BIN --version）"
else
  bad "vllm CLI 不可达（command -v vllm 与 $VLLM_PYTHON 同目录的 vllm 均不可用）"
fi
if nvidia-smi -L >/dev/null 2>&1; then
  ok "nvidia-smi -L 可枚举 GPU"
else
  bad "nvidia-smi -L 失败"
fi
if mkdir -p "$REMOTE_DIR"; then
  ok "REMOTE_DIR 就绪: $REMOTE_DIR"
else
  bad "创建 $REMOTE_DIR 失败"
fi

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
