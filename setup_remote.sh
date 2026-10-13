#!/usr/bin/env bash
# setup_remote.sh — 主机环境校验（零破坏）：docker 校验镜像、native 默认只检查不安装。
# 幂等：可安全重复运行；仅在 target.yaml 确认的 work_dir 内写入。在编排机（本仓库目录）
# 执行，经 ssh 作用到 head 主机。用法: bash setup_remote.sh
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

# 提取 head 主机信息：ssh 地址（1）/ work_dir（2）/ remote.vllm_python（3）/
# runtime kind（4）/ docker 镜像（5）/ install_allowed（6）
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
print(runtime.get("kind") or "native")
print(runtime.get("image") or "")
print(str((runtime.get("install_allowed") or False)).lower())
PYEOF
}

main() {
  local out head_ssh work_dir vllm_python runtime_kind runtime_image install_allowed
  out="$(extract_head)" || exit 1
  head_ssh="$(printf '%s\n' "$out" | sed -n '1p')"
  work_dir="$(printf '%s\n' "$out" | sed -n '2p')"
  vllm_python="$(printf '%s\n' "$out" | sed -n '3p')"
  runtime_kind="$(printf '%s\n' "$out" | sed -n '4p')"
  runtime_image="$(printf '%s\n' "$out" | sed -n '5p')"
  install_allowed="$(printf '%s\n' "$out" | sed -n '6p')"

  if [ -z "$work_dir" ]; then
    echo "请在 target.yaml 的 head 主机上提供 work_dir（主机专属目录，需用户确认；模板仅写此目录）" >&2
    exit 1
  fi

  echo "== setup_remote.sh：目标 head 主机 $head_ssh（python=$vllm_python，runtime=$runtime_kind，work_dir=$work_dir） =="
  if ssh "${SSH_OPTS[@]}" "$head_ssh" \
      "bash -s $(sh_quote "$work_dir") $(sh_quote "$vllm_python") $(sh_quote "$runtime_kind") $(sh_quote "$runtime_image") $(sh_quote "$install_allowed")" <<'RMT'
# ── 远端脚本（在 head 主机上执行）──────────────────────────────
set -u
WORK_DIR="$1"; VLLM_PYTHON="$2"; RUNTIME_KIND="$3"; RUNTIME_IMAGE="$4"; INSTALL_ALLOWED="$5"
case "$WORK_DIR" in
  '~/'*) WORK_DIR="$HOME${WORK_DIR#\~}" ;;
esac
PASS=0; FAIL=0
ok()  { printf '  [PASS] %s\n' "$1"; PASS=$((PASS+1)); }
bad() { printf '  [FAIL] %s\n' "$1"; FAIL=$((FAIL+1)); }

echo "== 零破坏横幅 =="
echo "  本脚本对主机零破坏：仅在 $WORK_DIR 内写入；默认不 pull 镜像、不安装环境"

if [ "$RUNTIME_KIND" = "docker" ]; then
  # ── docker 运行时：零 pull，只用主机上已预先准备的镜像 ──
  echo "== [1/3] docker 检查 =="
  DOCKER_FAIL=0
  if docker --version >/dev/null 2>&1; then
    ok "docker 可用（$(docker --version)）"
  else
    bad "docker 不可用（docker --version 失败）"
    DOCKER_FAIL=1
  fi
  if [ "$DOCKER_FAIL" -eq 1 ] || [ -z "$RUNTIME_IMAGE" ]; then
    [ -z "$RUNTIME_IMAGE" ] && bad "docker 模式缺少 harness.yaml 的 runtime.image"
  else
    echo "== [2/3] 镜像存在性检查（零 pull） =="
    if docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
      ok "镜像已就绪: $RUNTIME_IMAGE"
    else
      bad "镜像 $RUNTIME_IMAGE 未在主机上预先准备；本模板默认不 pull（保持主机零新增），请人工准备后再运行"
      DOCKER_FAIL=1
    fi
    echo "== [3/3] 容器入口验证 =="
    if [ "$DOCKER_FAIL" -eq 0 ] && docker run --rm --gpus all --entrypoint vllm "$RUNTIME_IMAGE" --version >/dev/null 2>&1; then
      ok "docker run --gpus all --entrypoint vllm $RUNTIME_IMAGE --version"
    else
      bad "容器 entrypoint vllm 不可用（镜像不含 vllm 控制台脚本？）"
      DOCKER_FAIL=1
    fi
  fi
  if mkdir -p "$WORK_DIR"; then
    ok "WORK_DIR 就绪: $WORK_DIR"
  else
    bad "创建 $WORK_DIR 失败"
  fi
else
  # ── native 运行时：默认不安装（install_allowed=false）；允许时仅装进 work_dir 内 venv ──
  echo "== [1/4] CUDA/GPU 检测（只读） =="
  CUDA_MAJ="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\.[0-9]*.*/\1/p' | head -1)"
  if [ -n "$CUDA_MAJ" ]; then
    ok "nvidia-smi 可用；CUDA major=$CUDA_MAJ"
  else
    bad "nvidia-smi 不可用或读不到 CUDA 版本（vLLM 需要 NVIDIA GPU + 驱动）"
  fi
  # wheel 选择仅在允许安装时才有意义；12 → +cu129，13 → 默认，其余不猜
  if [ "$INSTALL_ALLOWED" = "true" ]; then
    case "$CUDA_MAJ" in
      12) WHEEL="vllm==0.31.0+cu129" ;;
      13) WHEEL="vllm==0.31.0" ;;
      *)  WHEEL="" ; bad "无法确定/不支持的 CUDA 主版本（'$CUDA_MAJ'）；支持 12.x / 13.x" ;;
    esac
    [ -n "${WHEEL:-}" ] && echo "    选用 wheel: $WHEEL（docs/vllm-facts.md §7）"
  else
    echo "    install_allowed=false：不安装；仅检查主机已有环境"
  fi

  echo "== [2/4] 环境安装（零破坏：默认不装；仅限 $WORK_DIR 内） =="
  VENV=""
  if [ "$INSTALL_ALLOWED" = "true" ]; then
    VENV="$WORK_DIR/.venv"
    if [ -n "${WHEEL:-}" ] && "$VLLM_PYTHON" -m venv "$VENV" >/dev/null 2>&1; then
      VLLM_PY="$VENV/bin/python"
      if command -v uv >/dev/null 2>&1; then
        if LOG="$(uv pip install --python "$VLLM_PY" "$WHEEL" 2>&1)"; then
          INSTALL_OK=1
        else
          printf '%s\n' "$LOG" >&2; INSTALL_OK=0
        fi
      elif LOG="$("$VLLM_PY" -m pip install "$WHEEL" 2>&1)"; then
        INSTALL_OK=1
      else
        printf '%s\n' "$LOG" >&2; INSTALL_OK=0
      fi
      if [ "${INSTALL_OK:-0}" = "1" ]; then
        ok "vllm $WHEEL 已装入 $VENV（仅 work_dir 内；建议 harness.yaml 的 remote.vllm_python=$VLLM_PY）"
        VLLM_PYTHON="$VLLM_PY"
      else
        bad "安装 vllm 到 $VENV 失败"
      fi
    else
      bad "创建 $VENV 失败（$VLLM_PYTHON 无 venv 模块或 wheel 未定）"
    fi
  else
    echo "    默认不安装任何环境（主机零新增）"
  fi

  echo "== [3/4] 验证 vllm 可用性 =="
  VLLM_VER="$("$VLLM_PYTHON" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  if [ -n "$VLLM_VER" ]; then
    ok "vllm import: $VLLM_VER"
  else
    if [ "$INSTALL_ALLOWED" = "true" ] && [ -n "$VENV" ] && [ -x "$VENV/bin/python" ]; then
      bad "venv 内 vllm import 失败（$VLLM_PYTHON -c 'import vllm'）"
    else
      bad "native 模式未允许安装（runtime.install_allowed=false）且主机无现成 vllm；" \
          "请人工准备环境，或将 install_allowed 设为 true（安装仅限 $WORK_DIR 内）"
    fi
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
  echo "== [4/4] WORK_DIR =="
  if mkdir -p "$WORK_DIR"; then
    ok "WORK_DIR 就绪: $WORK_DIR"
  else
    bad "创建 $WORK_DIR 失败"
  fi
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
