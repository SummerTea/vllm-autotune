#!/usr/bin/env bash
# run.sh — 两段式远程执行驱动器（启动 / 轮询 / 取结果），本机 shell 永不阻塞于 trial。
#   无参数  : 启动一轮 trial（rsync → 远端 nohup 后台跑 → 秒回）
#   --poll  : 单次状态检查；退出码 3=仍在运行 0=done 1=crash（无 status.json 也按 3）
#   --fetch : 打印本机最新 result.json（来自上次 done 的回传副本）
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

RSYNC_E="ssh -o BatchMode=yes -o ConnectTimeout=10"
if [ -n "$SSH_PORT" ]; then RSYNC_E+=" -p $SSH_PORT"; fi
if [ -n "$SSH_KEY" ]; then RSYNC_E+=" -i $SSH_KEY"; fi
RSYNC_OPTS=(-a -q --timeout=30 -e "$RSYNC_E")

# 解析 yaml 用的 python：优先仓库 venv（pyyaml 是已声明依赖），退化到 python3/python
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python; fi

# ── 工具函数 ──────────────────────────────────────────────────────────
# 嵌入远端 shell 命令 / rsync 远端路径：安全字符原样输出（~ 由远端展开），否则 %q 引用
path_for_remote() {
  case "$1" in
    *[!a-zA-Z0-9_./~-]*) printf '%q' "$1" ;;
    *) printf '%s' "$1" ;;
  esac
}
# 单引号包裹并转义内部单引号，用于嵌入远端 shell 命令串
sh_quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }

# 提取 head 主机 ssh 地址 / remote.vllm_python / config.yaml env（参数 1 = 是否需要 env）
extract_meta() {
  "$PY" - "$1" <<'PYEOF'
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
if sys.argv[1] == "1":
    for k, v in (load("config.yaml").get("env") or {}).items():
        print(f"{k}={v}")
PYEOF
}

# 远端是否有 trial 在跑（bench.pid 存在且进程存活）；kill -0 探活
is_trial_running() {  # $1 = head_ssh
  local head_ssh="$1" p out rc=0
  p="$(path_for_remote "$REMOTE_DIR")"
  out="$(ssh "${SSH_OPTS[@]}" "$head_ssh" \
    "if [ -f $p/bench.pid ] && kill -0 \"\$(cat \"$p/bench.pid\")\" 2>/dev/null; then echo ALIVE; fi" 2>/dev/null)" || rc=$?
  [ "$rc" -eq 0 ] && [ -n "$out" ]
}

# 把远端 $REMOTE_DIR 下某文件回传本机（文件不存在则跳过，失败仅警告）
rsync_back() {  # $1 = head_ssh  $2 = 远端文件名（相对 REMOTE_DIR）
  local head_ssh="$1" f="$2" p
  p="$(path_for_remote "$REMOTE_DIR")"
  if ssh "${SSH_OPTS[@]}" "$head_ssh" "test -f $p/$f" 2>/dev/null; then
    rsync "${RSYNC_OPTS[@]}" "$head_ssh:$p/$f" "./$f" 2>/dev/null \
      || echo "警告: rsync 回传 $f 失败" >&2
  fi
}

# ── 动作 ───────────────────────────────────────────────────────────────
cmd_start() {
  local f
  for f in config.yaml harness.yaml target.yaml bench.py; do
    if [ ! -f "$f" ]; then
      echo "缺少 $f（bench.py 由 Phase 3 Lane A 交付，请确认在仓库根目录）" >&2
      return 1
    fi
  done

  local out head_ssh vllm_python
  out="$(extract_meta 1)" || return 1
  head_ssh="$(printf '%s\n' "$out" | sed -n '1p')"
  vllm_python="$(printf '%s\n' "$out" | sed -n '2p')"
  local config_env=()
  while IFS= read -r line; do config_env+=("$line"); done < <(printf '%s\n' "$out" | sed -n '3,$p')

  if is_trial_running "$head_ssh"; then
    echo "远端已有 trial 在运行（bench.pid 存活）：请先运行 bash run.sh --poll 等其结束，勿重复启动。" >&2
    return 1
  fi

  local p src
  p="$(path_for_remote "$REMOTE_DIR")"
  src=(config.yaml harness.yaml target.yaml bench.py)
  if [ -f env.json ]; then src+=(env.json); fi
  if [ -f model_profile.json ]; then src+=(model_profile.json); fi
  rsync "${RSYNC_OPTS[@]}" "${src[@]}" "$head_ssh:$p/" \
    || { echo "rsync 同步到 $head_ssh 失败" >&2; return 1; }

  # 远端 nohup 启动（冻结契约：--config/--harness/--target/--out-dir 与 bench.pid）。
  # 附加 </dev/null 与 mkdir -p：防 ssh 因后台进程占用 stdin 而挂起（不改 bench.py 契约）。
  local env_str kv
  env_str=""
  for kv in "${config_env[@]}"; do env_str+=" $(sh_quote "$kv")"; done
  if ! ssh "${SSH_OPTS[@]}" -n "$head_ssh" \
      "mkdir -p $p && cd $p && nohup env$env_str $(path_for_remote "$vllm_python") bench.py --config config.yaml --harness harness.yaml --target target.yaml --out-dir . </dev/null > bench.stdout 2>&1 & echo \$! > bench.pid"; then
    echo "远端启动失败（请查 $head_ssh:$p/bench.stdout）" >&2
    return 1
  fi

  # 抓一次远端 PID / 状态（status.json 可能还没生成，属于正常）
  local pid st
  pid="$(ssh "${SSH_OPTS[@]}" "$head_ssh" "cat $p/bench.pid 2>/dev/null || true")" || true
  st="$(ssh "${SSH_OPTS[@]}" "$head_ssh" "cat $p/status.json 2>/dev/null || true")" || true
  echo "started; pid=${pid:-?}; poll with: bash run.sh --poll"
  if [ -n "$st" ]; then echo "initial status: $st"; fi
  return 0
}

cmd_poll() {
  local out head_ssh p
  out="$(extract_meta 0)" || return 1
  head_ssh="$(printf '%s\n' "$out" | sed -n '1p')"
  p="$(path_for_remote "$REMOTE_DIR")"

  local rc=0 raw
  raw="$(ssh "${SSH_OPTS[@]}" "$head_ssh" "cat $p/status.json 2>/dev/null || true")" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "无法读取远端状态（ssh 失败）：请检查网络与 $head_ssh" >&2
    return 3
  fi
  if [ -z "$raw" ]; then
    echo "starting（远端尚无 status.json）；继续: bash run.sh --poll"
    return 3
  fi

  local state phase message
  state="$(printf '%s' "$raw" | sed -n 's/.*"state"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
  phase="$(printf '%s' "$raw" | sed -n 's/.*"phase"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
  message="$(printf '%s' "$raw" | sed -n 's/.*"message"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"

  case "$state" in
    running)
      # 说明：poll 以 status.json 为准；若进程被外部杀死而 status.json 停留 running，需人工介入
      echo "running（phase=${phase:-?}${message:+, message=$message}）；仍运行中: bash run.sh --poll"
      return 3
      ;;
    done)
      rsync_back "$head_ssh" result.json
      rsync_back "$head_ssh" model_profile.json
      echo "=== remote summary ==="
      local blk
      blk="$(ssh "${SSH_OPTS[@]}" "$head_ssh" \
        "awk '/^status:/{start=last} /^---/{last=NR} start&&NR>=start{print}' $p/bench.stdout 2>/dev/null || true")" || true
      if [ -z "${blk:-}" ]; then
        blk="$(ssh "${SSH_OPTS[@]}" "$head_ssh" "sed -n '/^---/,\$p' $p/bench.stdout 2>/dev/null || true")" || true
      fi
      printf '%s\n' "$blk"
      echo "=== end summary ==="
      return 0
      ;;
    crash)
      rsync_back "$head_ssh" result.json
      echo "crash：${message:-详见远端 status.json / bench.stdout}"
      return 1
      ;;
    *)
      echo "未知 state=$state；原样输出远端 status.json："
      printf '%s\n' "$raw"
      return 3
      ;;
  esac
}

cmd_fetch() {
  if [ -f result.json ]; then
    cat result.json
    return 0
  fi
  echo "本机暂无 result.json（尚无 done 的 trial；先运行 bash run.sh --poll）" >&2
  return 1
}

usage() {
  cat <<'EOF'
用法: bash run.sh [--poll|--fetch|--help]
  无参数   启动一轮 trial（rsync → 远端 nohup 后台运行 → 秒回）
  --poll   单次状态检查；退出码: 0=done 1=crash 3=仍在运行
  --fetch  打印本机最新 result.json（来自上次 done 的回传副本）
EOF
}

case "${1:-}" in
  --poll) cmd_poll ;;
  --fetch) cmd_fetch ;;
  --help | -h) usage ;;
  "") cmd_start ;;
  *) usage >&2; exit 2 ;;
esac
