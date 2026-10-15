#!/usr/bin/env bash
# run.sh — 两段式远程执行驱动器（启动 / 轮询 / 取结果），本机 shell 永不阻塞于 trial。
#   无参数  : 启动一轮 trial（rsync → 远端 nohup 后台跑 → 秒回）
#   --detect-gmu : 启动 gmu 预检（同 start；远端 bench.py --detect-gmu，结果回传 gmu_probe.json）
#   --poll  : 单次状态检查；退出码 3=仍在运行 0=done 1=crash（无 status.json 也按 3）
#   --fetch : 打印本机最新 result.json（来自上次 done 的回传副本）
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ── .env：手工解析而非 source，保留字面值（work_dir 的 ~ 留给远端 shell 展开）──
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
            # SSH_KEY 是本地路径：以 ~/ 开头时在本机展开（WORK_DIR 不做此处理）
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
# REMOTE_DIR 不再有默认值：主来源是 target.yaml 的 work_dir；
# .env 的 REMOTE_DIR 仅为向后兼容的覆盖项（已弃用，见 resolve_remote_dir）
REMOTE_DIR_ENV="${REMOTE_DIR:-}"
REMOTE_DIR=""

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

# 提取 head 主机 ssh 地址（第 1 行）/ work_dir（第 2 行）/ remote.vllm_python（第 3 行）
# / config.yaml env（参数 1 = 1 时，第 4 行起）
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
print(heads[0].get("work_dir") or "")
print(vllm_python)
if sys.argv[1] == "1":
    for k, v in (load("config.yaml").get("env") or {}).items():
        print(f"{k}={v}")
PYEOF
}

# 远端专属目录：主来源 = target.yaml 的 work_dir（模板唯一允许写入的位置，需用户确认）；
# .env REMOTE_DIR 仅作已弃用的覆盖项
resolve_remote_dir() {  # $1 = work_dir
  local work_dir="$1"
  if [ -z "$work_dir" ]; then
    echo "请在 target.yaml 的 head 主机上提供 work_dir（主机专属目录，需用户确认；模板仅写此目录）" >&2
    return 1
  fi
  if [ -n "$REMOTE_DIR_ENV" ] && [ "$REMOTE_DIR_ENV" != "$work_dir" ]; then
    echo "警告: .env 的 REMOTE_DIR（${REMOTE_DIR_ENV}）覆盖 target.yaml 的 work_dir（${work_dir}）；" \
         "REMOTE_DIR 已弃用，建议改 target.yaml" >&2
    printf '%s' "$REMOTE_DIR_ENV"
  else
    printf '%s' "$work_dir"
  fi
}

# 探活远端 trial 三态：echo ALIVE(存活) / DEAD(进程不存在) / UNKNOWN(ssh 失败)
trial_alive_state() {  # $1 = head_ssh
  local head_ssh="$1" p out rc=0
  p="$(path_for_remote "$REMOTE_DIR")"
  out="$(ssh "${SSH_OPTS[@]}" "$head_ssh" \
    "if [ -f $p/bench.pid ] && kill -0 \"\$(cat $p/bench.pid)\" 2>/dev/null; then echo ALIVE; else echo DEAD; fi" 2>/dev/null)" || rc=$?
  if [ "$rc" -ne 0 ]; then echo UNKNOWN; return; fi
  case "$out" in ALIVE) echo ALIVE ;; DEAD) echo DEAD ;; *) echo UNKNOWN ;; esac
}

# 远端 trial 是否存活（仅 ALIVE 为真；用于启动前的重复启动守卫）
is_trial_running() { [ "$(trial_alive_state "$1")" = ALIVE ]; }

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
cmd_start() {  # $1 = 可选 "--detect-gmu"：启动 gmu 预检而非正式 trial
  local mode="${1:-}" bench_extra=""
  if [ "$mode" = "--detect-gmu" ]; then bench_extra=" --detect-gmu"; fi
  local f
  for f in config.yaml harness.yaml target.yaml bench.py; do
    if [ ! -f "$f" ]; then
      echo "缺少 ${f}（请确认在仓库根目录）" >&2
      return 1
    fi
  done

  local out head_ssh work_dir vllm_python
  out="$(extract_meta 1)" || return 1
  head_ssh="$(printf '%s\n' "$out" | sed -n '1p')"
  work_dir="$(printf '%s\n' "$out" | sed -n '2p')"
  vllm_python="$(printf '%s\n' "$out" | sed -n '3p')"
  REMOTE_DIR="$(resolve_remote_dir "$work_dir")" || return 1
  local config_env=()
  while IFS= read -r line; do config_env+=("$line"); done < <(printf '%s\n' "$out" | sed -n '4,$p')

  if is_trial_running "$head_ssh"; then
    echo "远端已有 trial 在运行（bench.pid 存活）：请先运行 bash run.sh --poll 等其结束，勿重复启动。" >&2
    return 1
  fi

  local p
  p="$(path_for_remote "$REMOTE_DIR")"

  # 本机 trial 身份：日期-时间-本地 shell PID（持久化到 .trial_id；poll 侧据此识别陈旧远端产物）
  LOCAL_TRIAL_ID="$(date +%Y%m%dT%H%M%S)-$$"
  printf '%s\n' "$LOCAL_TRIAL_ID" > .trial_id
  # 已确认无存活 trial：清掉上一个 trial 的远端状态产物（只删 4 个文件，不碰 run.log / 模型；
  # gmu_probe.json 一并清，防陈旧探测结果被 rsync 回传误作本次的）
  ssh "${SSH_OPTS[@]}" "$head_ssh" "rm -f $p/status.json $p/result.json $p/bench.pid $p/gmu_probe.json" 2>/dev/null || true
  # 探测模式：先清本机旧 gmu_probe.json，失败的探测不会留下上一次的 gmu_suggested（普通 trial 不清）
  if [ "$mode" = "--detect-gmu" ]; then rm -f ./gmu_probe.json; fi

  local src
  src=(config.yaml harness.yaml target.yaml bench.py)
  if [ -f env.json ]; then src+=(env.json); fi
  if [ -f model_profile.json ]; then src+=(model_profile.json); fi
  rsync "${RSYNC_OPTS[@]}" "${src[@]}" "$head_ssh:$p/" \
    || { echo "rsync 同步到 $head_ssh 失败" >&2; return 1; }

  # 远端 nohup 启动（冻结契约：--config/--harness/--target/--out-dir 与 bench.pid）。
  local env_str kv
  env_str=""
  # bash 3.2 (macOS 自带) 下 set -u + 空数组展开会报 unbound；仅当非空时展开
  if [ "${#config_env[@]}" -gt 0 ]; then
    for kv in "${config_env[@]}"; do env_str+=" $(sh_quote "$kv")"; done
  fi
  # 远端启动：必须是「简单命令 + 全重定向后台」，ssh 才会立即返回——
  # 远端 shell 会等待任何仍持有会话 fd 的后台子 shell（实测：`{ }`/`&&` 列表式后台会挂到
  # trial 结束；`nohup <cmd> >f 2>&1 </dev/null &` 才秒回）。解释器：harness.remote.vllm_python
  # 仅作首选，缺失回退 python3/python（主机常只有 python3）。
  if ! ssh "${SSH_OPTS[@]}" -n "$head_ssh" \
      "mkdir -p $p; nohup env$env_str sh -c 'cd $p && PY=\$(command -v $(path_for_remote "$vllm_python") 2>/dev/null || command -v python3 2>/dev/null || command -v python 2>/dev/null) && exec \"\$PY\" bench.py --config config.yaml --harness harness.yaml --target target.yaml --out-dir . --trial-id $LOCAL_TRIAL_ID$bench_extra' > $p/bench.stdout 2>&1 < /dev/null & echo \$! > $p/bench.pid"; then
    echo "远端启动失败（请查 $head_ssh:$p/bench.stdout）" >&2
    return 1
  fi

  # 抓一次远端 PID / 状态（status.json 可能还没生成，属于正常）
  local pid st started_tag
  pid="$(ssh "${SSH_OPTS[@]}" "$head_ssh" "cat $p/bench.pid 2>/dev/null || true")" || true
  st="$(ssh "${SSH_OPTS[@]}" "$head_ssh" "cat $p/status.json 2>/dev/null || true")" || true
  if [ "$mode" = "--detect-gmu" ]; then started_tag=" (detect-gmu)"; else started_tag=""; fi
  echo "started${started_tag}; pid=${pid:-?}; poll with: bash run.sh --poll"
  if [ -n "$st" ]; then echo "initial status: $st"; fi
  return 0
}

cmd_poll() {
  local out head_ssh work_dir p
  out="$(extract_meta 0)" || return 1
  head_ssh="$(printf '%s\n' "$out" | sed -n '1p')"
  work_dir="$(printf '%s\n' "$out" | sed -n '2p')"
  REMOTE_DIR="$(resolve_remote_dir "$work_dir")" || return 1
  p="$(path_for_remote "$REMOTE_DIR")"

  local rc=0 raw
  raw="$(ssh "${SSH_OPTS[@]}" "$head_ssh" "cat $p/status.json 2>/dev/null || true")" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "无法读取远端状态（ssh 失败）：请检查网络与 $head_ssh" >&2
    return 3
  fi
  if [ -z "$raw" ]; then
    # 尚无 status.json：可能在启动中，也可能已死在写出首个状态之前（如 vllm CLI 不可达 / 配置错误）。
    # 探活区分，避免该失败路径在 poll 层变成永久悬挂。
    case "$(trial_alive_state "$head_ssh")" in
      DEAD)
        rsync_back "$head_ssh" result.json
        echo "远端进程已消失且未产出 status.json（按 crash 处理：常见于 vllm CLI 不可达 / 配置错误，见远端 bench.stdout）"
        return 1 ;;
      ALIVE)
        echo "starting（远端尚无 status.json）；继续: bash run.sh --poll"
        return 3 ;;
      *)
        echo "starting（状态未知，ssh 抖动）；稍后重试: bash run.sh --poll"
        return 3 ;;
    esac
  fi

  # status.json 单字段解析小工具（首次读取与 TOCTOU 重读共用，避免复制粘贴 sed）
  status_field() {  # $1 = raw json 文本  $2 = 字段名
    printf '%s' "$1" | sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -1
  }
  # done 收尾（正常 done 分支与 running→DEAD 重读后共用）：回传产物 + 远端 summary
  finish_done() {
    rsync_back "$head_ssh" result.json
    rsync_back "$head_ssh" model_profile.json
    rsync_back "$head_ssh" gmu_probe.json
    echo "=== remote summary ==="
    local gmu_line blk
    gmu_line="$(ssh "${SSH_OPTS[@]}" "$head_ssh" \
      "grep '^gmu_suggested:' $p/bench.stdout 2>/dev/null || true")" || true
    if [ -n "$gmu_line" ]; then
      printf '%s\n' "$gmu_line"    # 探测模式：结果已回传本机 gmu_probe.json
    else
      blk="$(ssh "${SSH_OPTS[@]}" "$head_ssh" \
        "awk '/^status:/{start=last} /^---/{last=NR} start&&NR>=start{print}' $p/bench.stdout 2>/dev/null || true")" || true
      if [ -z "${blk:-}" ]; then
        blk="$(ssh "${SSH_OPTS[@]}" "$head_ssh" "sed -n '/^---/,\$p' $p/bench.stdout 2>/dev/null || true")" || true
      fi
      printf '%s\n' "$blk"
    fi
    echo "=== end summary ==="
    return 0
  }
  # crash 收尾（正常 crash 分支与 running→DEAD 重读后共用）
  finish_crash() {  # $1 = crash 消息（空则用通用提示）
    rsync_back "$head_ssh" result.json
    rsync_back "$head_ssh" gmu_probe.json
    echo "crash：${1:-详见远端 status.json / bench.stdout}"
    return 1
  }
  # 陈旧产物防护（done/crash/running 一律先过此关）：status.json 带 trial_id 时，与本机
  # .trial_id 期望不符 → 属于别的 trial（错配/残留），不消费结果，按 crash 返回 1 让循环
  # 退出而非悬挂。.trial_id 缺失（如断点续跑会话）→ 完全按旧行为（向后兼容）。
  local expected_trial_id=""
  if [ -f .trial_id ]; then
    expected_trial_id="$(cat .trial_id)"
  fi
  guard_trial_mismatch() {  # $1 = raw json 文本；命中错配 → return 1
    local rid
    rid="$(status_field "$1" trial_id)"
    if [ -n "$rid" ] && [ -n "$expected_trial_id" ] && [ "$rid" != "$expected_trial_id" ]; then
      echo "警告：远端 status.json 属于另一个 trial（remote trial_id=${rid}，本机期望=${expected_trial_id}）：" \
           "疑似陈旧/错配产物，本次结果不消费。请确认无并发启动后重试。"
      return 1
    fi
    return 0
  }

  local state phase message
  state="$(status_field "$raw" state)"
  phase="$(status_field "$raw" phase)"
  message="$(status_field "$raw" message)"
  guard_trial_mismatch "$raw" || return 1

  case "$state" in
    running)
      # 以 status.json 为主，但先探活远端进程：DEAD（OOM/断电/被 kill）→ 按 crash 收尾，
      # 避免 --poll 永久返回 3；UNKNOWN（ssh 抖动）→ 视为仍运行，避免误杀健康 trial。
      case "$(trial_alive_state "$head_ssh")" in
        DEAD)
          # TOCTOU 窗口：读到 running 后进程恰好收尾退出——bench.py 在写出终态 status 之前的
          # 清理（如 _wait_mem_freed）可耗时数十秒，探活会先看到 DEAD。重读一次 status.json：
          # 已转 done/crash → 按终态正常收尾；重读 ssh 抖动 → 暂按运行中（return 3，勿误判 crash）；
          # 仍 running / 不可读 → 按 crash（同旧行为）。
          local rr=0
          raw="$(ssh "${SSH_OPTS[@]}" "$head_ssh" "cat $p/status.json 2>/dev/null || true")" || rr=$?
          if [ "$rr" -ne 0 ]; then
            echo "running（重读远端状态 ssh 抖动，暂按运行中）：bash run.sh --poll"
            return 3
          fi
          state="$(status_field "$raw" state)"
          phase="$(status_field "$raw" phase)"
          message="$(status_field "$raw" message)"
          guard_trial_mismatch "$raw" || return 1
          case "$state" in
            done) finish_done ;;
            crash) finish_crash "$message" ;;
            *)
              rsync_back "$head_ssh" result.json
              echo "远端进程已消失（按 crash 处理）：${message:-详见远端 status.json / bench.stdout}"
              return 1 ;;
          esac
          ;;
        ALIVE)
          echo "running（phase=${phase:-?}${message:+, message=$message}）；仍运行中: bash run.sh --poll"
          return 3 ;;
        *)
          echo "running（探活 ssh 抖动，暂按运行中）：bash run.sh --poll"
          return 3 ;;
      esac
      ;;
    done)
      finish_done
      ;;
    crash)
      finish_crash "$message"
      ;;
    *)
      echo "未知 state=${state}；原样输出远端 status.json："
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

# 只读占用探测（供 program.md「环境失效与共租处置」的有限重探）：打印远端 GPU 利用率/显存与
# 外部 compute 进程；退出码 0=空闲、1=有外部占用（或 ssh 失败）。不启 trial、不写远端。
cmd_check_gpus() {
  local out head_ssh raw apps
  out="$(extract_meta 0)" || return 1
  head_ssh="$(printf '%s\n' "$out" | sed -n '1p')"
  raw="$(ssh "${SSH_OPTS[@]}" -n "$head_ssh" \
    'echo "--- GPU (index,memory.used,util.gpu) ---"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(nvidia-smi 不可用)"; echo "--- COMPUTE_APPS (pid,proc,mem) ---"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true')" \
    || { echo "无法读取远端 GPU 状态（ssh 失败）：$head_ssh" >&2; return 1; }
  printf '%s\n' "$raw"
  apps="$(printf '%s\n' "$raw" | sed -n '/--- COMPUTE_APPS/,$p' | tail -n +2 | tr -d '[:space:]')"
  if [ -n "$apps" ]; then
    echo "判定：非空闲——检测到外部 GPU 进程（共租会污染吞吐/时延测量）" >&2
    return 1
  fi
  echo "判定：空闲——无外部 GPU 进程"
  return 0
}

usage() {
  cat <<'EOF'
用法: bash run.sh [--poll|--fetch|--detect-gmu|--check-gpus|--help]
  无参数   启动一轮 trial（rsync → 远端 nohup 后台运行 → 秒回）
  --detect-gmu  启动 gmu 预检（同 start：远端 bench.py --detect-gmu，结果回传 gmu_probe.json）
  --poll   单次状态检查；退出码: 0=done 1=crash 3=仍在运行
  --fetch  打印本机最新 result.json（来自上次 done 的回传副本）
  --check-gpus  只读探测 head 主机 GPU 占用（外部 compute 进程）；退出码 0=空闲 1=有外部占用
EOF
}

case "${1:-}" in
  --detect-gmu) cmd_start --detect-gmu ;;
  --poll) cmd_poll ;;
  --fetch) cmd_fetch ;;
  --check-gpus) cmd_check_gpus ;;
  --help | -h) usage ;;
  "") cmd_start ;;
  *) usage >&2; exit 2 ;;
esac
