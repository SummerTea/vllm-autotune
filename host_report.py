"""host_report.py — 主机探索报告（HTML）渲染。

纯函数：render_host_report(target, facts, harness, model_profile) -> str
无文件 IO、无网络、无全局状态、不依赖 probe.py；返回自包含可离线打开的 HTML。

数据形状以 env.json 为准；host-only 时大量字段为 None / 空列表，一律优雅降级为「—」。
"""

from __future__ import annotations

import html
import re

_DASH = "—"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------- 基础格式化
def _e(v) -> str:
    """转义后的展示文本；None/空串 → —。"""
    if v is None:
        return _DASH
    s = str(v).strip()
    return html.escape(s) if s else _DASH


def _gi(n) -> str:
    """字节 → GiB（1 位小数）。"""
    try:
        return f"{float(n) / (1024 ** 3):.1f}"
    except (TypeError, ValueError):
        return _DASH


def _mib_to_gib(n) -> str:
    try:
        return f"{float(n) / 1024:.1f}"
    except (TypeError, ValueError):
        return _DASH


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _status_class(s) -> str:
    return {"pass": "st-pass", "warn": "st-warn", "fail": "st-fail"}.get(str(s or ""), "st-info")


def _badge(s) -> str:
    label = _e(s)
    return f'<span class="badge {_status_class(s)}">{label}</span>'


def _table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
    )
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _fmt_ports(v) -> str:
    """容器的 ports 字段：str / list / dict 都能显示。"""
    if v is None:
        return _DASH
    if isinstance(v, dict):
        return _e(", ".join(f"{k}→{x}" for k, x in v.items()))
    if isinstance(v, list):
        return _e(", ".join(str(x) for x in v))
    return _e(v)


def _fmt_any(v) -> str:
    if isinstance(v, (list, tuple)):
        return _e(", ".join(str(x) for x in v)) if v else _DASH
    return _e(v)


# ---------------------------------------------------------------- CSS（内联、无外链）
_CSS = """
:root{
  --bg:#f2f4f7; --card:#ffffff; --ink:#1c2530; --muted:#5b6b7c;
  --line:#d8dee6; --head:#eef1f5;
  --pass:#1a7f37; --passbg:#e6f4ea;
  --warn:#9a6700; --warnbg:#fdf3d6;
  --fail:#c62828; --failbg:#fbe9e7;
  --accent:#2f5d8a;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; padding:32px 20px 64px;
  background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC",
    "PingFang SC","Microsoft YaHei",sans-serif;
  font-size:14px; line-height:1.6;
}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px;letter-spacing:.3px}
.sub{color:var(--muted);font-size:13px;margin:0 0 24px}
h2{
  font-size:16px;margin:34px 0 12px;padding-bottom:6px;
  border-bottom:1px solid var(--line);color:var(--accent);
}
h3{font-size:14px;margin:18px 0 8px;color:var(--muted);font-weight:600}
section.host{margin-top:18px}
.hosttitle{
  display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;
  margin:8px 0 18px;
}
.hosttitle .nm{font-size:19px;font-weight:700}
.hosttitle .meta{color:var(--muted);font-size:13px}
.cards{display:flex;flex-wrap:wrap;gap:12px}
.card{
  flex:1 1 220px;min-width:180px;max-width:100%;
  background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:12px 14px;
}
.card .k{font-size:12px;color:var(--muted);text-transform:none;letter-spacing:.3px}
.card .v{font-size:15px;font-weight:600;margin-top:2px;word-break:break-word}
.card .s{font-size:12px;color:var(--muted);margin-top:4px;word-break:break-word}
.card.wide{flex-basis:100%}
table{
  border-collapse:collapse;width:100%;background:var(--card);
  border:1px solid var(--line);border-radius:8px;overflow:hidden;
  margin:6px 0 4px;font-variant-numeric:tabular-nums;
}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--head);font-size:12px;color:var(--muted);font-weight:600}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:#fafbfc}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.st-pass{background:var(--passbg);color:var(--pass)}
.st-warn{background:var(--warnbg);color:var(--warn)}
.st-fail{background:var(--failbg);color:var(--fail)}
.st-info{background:#eaeef2;color:var(--muted)}
.ok{color:var(--pass);font-weight:600}
.bad{color:var(--fail);font-weight:600}
.muted{color:var(--muted)}
.alert{
  background:var(--failbg);border:1px solid var(--fail);border-left-width:5px;
  color:var(--fail);border-radius:8px;padding:10px 14px;margin:12px 0;font-weight:600;
}
.freeok{
  color:var(--pass);font-weight:600;background:var(--passbg);
  border:1px solid #b7e0c2;border-radius:8px;padding:8px 14px;margin:12px 0;display:inline-block;
}
ul.list{margin:6px 0;padding-left:20px}
ul.list li{margin:2px 0}
pre.raw{
  background:#0f1720;color:#d6e2ee;border-radius:8px;padding:14px;
  overflow:auto;font-size:12px;line-height:1.45;max-height:420px;
}
details{margin:10px 0}
details>summary{cursor:pointer;color:var(--accent);font-size:13px;padding:4px 0}
.concl{
  background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:16px 18px;line-height:1.85;
}
.note{color:var(--muted);font-size:12px;margin-top:6px}
@media (max-width:640px){
  body{padding:20px 12px 48px}
  .card{flex-basis:100%}
  table{display:block;overflow-x:auto;white-space:nowrap}
}
"""


# ---------------------------------------------------------------- 各段渲染
def _summary_cards(host: dict, hinfo: dict) -> str:
    gpu = host.get("gpu") or {}
    gpus = _as_list(gpu.get("gpus"))
    count = gpu.get("count")
    if count is None:
        count = len(gpus)
    gname = gpus[0].get("name") if gpus else None
    per_mib = gpus[0].get("memory_total") if gpus else None
    total_gib = None
    try:
        total_gib = f"{count * float(per_mib) / 1024:.0f}"
    except (TypeError, ValueError):
        total_gib = _DASH
    gpu_v = f"{_e(count)}× {_e(gname)}" if (count or gname) else _DASH
    gpu_s = f"单卡 {_mib_to_gib(per_mib)} GiB · 合计 {total_gib} GiB"

    topo = ((gpu.get("topo") or {}).get("summary"))
    topo_map = {
        "nvlink-full": "NVLink 全互联",
        "pcie": "仅 PCIe（无 NVLink）",
        "mixed": "混合",
    }
    topo_txt = topo_map.get(topo, _DASH)
    icb = host.get("interconnect") or {}
    rdma_bw = (icb.get("rdma_bw_gbps") or {}).get("bw_gbps")
    p2p = icb.get("gpu_p2p_bw_gbps") or {}
    icb_s = f"RDMA 实测 {_e(rdma_bw) if rdma_bw is not None else _DASH} Gb/s · " \
            f"P2P 组内 {_e(p2p.get('intra_numa'))} / 跨 {_e(p2p.get('cross_numa'))} Gb/s"

    cpu = host.get("cpu") or {}
    cpu_v = _e(cpu.get("model_name"))
    cpu_s = f"{_e(cpu.get('cores'))} 核 · {_e(cpu.get('numa_nodes'))} NUMA"

    disk = host.get("disk") or {}
    mounts = disk.get("mounts")
    if isinstance(mounts, list) and mounts:
        parts = []
        for m in mounts:
            if isinstance(m, dict):
                p = m.get("mount") or m.get("path")
                parts.append(f"{_e(p)} {_gi(m.get('avail_bytes'))}")
        disk_txt = "；".join(parts) if parts else _DASH
    else:
        av = disk.get("avail_bytes")
        disk_txt = f"{_gi(av)} GiB" if av is not None else _DASH

    # 占用判定
    dk = host.get("docker") or {}
    ncont = len(_as_list(dk.get("containers")))
    nport = len(_as_list((host.get("ports") or {}).get("listening")))
    nuser = len(_as_list((host.get("users") or {}).get("logged_in")))
    if ncont or nport or nuser:
        busy = f'<span class="bad">有人占用</span>（容器 {ncont} · 端口 {nport} · 用户 {nuser}）'
    else:
        busy = '<span class="ok">空闲</span>（无容器/监听端口/登录用户）'

    cards = [
        ("主机", _e(host.get("hostname")), f"ssh {_e(hinfo.get('ssh'))} · role {_e(hinfo.get('role') or host.get('role'))}"),
        ("GPU", gpu_v, gpu_s),
        ("互联", topo_txt, icb_s),
        ("CPU", cpu_v, cpu_s),
        ("磁盘余量", disk_txt, f"权重路径 {_e(disk.get('model_path'))}"),
        ("当前占用", busy, "由容器 / 监听端口 / 登录用户综合判断"),
    ]
    out = []
    for k, v, s in cards:
        out.append(f'<div class="card"><div class="k">{html.escape(k)}</div>'
                   f'<div class="v">{v}</div><div class="s">{s}</div></div>')
    return f'<div class="cards">{"".join(out)}</div>'


def _section_gpu(host: dict) -> str:
    gpu = host.get("gpu") or {}
    gpus = _as_list(gpu.get("gpus"))
    count = gpu.get("count")
    if count is None:
        count = len(gpus)

    rows = []
    for g in gpus:
        if not isinstance(g, dict):
            continue
        rows.append([
            _e(g.get("index")),
            _e(g.get("name")),
            _mib_to_gib(g.get("memory_total")),
            _e(g.get("driver_version")),
            _e(g.get("persistence_mode")),
            f'{_e(g.get("clocks_sm"))} MHz',
            f'{_e(g.get("clocks_max_sm"))} MHz',
            _e(g.get("numa_node")),
        ])
    gpu_tbl = _table(
        ["#", "型号", "显存 GiB", "驱动版本", "持久模式", "SM 时钟", "最大 SM 时钟", "NUMA"],
        rows, "gpu",
    ) or '<p class="muted">未检测到 GPU。</p>'

    # tp 建议
    topo = ((gpu.get("topo") or {}).get("summary"))
    groups = _as_list(host.get("numa_groups"))
    tp_line = ""
    if topo == "nvlink-full":
        tp_line = f'<p><b>可用 tp 范围：</b>NVLink 全互联，tp 可到 {_e(count)}。</p>'
    elif groups:
        try:
            mx = max(len(_as_list(g.get("gpus"))) for g in groups if isinstance(g, dict))
            tp_line = (f'<p><b>可用 tp 范围：</b>建议 tp ≤ {_e(mx)}'
                       f'（限同一 NUMA 组，依据 numa_groups）。</p>')
        except ValueError:
            tp_line = ""

    # numa 分组
    if groups:
        grp = "；".join(
            f'NUMA {_e(g.get("numa_node"))}: GPU {_fmt_any(g.get("gpus"))}'
            for g in groups if isinstance(g, dict)
        )
        grp_html = f'<h3>NUMA 分组</h3><p>{grp}</p>'
    else:
        grp_html = ""

    # 拓扑
    topo_raw = (gpu.get("topo") or {}).get("raw")
    topo_html = ""
    if topo_raw:
        clean = _ANSI.sub("", str(topo_raw))
        topo_html = (
            f'<h3>拓扑摘要：{_e(topo)}</h3>'
            f'<details><summary>nvidia-smi topo -m 原文</summary>'
            f'<pre class="raw">{html.escape(clean)}</pre></details>'
        )

    cpu = host.get("cpu") or {}
    mem = host.get("mem") or {}
    cpu_mem = (
        f'<h3>CPU / 内存</h3>'
        f'<p>{_e(cpu.get("model_name"))} · {_e(cpu.get("cores"))} 核 · '
        f'{_e(cpu.get("numa_nodes"))} NUMA · governor {_e(cpu.get("governor"))}；'
        f'THP {_e(mem.get("thp"))} · swappiness {_e(mem.get("swappiness"))}</p>'
    )
    return (
        '<h2>1. GPU 环境</h2>'
        + gpu_tbl + tp_line + grp_html + topo_html + cpu_mem
    )


def _section_net(host: dict) -> str:
    net = host.get("net") or {}
    ifaces = _as_list(net.get("interfaces"))
    rows = [[_e(i.get("name")), _e(i.get("mtu"))] for i in ifaces if isinstance(i, dict)]
    iface_tbl = _table(["接口", "MTU"], rows) or '<p class="muted">未采集到接口。</p>'

    rdma = net.get("rdma") or {}
    kind = rdma.get("kind")
    kind_txt = {"roce": "RoCE（Ethernet）", "infiniband": "InfiniBand"}.get(kind, _e(kind))
    devs = _as_list(rdma.get("devices"))
    drow = [
        [_e(d.get("name")), _e(d.get("state")), _e(d.get("netdev")),
         _e(d.get("speed_gbps")), _e(d.get("numa_node"))]
        for d in devs if isinstance(d, dict)
    ]
    dev_tbl = _table(["设备", "状态", "netdev", "速率 Gb/s", "NUMA"], drow) \
        or '<p class="muted">无 RDMA 设备。</p>'

    icb = host.get("interconnect") or {}
    rdma_bw = (icb.get("rdma_bw_gbps") or {}).get("bw_gbps")
    p2p = icb.get("gpu_p2p_bw_gbps") or {}
    bw_html = (
        f'<h3>实测带宽</h3><p>RDMA {_e(rdma_bw) if rdma_bw is not None else _DASH} Gb/s · '
        f'GPU P2P 组内 {_e(p2p.get("intra_numa"))} / 跨 NUMA {_e(p2p.get("cross_numa"))} Gb/s</p>'
        + (f'<p class="note">{_e(p2p.get("note"))}</p>' if p2p.get("note") else "")
    )
    return (
        '<h2>2. 网络环境</h2>'
        f'<h3>接口</h3>{iface_tbl}'
        f'<h3>RDMA：{kind_txt}</h3>{dev_tbl or "<p class=chr(34)mutedchr(34)>无 RDMA 设备。</p>"}'
        + bw_html
    )


def _section_docker(host: dict) -> str:
    dk = host.get("docker") or {}
    ver = dk.get("version")
    imgs = _as_list(dk.get("images"))
    if imgs:
        rows = []
        for im in imgs:
            if isinstance(im, dict):
                rows.append([_e(im.get("repo") or im.get("name")),
                             _e(im.get("size")), _e(im.get("created"))])
            else:
                rows.append([_e(im), _DASH, _DASH])
        tbl = _table(["镜像仓库", "大小", "创建时间"], rows)
    else:
        names = _as_list(dk.get("vllm_images"))
        if names:
            lis = "".join(f"<li>{_e(n)}</li>" for n in names)
            tbl = f'<p class="muted">未采集镜像明细（repo/size/created），仅列出 vLLM 相关镜像名：</p><ul class="list">{lis}</ul>'
        else:
            tbl = '<p class="muted">未发现 vLLM 推理镜像。</p>'
    head = f'<p class="muted">Docker：{_e(ver)}</p>' if ver else ""
    return f'<h2>3. 可用 vLLM 推理镜像</h2>{head}{tbl}'


def _section_models(host: dict) -> str:
    inv = _as_list(host.get("model_inventory"))
    if not inv:
        return '<h2>4. 模型介质</h2><p class="muted">未发现模型（或未配置扫描目录）。</p>'

    gpu = host.get("gpu") or {}
    gpus = _as_list(gpu.get("gpus"))
    count = gpu.get("count")
    if count is None:
        count = len(gpus)
    per_mib = gpus[0].get("memory_total") if gpus else None
    try:
        vram_bytes = float(count) * float(per_mib) * 1024 * 1024
    except (TypeError, ValueError):
        vram_bytes = None

    def _fits(entry) -> str:
        sz = entry.get("size_bytes") if isinstance(entry, dict) else None
        if sz is None or vram_bytes is None:
            return f'<span class="muted">{_DASH}</span>'
        try:
            fits = float(sz) <= vram_bytes
        except (TypeError, ValueError):
            return f'<span class="muted">{_DASH}</span>'
        return ('<span class="ok">可</span>' if fits
                else '<span class="bad">装不下</span>')

    def _key(e):
        try:
            return float(e.get("size_bytes") or 0)
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted([e for e in inv if isinstance(e, dict)], key=_key, reverse=True)
    rows = []
    for e in ordered:
        arch = e.get("architectures") or e.get("model_type")
        rows.append([
            _e(e.get("dir") or e.get("path")),
            _gi(e.get("size_bytes")),
            _e(e.get("shards") or e.get("num_shards")),
            _e(e.get("quant_method")),
            _fmt_any(arch),
            _e(e.get("dtype")),
            _fits(e),
        ])
    tbl = _table(["目录", "大小 GiB", "分片数", "量化", "架构", "dtype", "能否装下"], rows)
    return f'<h2>4. 模型介质</h2>{tbl}'


def _section_occupancy(host: dict, port) -> str:
    dk = host.get("docker") or {}
    conts = _as_list(dk.get("containers"))
    crows = [
        [_e(c.get("name")), _e(c.get("image")), _e(c.get("status")), _fmt_ports(c.get("ports"))]
        for c in conts if isinstance(c, dict)
    ]
    ctbl = _table(["容器", "镜像", "状态", "端口"], crows)

    ports = host.get("ports") or {}
    plist = _as_list(ports.get("listening"))
    if plist:
        items = "".join(
            f"<li>{_e(p.get('port') if isinstance(p, dict) else p)}"
            + (f" <span class='muted'>({_e(p.get('process') or p.get('proto'))})</span>"
               if isinstance(p, dict) and (p.get("process") or p.get("proto")) else "")
            + "</li>"
            for p in plist
        )
        ptbl = f'<h3>监听端口</h3><ul class="list">{items}</ul>'
    else:
        ptbl = '<h3>监听端口</h3><p class="muted">无。</p>'

    conflict = ""
    if ports.get("vllm_port_conflict"):
        conflict = (f'<div class="alert">端口 {_e(port)} 已被占用，'
                    f'与模板默认 vllm_port 冲突。</div>')

    users = host.get("users") or {}
    ulog = _as_list(users.get("logged_in"))
    if ulog:
        items = "".join(
            f"<li>{_e(u.get('user') if isinstance(u, dict) else u)}"
            + (f" <span class='muted'>@{_e(u.get('from'))}</span>"
               if isinstance(u, dict) and u.get("from") else "")
            + "</li>"
            for u in ulog
        )
        utbl = f'<h3>登录用户</h3><ul class="list">{items}</ul>'
    else:
        utbl = '<h3>登录用户</h3><p class="muted">无。</p>'

    if not crows and not plist and not ulog and not conflict:
        empty = '<div class="freeok">未见外部占用。</div>'
    else:
        empty = ""
    return (
        '<h2>5. 当前占用 / 共租</h2>'
        + ctbl + ptbl + conflict + utbl + empty
    )


def _section_checklist(host: dict) -> str:
    items = _as_list(host.get("checklist"))
    if not items:
        return '<h2>6. 系统检查清单</h2><p class="muted">无检查项。</p>'
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append([_badge(it.get("status")), _e(it.get("name")),
                    _e(it.get("detail")), _e(it.get("fix"))])
    tbl = _table(["状态", "项目", "详情", "修复建议"], rows, "check")
    return f'<h2>6. 系统检查清单</h2>{tbl}'


def _section_conclusion(host: dict, port) -> str:
    gpu = host.get("gpu") or {}
    gpus = _as_list(gpu.get("gpus"))
    count = gpu.get("count")
    if count is None:
        count = len(gpus)
    gname = gpus[0].get("name") if gpus else None
    topo = ((gpu.get("topo") or {}).get("summary"))
    groups = _as_list(host.get("numa_groups"))

    s: list[str] = []
    if gname and count:
        s.append(f"GPU {count}×{gname} 可用")
    if topo == "nvlink-full":
        s.append("NVLink 全互联，tp 可放大到全卡数")
    elif topo == "pcie":
        if groups:
            try:
                mx = max(len(_as_list(g.get("gpus"))) for g in groups if isinstance(g, dict))
            except ValueError:
                mx = None
            s.append(f"无 NVLink（仅 PCIe），tp 建议限制在同一 NUMA 组内（≤ {mx}）")
        else:
            s.append("无 NVLink（仅 PCIe）")

    ports = host.get("ports") or {}
    if ports.get("vllm_port_conflict"):
        s.append(f"端口 {port} 已被占用，需改端口")

    ncont = len(_as_list((host.get("docker") or {}).get("containers")))
    nport = len(_as_list(ports.get("listening")))
    nuser = len(_as_list((host.get("users") or {}).get("logged_in")))
    if ncont or nport or nuser:
        s.append("主机存在外部占用（容器 / 端口 / 登录用户），需评估共租影响")

    seen: dict[str, str] = {}
    for it in _as_list(host.get("checklist")):
        if isinstance(it, dict) and it.get("status") in ("fail", "warn"):
            seen.setdefault(str(it.get("name")), _e(it.get("detail")))
    for nm, detail in seen.items():
        s.append(f"{_e(nm)}：{detail}")

    if not s:
        s.append("未见明显阻塞项")
    return f'<h2>7. 结论</h2><div class="concl">{"；".join(s)}。</div>'


# ---------------------------------------------------------------- 对外接口
def render_host_report(target: dict, facts: dict, harness: dict,
                       model_profile: dict) -> str:
    """返回一段完整、自包含、可离线打开的 HTML 文档字符串。"""
    target = target or {}
    facts = facts or {}
    harness = harness or {}
    port = ((harness.get("remote") or {}).get("vllm_port"))

    hosts_meta = {}
    for h in _as_list((target or {}).get("hosts")):
        if isinstance(h, dict) and h.get("name"):
            hosts_meta[h["name"]] = h

    order = [n for n in hosts_meta if n in facts]
    order += [n for n in facts if n not in hosts_meta]

    body = []
    for name in order:
        host = facts.get(name) or {}
        hinfo = hosts_meta.get(name) or {"name": name}
        title = (
            '<div class="hosttitle">'
            f'<span class="nm">{_e(host.get("hostname") or name)}</span>'
            f'<span class="meta">ssh {_e(hinfo.get("ssh"))} · role '
            f'{_e(hinfo.get("role") or host.get("role"))}</span></div>'
        )
        body.append(
            f'<section class="host">{title}'
            + _summary_cards(host, hinfo)
            + _section_gpu(host)
            + _section_net(host)
            + _section_docker(host)
            + _section_models(host)
            + _section_occupancy(host, port)
            + _section_checklist(host)
            + _section_conclusion(host, port)
            + "</section>"
        )

    return (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>主机探索报告</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n<div class=\"wrap\">\n"
        "<h1>主机探索报告</h1>\n"
        "<p class=\"sub\">Phase 1 环境探测结果 · 单文件离线视图</p>\n"
        + "\n".join(body)
        + "\n</div>\n</body>\n</html>\n"
    )


if __name__ == "__main__":  # 自测：读 env.json + model_profile.json 渲染 host_report.html
    import json

    with open("env.json", encoding="utf-8") as f:
        env = json.load(f)
    with open("model_profile.json", encoding="utf-8") as f:
        mp = json.load(f)

    harness = {"remote": env.get("remote", {})}
    html_out = render_host_report(env.get("target") or {}, env.get("hosts") or {},
                                  harness, mp)
    with open("host_report.html", "w", encoding="utf-8") as f:
        f.write(html_out)

    # 基本自检
    assert "<script" not in html_out.lower()
    assert "http://" not in html_out and "https://" not in html_out
    print(f"host_report.html 已生成：{len(html_out)} 字符，无 <script>，无外链。")
