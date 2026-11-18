#!/usr/bin/env python3
"""
按传感器（stage）拆分聚合，每张表首行为 CAD 参考。

用法:
  python3 aggregate_calibration.py <root1> <root2> [...] [--continue-seq] [-o out.md]

示例:
  python3 aggregate_calibration.py \
      ~/Desktop/rei_calibr_ws/rei-calibr/configs/experiments/10-29 \
      ~/Desktop/rei_calibr_ws/rei-calibr/configs/experiments/10-29-tree \
      --continue-seq \
      -o ~/Desktop/rei_calibr_ws/rei-calibr/configs/experiments/TOTAL_10-29.md
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

STAGE_ORDER = [
    "radar_main", "aid_radar_main", "event_main", "livox_imu_main", "dvs_imu_main",
]


# ---------------- 解析 ----------------

def parse_md_table(lines, start):
    i, n = start, len(lines)
    while i < n and not lines[i].strip().startswith("|"):
        if lines[i].startswith("#"):
            return None, [], i
        i += 1
    if i >= n:
        return None, [], i
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    i += 1
    if i < n and re.match(r"^\|[\s\-:|]+\|?$", lines[i].strip()):
        i += 1
    rows = []
    while i < n and lines[i].strip().startswith("|"):
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1
    return header, rows, i


def parse_sections(text):
    sections, current = {}, []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(#{2,4})\s+(.+?)\s*$", lines[i])
        if m:
            level = len(m.group(1))
            current = current[: level - 2] + [m.group(2).strip()]
            i += 1
            continue
        if lines[i].strip().startswith("|") and current:
            header, rows, ni = parse_md_table(lines, i)
            if header is not None:
                sections[current[-1]] = {"header": header, "rows": rows}
                i = ni
                continue
        i += 1
    return sections


def parse_se3(text):
    blocks, lines = {}, text.splitlines()
    i, cur = 0, None
    while i < len(lines):
        m = re.match(r"^###\s+(.+?):\s*`", lines[i])
        if m:
            cur = m.group(1).strip()
        if lines[i].strip().startswith("```text") and cur:
            j, content = i + 1, []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                content.append(lines[j])
                j += 1
            blocks[cur] = "\n".join(content).strip()
            cur = None
            i = j
        i += 1
    return blocks


def load_seq(p):
    t = p.read_text(encoding="utf-8", errors="replace")
    return {"sections": parse_sections(t), "se3": parse_se3(t)}


# ---------------- 工具 ----------------

def fmt_md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        cells = [(str(c).replace("|", "\\|") if c is not None else "") for c in r]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[: len(header)]
        out.append("| " + " | ".join(cells) + " |")
    return out


def col(header, *keywords):
    for i, h in enumerate(header):
        for kw in keywords:
            if kw.lower() in h.lower():
                return i
    return -1


def stage_rank(n):
    return STAGE_ORDER.index(n) if n in STAGE_ORDER else len(STAGE_ORDER)


def extract_cad(session):
    """返回 {stage: {'p': str, 'euler': str}}，同 session 内取首个有 CAD 的序列。"""
    cad = {}
    for seq in sorted(session["seqs"]):
        c = session["seqs"][seq]["sections"].get("CAD Reference")
        if not c:
            continue
        p_i = col(c["header"], "p_GT_sensor", "p_GT")
        e_i = col(c["header"], "euler")
        for row in c["rows"]:
            if not row or not row[0]:
                continue
            cad[row[0]] = {
                "p": row[p_i] if p_i >= 0 and p_i < len(row) else "",
                "euler": row[e_i] if e_i >= 0 and e_i < len(row) else "",
            }
        break  # 同一 session 内 CAD 相同，取一个序列即可
    return cad


def main():
    ap = argparse.ArgumentParser(description="按传感器拆分聚合 calibration summary")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--no-se3", dest="with_se3", action="store_false")
    ap.add_argument("--continue-seq", action="store_true",
                    help="后续 session 的序列编号接前一个 session 最大编号之后连续编号")
    ap.set_defaults(with_se3=True)
    args = ap.parse_args()

    roots = [Path(p).resolve() for p in args.roots]
    for r in roots:
        if not r.is_dir():
            print(f"不是目录: {r}", file=sys.stderr)
            sys.exit(1)

    sessions = []
    for r in roots:
        seqs = {}
        for p in r.iterdir():
            if p.is_dir() and p.name.isdigit():
                sp = p / "summary.md"
                if sp.is_file():
                    try:
                        seqs[int(p.name)] = load_seq(sp)
                    except Exception as e:
                        print(f"解析 {sp} 失败: {e}", file=sys.stderr)
        sessions.append({
            "name": r.name,
            "seqs": seqs,
            "cad": extract_cad({"seqs": seqs}),
        })

    if all(len(s["seqs"]) == 0 for s in sessions):
        print("没有找到任何 summary.md", file=sys.stderr)
        sys.exit(1)

    # ---------- 显示编号 ----------
    display = {}
    if args.continue_seq:
        nxt = 1
        for s in sessions:
            for seq in sorted(s["seqs"]):
                display[(s["name"], seq)] = nxt
                nxt += 1
    else:
        for s in sessions:
            for seq in sorted(s["seqs"]):
                display[(s["name"], seq)] = seq

    # ---------- 输出路径 ----------
    if args.output:
        out_path = Path(args.output)
    else:
        tag = "_".join(s["name"] for s in sessions)
        out_path = roots[0].parent / f"aggregate_{tag}.md"

    lines = []
    lines.append("# Calibration Aggregate (per-modality)")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Sessions: {len(sessions)}")
    for s in sessions:
        seq_str = ", ".join(f"{display[(s['name'], q)]}<-{q}" for q in sorted(s["seqs"]))
        lines.append(f"  - `{s['name']}` ({len(s['seqs'])} seqs): {seq_str or '（空）'}")
    total = sum(len(s["seqs"]) for s in sessions)
    lines.append(f"- Total sequences: {total}")
    lines.append(f"- Seq numbering: {'continue' if args.continue_seq else 'per-session'}")
    lines.append("")

    # ---------- 收集 Stage Results ----------
    # stage -> list of (session_name, display_seq, original_seq, row_cells)
    stage_data = {st: [] for st in STAGE_ORDER}
    stage_header = None   # 用第一个有 Stage Results 的序列的表头
    for s in sessions:
        for seq in sorted(s["seqs"]):
            sr = s["seqs"][seq]["sections"].get("Stage Results")
            if not sr:
                continue
            if stage_header is None:
                stage_header = sr["header"]
            for row in sr["rows"]:
                if not row or not row[0]:
                    continue
                st = row[0]
                if st not in stage_data:
                    stage_data[st] = []
                stage_data[st].append(
                    (s["name"], display[(s["name"], seq)], seq, row)
                )

    if stage_header is not None:
        # 定位关键列
        i_type  = col(stage_header, "type")
        i_rms   = col(stage_header, "rms")
        i_tau   = col(stage_header, "tau")               # 第一个 tau 是 tau (s)
        i_tepo  = col(stage_header, "epoch")             # tau after epoch alignment
        i_p     = col(stage_header, "p_GT_sensor", "p_GT")
        i_eul   = col(stage_header, "euler")
        i_scale = col(stage_header, "scale")

        # 如果 i_tau 和 i_tepo 是同一列（都命中 "tau"），重新精确找
        if i_tau == i_tepo:
            for idx, h in enumerate(stage_header):
                if "tau" in h.lower() and "epoch" not in h.lower():
                    i_tau = idx
                    break
            for idx, h in enumerate(stage_header):
                if "epoch" in h.lower():
                    i_tepo = idx
                    break

        def cell(row, idx, default="—"):
            return row[idx] if 0 <= idx < len(row) and row[idx] != "" else default

        # ---------- 1. 按传感器拆分 ----------
        lines.append("## 1. Stage Results by Modality")
        lines.append("")
        lines.append("每张表首行为 CAD 参考（`CAD`），其后为各序列估计值。"
                     "`Seq` 为全局编号，`OriginalSeq` 为源目录编号。")
        lines.append("")

        for stage in STAGE_ORDER:
            entries = stage_data.get(stage, [])
            if not entries:
                continue
            entries.sort(key=lambda e: int(e[1]))  # 按全局 Seq 升序

            lines.append(f"### {stage}")
            lines.append("")

            # 表头
            header = [
                "Session", "Seq", "OriginalSeq", "Type",
                "RMS", "tau (s)", "tau (epoch)", "p_GT_sensor (m)",
                "Euler xyz sensor -> GT (deg)", "Scale",
            ]

            rows_out = []

            # ---- CAD 行（首行） ----
            # 收集所有 session 对这个 stage 的 CAD
            cad_by_sess = {}
            for s in sessions:
                c = s["cad"].get(stage)
                if c:
                    cad_by_sess[s["name"]] = c

            if cad_by_sess:
                # 去重判断
                uniq = {(v["p"], v["euler"]) for v in cad_by_sess.values()}
                if len(uniq) == 1:
                    # 所有 session 一致 → 一行
                    p_val, e_val = next(iter(uniq))
                    rows_out.append([
                        "**CAD** (all)", "—", "—", "CAD",
                        "—", "—", "—", p_val, e_val, "—",
                    ])
                else:
                    # 不一致 → 每 session 一行
                    for s in sessions:
                        c = cad_by_sess.get(s["name"])
                        if not c:
                            continue
                        rows_out.append([
                            f"**CAD** (`{s['name']}`)", "—", "—", "CAD",
                            "—", "—", "—", c["p"], c["euler"], "—",
                        ])

            # ---- 数据行 ----
            for sess, dseq, oseq, row in entries:
                rows_out.append([
                    sess,
                    str(dseq),
                    str(oseq),
                    cell(row, i_type),
                    cell(row, i_rms),
                    cell(row, i_tau),
                    cell(row, i_tepo),
                    cell(row, i_p),
                    cell(row, i_eul),
                    cell(row, i_scale),
                ])

            lines.extend(fmt_md_table(header, rows_out))
            lines.append("")

    # ---------- 2. Data Quality（大表，保持原样） ----------
    q_header, q_rows = None, []
    for s in sessions:
        for seq in sorted(s["seqs"]):
            q = s["seqs"][seq]["sections"].get("Data Quality and Motion Excitation")
            if not q:
                continue
            if q_header is None:
                q_header = q["header"]
            for row in q["rows"]:
                if row and row[0]:
                    q_rows.append([s["name"], display[(s["name"], seq)], seq] + row)
    if q_header:
        q_rows.sort(key=lambda r: (int(r[1]), stage_rank(r[3])))
        lines.append("## 2. Data Quality and Motion Excitation")
        lines.append("")
        lines.append("跨所有序列的大表，`Session`/`Seq`/`OriginalSeq` 用于定位。")
        lines.append("")
        header = ["Session", "Seq", "OriginalSeq"] + q_header
        rows = [[r[0], str(r[1]), str(r[2])] + r[3:] for r in q_rows]
        lines.extend(fmt_md_table(header, rows))
        lines.append("")

    # ---------- 3. CAD 一致性检查 ----------
    lines.append("## 3. CAD Reference Consistency")
    lines.append("")
    all_stages = [st for st in STAGE_ORDER if any(st in s["cad"] for s in sessions)]
    if not all_stages:
        lines.append("（未找到 CAD Reference 数据）")
        lines.append("")
    else:
        header = ["Stage"] + [f"`{s['name']}` p_GT_sensor" for s in sessions] \
                            + [f"`{s['name']}` Euler" for s in sessions] + ["一致?"]
        rows = []
        for stage in all_stages:
            row = [stage]
            vals = []
            for s in sessions:
                c = s["cad"].get(stage, {})
                row.append(c.get("p", "—"))
                vals.append((c.get("p", ""), c.get("euler", "")))
            for s in sessions:
                c = s["cad"].get(stage, {})
                row.append(c.get("euler", "—"))
            row.append("✅" if len(set(vals)) == 1 else "❌")
            rows.append(row)
        lines.extend(fmt_md_table(header, rows))
        lines.append("")

    # ---------- 4. SE(3) 附录 ----------
    if args.with_se3:
        lines.append("## 4. Appendix: SE(3) Extrinsics")
        lines.append("")
        for s in sessions:
            lines.append(f"### Session: {s['name']}")
            lines.append("")
            for seq in sorted(s["seqs"]):
                se3 = s["seqs"][seq]["se3"]
                if not se3:
                    continue
                label = f"Seq {display[(s['name'], seq)]}"
                if display[(s["name"], seq)] != seq:
                    label += f" (original {seq})"
                lines.append("<details>")
                lines.append(f"<summary>{label}</summary>")
                lines.append("")
                for stage in STAGE_ORDER:
                    if stage in se3:
                        lines.append(f"#### {stage}")
                        lines.append("")
                        lines.append("```text")
                        lines.append(se3[stage])
                        lines.append("```")
                        lines.append("")
                lines.append("</details>")
                lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入: {out_path}")
    for s in sessions:
        print(f"  {s['name']}: {len(s['seqs'])} 序列")
    print(f"  Total: {total} 序列，编号模式: {'continue' if args.continue_seq else 'per-session'}")


if __name__ == "__main__":
    main()
