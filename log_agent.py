#!/usr/bin/env python3
import argparse
import json
import os
import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# -----------------------
# Deterministic tool logic
# -----------------------

ERROR_PATTERNS = [
    r"\b(UVM_FATAL|UVM_ERROR|UVM_WARNING)\b",
    r"\bFATAL\b",
    r"\bERROR\b",
    r"\bASSERT(ION)?\b.*\bFAIL\b",
    r"\bassertion failed\b",
    r"\bSegmentation fault\b|\bSEGFAULT\b",
    r"\bTiming violation\b",
    r"\bsetup\b.*\bviolation\b|\bhold\b.*\bviolation\b",
]
ERROR_RE = re.compile("|".join(f"(?:{p})" for p in ERROR_PATTERNS), re.IGNORECASE)

FILELINE_RE = re.compile(r"(?P<file>[\w./-]+\.(?:sv|v|vh|c|cc|cpp|h|hpp|py)):(?P<line>\d+)")
HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
NUM_RE = re.compile(r"\b\d+\b")
WS_RE  = re.compile(r"\s+")

def _normalize(s: str) -> str:
    s = s.strip()
    s = HEX_RE.sub("0x<HEX>", s)
    s = NUM_RE.sub("<N>", s)
    s = WS_RE.sub(" ", s)
    return s

def _sig(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def parse_and_cluster_logs(
    log_dir: str,
    max_files: int = 400,
    max_lines_per_file: int = 200_000
) -> Dict[str, Any]:
    """
    Parse logs, cluster error-like lines by normalized signature, and rank.
    Returns JSON-serializable dict.
    """
    p = Path(log_dir)
    if not p.exists() or not p.is_dir():
        return {"ok": False, "error": f"log_dir not found or not a dir: {log_dir}"}

    files = []
    for ext in ("*.log", "*.txt", "*.out"):
        files.extend(p.glob(f"**/{ext}"))
    files = sorted(set(files))[:max_files]

    clusters = defaultdict(lambda: {
        "hits": 0,
        "tests": set(),
        "files": set(),
        "examples": [],
        "filelines": Counter(),
        "normalized": None,
    })

    for fp in files:
        test = fp.stem
        try:
            lines = fp.read_text(errors="replace").splitlines()
        except Exception:
            continue

        for line in lines[:max_lines_per_file]:
            if not ERROR_RE.search(line):
                continue
            norm = _normalize(line)
            sig = _sig(norm)

            c = clusters[sig]
            c["hits"] += 1
            c["tests"].add(test)
            c["files"].add(str(fp))
            if c["normalized"] is None:
                c["normalized"] = norm

            m = FILELINE_RE.search(line)
            if m:
                c["filelines"][f"{m.group('file')}:{m.group('line')}"] += 1

            if len(c["examples"]) < 3:
                c["examples"].append(line.strip())

    # rank: tests affected > hits > files affected
    ranked = []
    for sig, c in clusters.items():
        ranked.append({
            "signature": sig,
            "normalized": c["normalized"] or "",
            "hits": c["hits"],
            "tests_affected": len(c["tests"]),
            "files_affected": len(c["files"]),
            "top_filelines": c["filelines"].most_common(5),
            "examples": c["examples"],
            "tests": sorted(list(c["tests"]))[:50],
        })

    ranked.sort(key=lambda x: (x["tests_affected"], x["hits"], x["files_affected"]), reverse=True)

    return {
        "ok": True,
        "log_dir": str(p),
        "files_scanned": len(files),
        "clusters_found": len(ranked),
        "clusters": ranked,
    }

def diff_cluster_summaries(prev_json_path: str, curr_json_path: str, top_k: int = 15) -> Dict[str, Any]:
    """
    Compare two saved cluster JSON outputs from this tool and show what's new/regressed.
    """
    prev = json.loads(Path(prev_json_path).read_text())
    curr = json.loads(Path(curr_json_path).read_text())

    if not prev.get("ok") or not curr.get("ok"):
        return {"ok": False, "error": "prev or curr json missing/invalid (expected output of parse_and_cluster_logs)"}

    prev_map = {c["signature"]: c for c in prev["clusters"]}
    curr_map = {c["signature"]: c for c in curr["clusters"]}

    new_sigs = [s for s in curr_map.keys() if s not in prev_map]
    gone_sigs = [s for s in prev_map.keys() if s not in curr_map]

    regressions = []
    for sig, c in curr_map.items():
        if sig in prev_map:
            dp = c["hits"] - prev_map[sig]["hits"]
            dt = c["tests_affected"] - prev_map[sig]["tests_affected"]
            if dp > 0 or dt > 0:
                regressions.append((sig, dt, dp))

    regressions.sort(key=lambda x: (x[1], x[2]), reverse=True)

    return {
        "ok": True,
        "new_clusters": [curr_map[s] for s in new_sigs[:top_k]],
        "gone_clusters": [prev_map[s] for s in gone_sigs[:top_k]],
        "top_regressions": [
            {"signature": sig, "delta_tests": dt, "delta_hits": dp,
             "normalized": curr_map[sig]["normalized"], "examples": curr_map[sig]["examples"]}
            for (sig, dt, dp) in regressions[:top_k]
        ],
    }

# -----------------------
# LLM "agent" controller
# -----------------------

TOOLS = [
    {
        "type": "function",
        "name": "parse_and_cluster_logs",
        "description": "Parse a directory of log files, cluster error-like lines by normalized signature, and rank clusters.",
        "parameters": {
            "type": "object",
            "properties": {
                "log_dir": {"type": "string", "description": "Path to directory containing log files"},
                "max_files": {"type": "integer", "description": "Maximum number of files to scan", "default": 400},
                "max_lines_per_file": {"type": "integer", "description": "Maximum lines per file to scan", "default": 200000},
            },
            "required": ["log_dir"]
        },
    },
    {
        "type": "function",
        "name": "diff_cluster_summaries",
        "description": "Diff two previously saved JSON outputs from parse_and_cluster_logs to find new clusters and regressions.",
        "parameters": {
            "type": "object",
            "properties": {
                "prev_json_path": {"type": "string", "description": "Path to previous run JSON"},
                "curr_json_path": {"type": "string", "description": "Path to current run JSON"},
                "top_k": {"type": "integer", "description": "Max items to return", "default": 15},
            },
            "required": ["prev_json_path", "curr_json_path"]
        },
    },
]

SYSTEM_INSTRUCTIONS = """You are a silicon/verification log triage agent.
You MUST ground all conclusions in tool outputs.
Do not invent root causes. If uncertain, say what evidence is missing.
Output format:
1) Top issues (ranked) with evidence
2) Likely category (assertion/scoreboard, x-prop/reset, timing, constraints, tooling/env, unknown)
3) Concrete next steps (2-4) that an engineer can try
Keep it crisp and practical.
"""

def run_agent(user_request: str) -> str:
    client = OpenAI()  # uses OPENAI_API_KEY from env :contentReference[oaicite:2]{index=2}

    input_list: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_request},
    ]

    # First call: model decides whether to call tools :contentReference[oaicite:3]{index=3}
    resp = client.responses.create(
        model="gpt-5.2",
        tools=TOOLS,
        input=input_list,
    )

    # Feed tool outputs back until the model stops requesting tools
    input_list += resp.output

    # Tool loop
    while True:
        tool_calls = [item for item in resp.output if item.type == "function_call"]
        if not tool_calls:
            return resp.output_text or ""

        for call in tool_calls:
            name = call.name
            args = json.loads(call.arguments) if call.arguments else {}
            if name == "parse_and_cluster_logs":
                out = parse_and_cluster_logs(**args)
            elif name == "diff_cluster_summaries":
                out = diff_cluster_summaries(**args)
            else:
                out = {"ok": False, "error": f"Unknown tool: {name}"}

            input_list.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(out),
            })

        resp = client.responses.create(
            model="gpt-5.2",
            tools=TOOLS,
            input=input_list,
        )
        input_list += resp.output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", required=True, help="Natural language request to the agent")
    ap.add_argument("--out", default="", help="Write the final answer to this file (e.g. triage_report.md)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY in environment.")

    ans = run_agent(args.ask)
    print(ans)
    if args.out:
        Path(args.out).write_text(ans)
        print(f"\nWrote report to: {args.out}")

if __name__ == "__main__":
    main()

