import json
import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI


# -----------------------
# Parsing / clustering
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
WS_RE = re.compile(r"\s+")


def normalize_line(line: str) -> str:
    line = line.strip()
    line = HEX_RE.sub("0x<HEX>", line)
    line = NUM_RE.sub("<N>", line)
    line = WS_RE.sub(" ", line)
    return line


def sig(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


@dataclass
class Event:
    file_name: str
    raw: str
    normalized: str
    signature: str
    fileline: Optional[str]


def parse_logs(files: List[Dict[str, Any]]) -> List[Event]:
    """
    files: list of dicts {name: str, text: str}
    """
    events: List[Event] = []
    for f in files:
        name = f["name"]
        for line in f["text"].splitlines():
            if not ERROR_RE.search(line):
                continue
            norm = normalize_line(line)
            s = sig(norm)
            m = FILELINE_RE.search(line)
            fl = f"{m.group('file')}:{m.group('line')}" if m else None
            events.append(Event(file_name=name, raw=line.strip(), normalized=norm, signature=s, fileline=fl))
    return events


def cluster_events(events: List[Event]) -> List[Dict[str, Any]]:
    clusters = defaultdict(lambda: {
        "hits": 0,
        "files": set(),
        "examples": [],
        "filelines": Counter(),
        "normalized": None,
    })

    for e in events:
        c = clusters[e.signature]
        c["hits"] += 1
        c["files"].add(e.file_name)
        if c["normalized"] is None:
            c["normalized"] = e.normalized
        if e.fileline:
            c["filelines"][e.fileline] += 1
        if len(c["examples"]) < 3:
            c["examples"].append(e.raw)

    ranked = []
    for s, c in clusters.items():
        ranked.append({
            "signature": s,
            "normalized": c["normalized"] or "",
            "hits": c["hits"],
            "files_affected": len(c["files"]),
            "top_filelines": c["filelines"].most_common(5),
            "examples": c["examples"],
        })

    ranked.sort(key=lambda x: (x["files_affected"], x["hits"]), reverse=True)
    return ranked


# -----------------------
# LLM summarization
# -----------------------
SYSTEM = """You are a silicon/verification log triage assistant.
You MUST only use the evidence in the provided clusters. Do not invent root causes.
Output:
1) Top issues (ranked) with evidence
2) Likely category (scoreboard/assertion, timing, reset/x-prop, constraints/clocking, tooling/env, unknown)
3) Concrete next steps (2-4), grounded in evidence
Keep it crisp and practical.
"""


def summarize_with_llm(model: str, top_clusters: List[Dict[str, Any]]) -> str:
    client = OpenAI()
    payload = {"clusters": top_clusters}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Log Triage Agent", layout="wide")
st.title("Log Triage Agent (Streamlit)")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"], index=0)
    top_k = st.slider("How many clusters to send to LLM", min_value=3, max_value=15, value=6)
    st.caption("Tip: smaller models are cheaper + faster. Keep clusters small for trust.")

uploaded = st.file_uploader(
    "Upload one or more log files (.log/.txt/.out)",
    type=["log", "txt", "out"],
    accept_multiple_files=True
)

col1, col2 = st.columns([1, 1], gap="large")

if uploaded:
    files = []
    for uf in uploaded:
        text = uf.read().decode("utf-8", errors="replace")
        files.append({"name": uf.name, "text": text})

    events = parse_logs(files)
    clusters = cluster_events(events)

    with col1:
        st.subheader("Deterministic triage (evidence)")
        st.write(f"Files uploaded: **{len(files)}**")
        st.write(f"Error-like lines found: **{len(events)}**")
        st.write(f"Clusters found: **{len(clusters)}**")

        if clusters:
            st.markdown("### Top clusters")
            for i, c in enumerate(clusters[:min(20, len(clusters))], start=1):
                with st.expander(f"{i}. sig={c['signature']} | hits={c['hits']} | files={c['files_affected']}"):
                    st.code(c["normalized"])
                    if c["top_filelines"]:
                        st.write("Top file:line:")
                        st.write(", ".join([f"{fl} ({n})" for fl, n in c["top_filelines"]]))
                    st.write("Examples:")
                    for ex in c["examples"]:
                        st.code(ex)

        st.download_button(
            "Download clusters.json",
            data=json.dumps(clusters, indent=2).encode("utf-8"),
            file_name="clusters.json",
            mime="application/json",
            disabled=(len(clusters) == 0),
        )

    with col2:
        st.subheader("LLM summary + next steps")
        if not clusters:
            st.info("No clusters to summarize. Upload logs containing ERROR/UVM_ERROR/etc.")
        else:
            if st.button("Run LLM triage", type="primary"):
                top = clusters[:top_k]
                with st.spinner("Calling model..."):
                    try:
                        answer = summarize_with_llm(model, top)
                        st.markdown(answer)
                        st.download_button(
                            "Download summary.md",
                            data=answer.encode("utf-8"),
                            file_name="triage_summary.md",
                            mime="text/markdown",
                        )
                    except Exception as e:
                        st.error(f"LLM call failed: {e}")
else:
    st.info("Upload some log files to get started.")

