# Agentic

An LLM-based agent for triaging silicon/verification logs using a
**trust-first architecture**:
- deterministic parsing & clustering
- LLM-driven summarization and next-step planning
- optional Streamlit UI for interactive analysis

## Features
- Upload or scan log files (.log/.txt/.out)
- Cluster errors by normalized signatures
- Rank issues by impact
- Evidence-grounded LLM summaries
- Designed for silicon verification & EDA workflows

## Architecture
LLM --> Tools --> Evidence --> Analysis/Summary

## How to run

### CLI interface example:
$ python3 log_agent.py --ask "Triage ./logs and summarize the main issues"

### Streamlit UI:
$ streamlit run app.py


