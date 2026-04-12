---
title: Hypothesis Intelligence Engine
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.42.0"
app_file: app.py
pinned: false
---

# 🧬 Hypothesis Intelligence Engine

## 🛡️ Hallucination Checker Improvements
- **Issue Found**: Brittle string-based matching caused false positives on numeric precision (e.g., `8` vs `8.0`). Integration was missing from the root environment path.
- **Fix**: Implemented a robust numeric-first checker using `math.isclose` for float comparison.
- **Files Modified**: `env.py`, `server/env.py`.
- **Validation**: No changes were made to UI, API contracts, or core scoring weights. The checker now populates `Action.hallucination_check` with detailed detection logs.

# Hypothesis Intelligence Engine (OpenEnv)
> **AI-Driven Logic Auditing & Hallucination Detection Environment**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blueviolet)](https://github.com/meta-pytorch/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Motivation
Modern LLMs excel at generating claims but frequently struggle with **grounding** those claims in empirical data. The Hypothesis Intelligence Engine provides a rigorous OpenEnv-compliant simulation where agents must audit data artifacts, generate verifiable hypotheses, and synthesize conclusions without fabricating evidence (Hallucination).

## Key Features
- **Multi-Step Reasoning Agent**: An upgraded inference pipeline utilizing iterative self-correction loops. The agent receives reward feedback from the environment and explicitly refines its reasoning over multiple steps.
- **Strict Phase 2 OpenEnv Compliance**: Full implementation of standardized `[START]`, `[STEP]`, and `[END]` event logging required by stringent Deep Validator pipelines.
- **Fail-Safe Pydantic Guardrails**: Safe isolation of agent execution using dual-layered try/except blocks and advanced Pydantic attribute resolution mapped correctly across varying legacy/new observation schemas.
- **Logic Auditing Protocol**: Agents follow structured logic parsing (Monotonic Trends, Causal Filtering, Strict Absolute Statements formatting).
- **Hallucination Detection Guard**: Programmatic mathematical parsing that aggressively penalizes numerical fabrication and bounds the final reward between 0.0 and 1.0.
- **FastAPI Backend Structure**: Application is built fully as a scalable Docker-deployed JSON API listening on Port 7860.

## OpenEnv Specifications

### Observation Space
- `task_id`: Unique identifier for the audit task explicitly detected to match OpenEnv validators.
- `claim`: The hypothesis statement under evaluation.
- `evidence_block`: The raw data artifacts provided for analysis.

### Action / Output Space
The internal evaluation outputs cleanly mapping to logic triggers:
- `verdict`: "Supported", "Refuted", or "Inconclusive".
- `reasoning`: The step-by-step logic chain mapped specifically against variables.
- `confidence_score`: Evaluated metric mappings dynamically factoring into a 70/30 weighted reward equation.

## Setup & Deployment

### Containerized Execution natively for HF Spaces
1. **Docker Build Requirements**: Image targets purely Python `3.10`.
2. **Launch Logic**: Server utilizes `uvicorn`. Connect securely:
```bash
uvicorn inference:app --host 0.0.0.0 --port 7860
```

### Native Request Payload Simulation
You can trigger local inferences leveraging Standard JSON HTTP requests locally tracking through FastAPI:
```json
// POST /predict
{
  "claim": "AI improves productivity",
  "dataset": ["Automation saves time", "AI reduces manual effort"]
}
```

## Project Links
- **Hugging Face Space Live Demo**: [Hypothesis-Intelligence-Engine](https://huggingface.co/spaces/Prakeya/Hypothesis-intelligence-engine)
- **GitHub Repository**: [Source Code](https://github.com/Prakeya/Hypothesis-intelligence-engine)
