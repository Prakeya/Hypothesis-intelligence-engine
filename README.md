---
title: Hypothesis Intelligence Engine
emoji: 🔬
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
---

# Hypothesis Intelligence Engine (OpenEnv)
> **AI-Driven Logic Auditing & Hallucination Detection Environment**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blueviolet)](https://github.com/meta-pytorch/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Motivation
Modern LLMs excel at generating claims but frequently struggle with **grounding** those claims in empirical data. The Hypothesis Intelligence Engine provides a rigorous OpenEnv-compliant simulation where agents must audit data artifacts, generate verifiable hypotheses, and synthesize conclusions without fabricating evidence (Hallucination).

## Key Features
- **Logic Auditing Protocol**: Agents follow structured logic parsing (Monotonic Trends, Causal Filtering, Strict Absolute Statements formatting).
- **Hallucination Detection Guard**: Programmatic reasoning parsing that aggressively traps numerical fabrication or extra-empirical assumptions.
- **FastAPI Backend Structure**: Application is built fully as a scalable Docker-deployed JSON API listening on Port 7860.
- **Benchmark Suite**: Native evaluator checking standard edge-cases for >95% accuracy targets.

## OpenEnv Specifications

### Observation Space
- `task_id`: Unique identifier for the audit task.
- `claim`: The hypothesis statement under evaluation.
- `dataset`: The raw data artifacts provided for analysis.

### Action / Output Space
The internal evaluation outputs cleanly mapping to logic triggers:
- `verdict`: "Supported", "Refuted", or "Inconclusive".
- `reasoning`: The step-by-step logic chain mapped specifically against variables.
- `confidence_score`: Evaluated metric mapping.

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
