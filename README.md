# Hypothesis Intelligence Engine (OpenEnv)
> **AI-Driven Logic Auditing & Hallucination Detection Environment**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blueviolet)](https://github.com/meta-pytorch/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Motivation
Modern LLMs excel at generating claims but frequently struggle with **grounding** those claims in empirical data. The Hypothesis Intelligence Engine provides a rigorous OpenEnv-compliant simulation where agents must audit data artifacts, generate verifiable hypotheses, and synthesize conclusions without fabricating evidence (Hallucination).

##  Key Features
- **Logic Auditing Protocol**: Agents follow a structured *Hypothesis  Method  Reasoning  Conclusion* flow.
- **Hallucination Detection**: Programmatic grading engine that specifically targets numerical fabrication.
- **Task Hierarchy**: 3 distinct tasks ranging from simple linear trends to non-monotonic confounding "traps."
- **Crystal Obsidian UI**: A premium, research-grade dashboard built with Streamlit for interactive auditing.

##  OpenEnv Specifications

### Observation Space
Typed Pydantic model `Observation`:
- `task_id`: Unique identifier for the audit task.
- `claim`: The hypothesis statement under evaluation.
- `dataset`: The raw data artifacts provided for analysis.
- `independent_var/dependent_var`: Metadata for statistical grounding.

### Action Space
Typed Pydantic model `Action`:
- `hypothesis`: The agent's refined investigative hypothesis.
- `method`: The technical protocol for the audit.
- `reasoning_steps`: The step-by-step logic chain.
- `conclusion`: Final synthesis grounded in data.

### Reward Function
The environment provides partial progress signals (0.0 - 1.0):
- **Logical Consistency**: +0.3
- **Reasoning Depth**: +0.2
- **Data Grounding**: +0.5
- **Hallucination Penalty**: -1.0 (strikes reward to 0.0)

##  Tasks & Graders

| ID | Difficulty | Objective | Grader |
|----|------------|-----------|--------|
| `baseline-correlation` | Easy | Verify linear trends in small datasets. | Deterministic |
| `nonlinear-dependency` | Medium | Detect anomalies in nonlinear growth. | Deterministic |
| `confounding-trap` | Hard | Identify non-monotonic "traps" in noisy data. | Deterministic |

##  Setup & Usage

###  Containerized Execution (Recommended)
1. **Build**: `docker build -t hypothesis-engine .`
2. **Run**: `docker run -p 8501:8501 hypothesis-engine`

###  Local Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

###  Baseline Inference
To run the standard OpenEnv audit:
```bash
python inference.py
```

##  Baseline Performance
The baseline agent (GPT-3.5) achieves an average score of **0.82** across all tasks with a 0% hallucination rate on the "Easy" tier.

##  Project Links
- **Hugging Face Space**: [Live Demo](https://huggingface.co/spaces/Prakeya/hypothesis-intelligence-v4)
- **GitHub Repository**: [Source Code](https://github.com/Prakeya/Hypothesis-intelligence-engine)
