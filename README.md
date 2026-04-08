# Hypothesis Intelligence Engine (OpenEnv)

## Motivation
Modern AI agents frequently struggle with grounding their reasoning in provided data artifacts, often leading to "Hallucinations." This environment simulates a real-world **Logic Auditing** task where an agent must verify a claim against a provided dataset, generate a valid investigative protocol, and synthesize a conclusion.

## OpenEnv Specifications

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
- **Base Participation**: 0.5
- **Logical Consistency**: +0.3
- **Reasoning Depth**: +0.2
- **Hallucination Penalty**: -1.0 (strikes reward to zero)

## Tasks & Graders

| ID | Difficulty | Description | Grader Type |
|----|------------|-------------|-------------|
| baseline-correlation | Easy | Simple linear trend verification in small sets. | Programmatic |
| nonlinear-dependency| Medium | Nonlinear trend with small anomalies. | Programmatic |
| confounding-trap | Hard | Confounding variables with a non-monotonic "trap." | Programmatic |

## Setup & Usage

### Local Execution (Docker)
1. Build: `docker build -t hypothesis-engine .`
2. Run: `docker run -p 8501:8501 hypothesis-engine`

### Baseline Inference
Run the standard OpenEnv audit:
```bash
python inference.py
```

## Baseline Performance
The baseline agent (GPT-3.5) achieves an average score of **0.82** across all tasks.
