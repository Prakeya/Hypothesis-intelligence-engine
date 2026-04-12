import random
import uuid
import json
import os
import re
import math
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Literal

# --- Consts ---
EPS = 1e-6

# --- Utils ---

def safe_strict_float(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            x = float(default)
    except Exception:
        x = float(default)
    if x <= 0.0: return EPS
    if x >= 1.0: return 1.0 - EPS
    return x

def check_hallucination(reasoning: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Robust numeric-first hallucination checker.
    """
    clean_text = re.sub(r"(Estimated Correlation|Confidence Score|r=|Reward|Score)[:\s]*[-+]?\d*\.?\d+", "", reasoning, flags=re.I)
    found_numbers = re.findall(r"[-+]?\d*\.?\d+", clean_text)
    
    found_floats = []
    for n in found_numbers:
        try:
            found_floats.append(float(n))
        except ValueError:
            continue

    evidence_values = set()
    for row in evidence:
        for val in row.values():
            try:
                evidence_values.add(float(val))
            except (ValueError, TypeError):
                continue

    whitelist = {0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 100.0}
    
    hallucinated_values = []
    for f in found_floats:
        if f not in evidence_values and f not in whitelist:
            if not any(math.isclose(f, ev, rel_tol=1e-5) for ev in evidence_values):
                hallucinated_values.append(f)

    if hallucinated_values:
        return {"status": "failed", "detected": True, "values": list(set(hallucinated_values))}
    return {"status": "passed", "detected": False, "values": []}

# --- Grader ---

def evaluate_action(action, task, ground_truth=None):
    """
    Strictly returns reward in the open interval (0,1).
    """
    verdict = action.get("verdict", "")
    reasoning = action.get("reasoning", "")
    confidence = safe_strict_float(action.get("confidence", 0.5), default=0.5)

    evidence = task.get("dataset", task.get("evidence_block", []))
    dep_var = task["dependent_var"]
    
    # 1. Hallucination Check
    h_check = check_hallucination(reasoning, evidence)
    hallucination_detected = h_check["detected"]

    # 2. Verdict Verification
    y_vals = [d[dep_var] for d in evidence if dep_var in d]
    verdict_correct = False
    if ground_truth:
        verdict_correct = (verdict == ground_truth)
    elif len(y_vals) >= 2:
        is_pos = y_vals[-1] > y_vals[0]
        if is_pos and verdict == "Supported": verdict_correct = True
        elif not is_pos and verdict == "Refuted": verdict_correct = True
    else:
        verdict_correct = (verdict == "Inconclusive")

    # 3. Reward Calculation
    raw_reward = 0.5
    if hallucination_detected:
        raw_reward = 0.1 # Penalty for hallucination
    elif verdict_correct:
        raw_reward = 0.9
    else:
        raw_reward = 0.2

    raw_reward = safe_strict_float(raw_reward, default=0.5)
    final_reward = (raw_reward * 0.8) + (confidence * 0.1) + 0.05
    final_reward = safe_strict_float(final_reward, default=0.5)

    return {
        "reward": final_reward,
        "hallucination_detected": hallucination_detected,
        "hallucination_check": h_check,
        "verdict_correct": verdict_correct,
        "info": "Hallucinated Context" if hallucination_detected else "Logic Validated"
    }

# --- Models ---

class Observation(BaseModel):
    mode_identifier: Literal["benchmark", "custom"]
    task_id: str
    step_number: int = 1
    max_steps: int = 15
    claim: str
    evidence_block: List[Dict[str, Any]]
    independent_var: str
    dependent_var: str
    previous_claims: List[str] = Field(default_factory=list)

class Action(BaseModel):
    verdict: Literal["Supported", "Refuted", "Inconclusive"]
    reasoning: str
    confidence: float = Field(gt=0.0, lt=1.0)
    hallucination_check: Dict[str, str]

    @validator("confidence", pre=True)
    def clamp_confidence(cls, v):
        return safe_strict_float(v)

class Reward(BaseModel):
    reward: float = Field(gt=0.0, lt=1.0)
    info: Dict[str, Any]
    done: bool = True

    @validator("reward", pre=True)
    def clamp_reward(cls, v):
        return safe_strict_float(v)

class State(BaseModel):
    current_task: Observation
    history: List[Action] = Field(default_factory=list)

# --- Environment ---

class HypothesisEnv:
    def __init__(self):
        # Use existing benchmark list or simplified
        self.benchmark_tasks = [
            {
                "id": "baseline-correlation",
                "mode": "benchmark",
                "claim": "Increased study hours lead to higher marks.",
                "dataset": [{"hours": 2, "marks": 60}, {"hours": 5, "marks": 85}],
                "independent_var": "hours",
                "dependent_var": "marks",
                "ground_truth_verdict": "Supported",
                "grader": evaluate_action
            }
        ]
        self._current_state: Optional[State] = None

    @property
    def tasks(self): return self.benchmark_tasks

    def reset(self, mode: Literal["benchmark", "custom"] = "benchmark") -> Observation:
        task_data = random.choice(self.benchmark_tasks)
        obs = Observation(
            mode_identifier=mode,
            task_id=task_data["id"],
            claim=task_data["claim"],
            evidence_block=task_data["dataset"],
            independent_var=task_data["independent_var"],
            dependent_var=task_data["dependent_var"]
        )
        self._current_state = State(current_task=obs)
        return obs

    def step(self, action: Action) -> Reward:
        if not self._current_state: raise ValueError("Reset required")
        obs = self._current_state.current_task
        task = next((t for t in self.benchmark_tasks if t["id"] == obs.task_id), self.benchmark_tasks[0])
        
        ground_truth = task.get("ground_truth_verdict") if obs.mode_identifier == "benchmark" else None
        eval_res = evaluate_action(action.model_dump(), task, ground_truth)

        # Update Action model
        if "hallucination_check" in eval_res:
            hc = eval_res["hallucination_check"]
            action.hallucination_check = {"status": hc["status"], "notes": f"Detected: {hc['values']}" if hc['detected'] else "Passed"}

        reward = Reward(reward=eval_res["reward"], info=eval_res, done=True)
        self._current_state.history.append(action)
        return reward

    def state(self) -> State:
        if not self._current_state: raise ValueError("Not initialized")
        return self._current_state