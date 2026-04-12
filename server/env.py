import random
import uuid
import math
import re
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, validator

# 🧬 Hypothesis Intelligence Engine - Server Core (Sync)
# Winner-Level Suite | Phase 2 Compliant

EPS = 0.01

# -----------------------------
# 🛠️ Core Utilities
# -----------------------------

def safe_strict_float(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x): x = 0.5
    except Exception: x = 0.5
    if x < 0.1: return 0.1
    if x > 0.9: return 0.9
    return x

def extract_numeric_values(data: List[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for row in data:
        raw_val = row.get(key)
        try:
            if raw_val is not None: values.append(float(raw_val))
        except (ValueError, TypeError): continue
    return values

def detect_trend(values: List[float]) -> str:
    if not values or len(values) < 2: return "neutral"
    positive = 0
    negative = 0
    for i in range(1, len(values)):
        diff = values[i] - values[i-1]
        if diff > 0: positive += 1
        elif diff < 0: negative += 1
    total = positive + negative
    if total == 0: return "neutral"
    pos_ratio = positive / total
    neg_ratio = negative / total
    if pos_ratio >= 0.9: return "strong_positive"
    if pos_ratio >= 0.7: return "positive"
    if neg_ratio >= 0.9: return "strong_negative"
    if neg_ratio >= 0.7: return "negative"
    return "mixed"

def check_hallucination(reasoning: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean_text = re.sub(r"(Estimated Correlation|Confidence Score|r=|Reward|Score)[:\s]*[-+]?\d*\.?\d+", "", reasoning, flags=re.I)
    found_numbers = re.findall(r"[-+]?\d*\.?\d+", clean_text)
    found_floats = []
    for n in found_numbers:
        try: found_floats.append(float(n))
        except ValueError: continue

    evidence_values = {float(v) for row in evidence for v in row.values() if isinstance(v, (int, float, str)) and re.match(r"^-?\d+\.?\d*$", str(v))}
    whitelist = {0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 100.0}
    hallucinated = [f for f in found_floats if f not in evidence_values and f not in whitelist 
                    and not any(math.isclose(f, ev, rel_tol=1e-5) for ev in evidence_values)]

    if hallucinated: return {"status": "failed", "detected": True, "values": list(set(hallucinated))}
    return {"status": "passed", "detected": False, "values": []}

# -----------------------------
# 📦 Structured Models
# -----------------------------

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
    hallucination_check: Dict[str, str] = Field(default_factory=dict)

    @validator("confidence", pre=True)
    def clamp_confidence(cls, v): return safe_strict_float(v)

class Reward(BaseModel):
    reward: float = Field(gt=0.0, lt=1.0)
    info: Dict[str, Any]
    done: bool = True

    @validator("reward", pre=True)
    def clamp_reward(cls, v): return safe_strict_float(v)

class State(BaseModel):
    current_task: Observation
    history: List[Action] = Field(default_factory=list)

# -----------------------------
# ⚖️ Unified Grader Logic (Sync)
# -----------------------------

def evaluate_action(action: Dict[str, Any], task: Dict[str, Any], ground_truth=None) -> Dict[str, Any]:
    verdict = action.get("verdict", "")
    confidence = safe_strict_float(action.get("confidence", 0.5), default=0.5)
    evidence = task.get("dataset", task.get("evidence_block", []))
    dep_var = task.get("dependent_var", "y")

    y_vals = extract_numeric_values(evidence, dep_var)
    raw_reward = 0.5
    if len(y_vals) >= 2:
        is_pos = y_vals[-1] > y_vals[0]
        if is_pos and verdict == "Supported": raw_reward = 0.9
        elif (not is_pos) and verdict == "Refuted": raw_reward = 0.9
        elif verdict == "Inconclusive": raw_reward = 0.5
        else: raw_reward = 0.2
    else: raw_reward = 0.5

    h_check = check_hallucination(action.get("reasoning", ""), evidence)
    raw_reward = safe_strict_float(raw_reward, default=0.5)
    final_reward = safe_strict_float((raw_reward * 0.8) + (confidence * 0.1) + 0.05)

    return {
        "reward": final_reward,
        "hallucination_check": h_check,
        "info": {
            "trend": detect_trend(y_vals), 
            "raw_reward": raw_reward, 
            "confidence": confidence, 
            "ground_truth": ground_truth, 
            "hallucination": h_check
        }
    }

# -----------------------------
# 🏘️ Simulation Environment
# -----------------------------

class HypothesisEnv:
    def __init__(self):
        self.tasks = [
            {"id": "Health", "domain": "Health", "mode": "benchmark", "claim": "Coffee consumption reduces sleep duration.", "dataset": [{"cups": 1, "sleep": 8}, {"cups": 2, "sleep": 7.5}, {"cups": 4, "sleep": 5}, {"cups": 6, "sleep": 3}], "independent_var": "cups", "dependent_var": "sleep", "ground_truth_verdict": "Supported", "grader": evaluate_action},
            {"id": "Retail", "domain": "Retail", "mode": "benchmark", "claim": "Higher temperatures decrease umbrella sales.", "dataset": [{"temp": 15, "sales": 50}, {"temp": 25, "sales": 20}, {"temp": 35, "sales": 5}], "independent_var": "temp", "dependent_var": "sales", "ground_truth_verdict": "Supported", "grader": evaluate_action},
            {"id": "Nutrition", "domain": "Nutrition", "mode": "benchmark", "claim": "Eating more sugar leads to weight loss.", "dataset": [{"sugar": 20, "weight": 70}, {"sugar": 50, "weight": 75}, {"sugar": 100, "weight": 82}], "independent_var": "sugar", "dependent_var": "weight", "ground_truth_verdict": "Refuted", "grader": evaluate_action},
            {"id": "Finance", "domain": "Finance", "mode": "benchmark", "claim": "Higher interest rates reduce loan applications.", "dataset": [{"rate": 1, "apps": 1000}, {"rate": 5, "apps": 700}, {"rate": 10, "apps": 350}], "independent_var": "rate", "dependent_var": "apps", "ground_truth_verdict": "Supported", "grader": evaluate_action},
            {"id": "Marketing", "domain": "Marketing", "mode": "benchmark", "claim": "Increased ad spend results in lower total revenue.", "dataset": [{"spend": 100, "revenue": 5000}, {"spend": 1000, "revenue": 15000}, {"spend": 5000, "revenue": 25000}], "independent_var": "spend", "dependent_var": "revenue", "ground_truth_verdict": "Refuted", "grader": evaluate_action},
            {"id": "Psychology", "domain": "Psychology", "mode": "benchmark", "claim": "Daily meditation improves memory recall.", "dataset": [{"min": 5, "recall": 80}, {"min": 15, "recall": 85}, {"min": 30, "recall": 92}], "independent_var": "min", "dependent_var": "recall", "ground_truth_verdict": "Supported", "grader": evaluate_action},
            {"id": "Physics", "domain": "Physics", "mode": "benchmark", "claim": "Higher altitude leads to lower air pressure.", "dataset": [{"alt": 0, "pressure": 1013}, {"alt": 1000, "pressure": 898}, {"alt": 2000, "pressure": 795}], "independent_var": "alt", "dependent_var": "pressure", "ground_truth_verdict": "Supported", "grader": evaluate_action},
            {"id": "Linguistics", "domain": "Linguistics", "mode": "benchmark", "claim": "More reading reduces vocabulary.", "dataset": [{"books": 1, "vocab": 5000}, {"books": 10, "vocab": 7500}, {"books": 50, "vocab": 12000}], "independent_var": "books", "dependent_var": "vocab", "ground_truth_verdict": "Refuted", "grader": evaluate_action},
            {"id": "Education", "domain": "Education", "mode": "benchmark", "claim": "Increased study hours lead to higher academic scores.", "dataset": [{"hours": 2, "marks": 60}, {"hours": 5, "marks": 85}, {"hours": 8, "marks": 70}, {"hours": 10, "marks": 95}], "independent_var": "hours", "dependent_var": "marks", "ground_truth_verdict": "Inconclusive", "grader": evaluate_action},
            {"id": "Sports", "domain": "Sports", "mode": "benchmark", "claim": "More training always improves running speed.", "dataset": [{"training_sessions": 1, "speed": 10.4}, {"training_sessions": 2, "speed": 9.9}, {"training_sessions": 4, "speed": 10.6}, {"training_sessions": 5, "speed": 10.1}], "independent_var": "training_sessions", "dependent_var": "speed", "ground_truth_verdict": "Inconclusive", "grader": evaluate_action}
        ]
        self.benchmark_tasks = self.tasks
        self._current_state: Optional[State] = None

    def reset(self, mode: Literal["benchmark", "custom"] = "benchmark") -> Observation:
        task = random.choice(self.tasks)
        obs = Observation(
            mode_identifier=mode, task_id=task["id"], claim=task["claim"],
            evidence_block=task["dataset"], independent_var=task["independent_var"], dependent_var=task["dependent_var"]
        )
        self._current_state = State(current_task=obs)
        return obs

    def step(self, action: Action) -> Reward:
        if not self._current_state: raise ValueError("Reset required")
        obs = self._current_state.current_task
        task = next((t for t in self.tasks if t["id"] == obs.task_id), self.tasks[0])
        res = evaluate_action(action.model_dump(), task)
        hc = res["hallucination_check"]
        action.hallucination_check = {"status": hc["status"], "notes": f"Detected: {hc['values']}" if hc['detected'] else "Passed"}
        r = Reward(reward=res["reward"], info=res["info"], done=True)
        self._current_state.history.append(action)
        return r

    def state(self) -> State:
        if not self._current_state: raise ValueError("Uninitialized")
        return self._current_state