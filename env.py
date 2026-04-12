import random
import uuid
import math
import re
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, validator

# 🧬 Hypothesis Intelligence Engine - Gold Standard Baseline
# 12-Task Benchmark Suite | Phase 2 Compliant
# Unified Logic Core (Isomorphic)

EPS = 1e-6

# -----------------------------
# 🛠️ Core Utilities
# -----------------------------

def safe_strict_float(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x): x = float(default)
    except Exception: x = float(default)
    if x <= 0.0: return EPS
    if x >= 1.0: return 1.0 - EPS
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
    if len(values) < 2: return "Insufficient"
    is_inc = all(x < y for x, y in zip(values, values[1:]))
    is_dec = all(x > y for x, y in zip(values, values[1:]))
    if is_inc: return "Increasing"
    if is_dec: return "Decreasing"
    return "Mixed"

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
# ⚖️ Unified Grader Logic (Zero-Trust)
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
        "info": {"trend": detect_trend(y_vals), "raw_reward": raw_reward, "confidence": confidence, "ground_truth": ground_truth, "hallucination": h_check}
    }

# -----------------------------
# 🏘️ Simulation Environment
# -----------------------------

class HypothesisEnv:
    def __init__(self):
        self.tasks = [
            {"id": "study-marks", "domain": "Education", "claim": "Increased study hours lead to higher academic scores.", "dataset": [{"hours": 2, "marks": 60}, {"hours": 8, "marks": 95}], "independent_var": "hours", "dependent_var": "marks", "ground_truth_verdict": "Supported"},
            {"id": "stress-sleep", "domain": "Health", "claim": "Optimal sleep patterns are inversely correlated with cortisol levels.", "dataset": [{"sleep": 4, "cortisol": 80}, {"sleep": 9, "cortisol": 20}], "independent_var": "sleep", "dependent_var": "cortisol", "ground_truth_verdict": "Refuted"},
            {"id": "caffeine-focus", "domain": "Neuro", "claim": "Caffeine consumption exhibits a non-linear effect on task focus.", "dataset": [{"mg": 0, "focus": 50}, {"mg": 200, "focus": 90}, {"mg": 600, "focus": 30}], "independent_var": "mg", "dependent_var": "focus", "ground_truth_verdict": "Inconclusive"},
            {"id": "ads-conv", "domain": "Marketing", "claim": "Digital advertising conversions follow a diminishing returns curve.", "dataset": [{"spend": 100, "conv": 10}, {"spend": 1000, "conv": 50}, {"spend": 5000, "conv": 55}], "independent_var": "spend", "dependent_var": "conv", "ground_truth_verdict": "Supported"},
            {"id": "age-reaction", "domain": "Biology", "claim": "Cognitive processing speed decreases as a function of age.", "dataset": [{"age": 20, "ms": 200}, {"age": 60, "ms": 450}], "independent_var": "age", "dependent_var": "ms", "ground_truth_verdict": "Supported"},
            {"id": "exercise-heart", "domain": "Health", "claim": "Regular cardiovascular activity correlates with reduced RHR.", "dataset": [{"gym_min": 0, "rhr": 75}, {"gym_min": 60, "rhr": 62}], "independent_var": "gym_min", "dependent_var": "rhr", "ground_truth_verdict": "Refuted"},
            {"id": "edu-income", "domain": "Econ", "claim": "Academic attainment levels are positively predictive of earnings.", "dataset": [{"yrs": 12, "pay": 30000}, {"yrs": 20, "pay": 120000}], "independent_var": "yrs", "dependent_var": "pay", "ground_truth_verdict": "Supported"},
            {"id": "urban-temp", "domain": "Env", "claim": "Increased urban density is a significant driver of heat islands.", "dataset": [{"pop": 1000, "c": 22}, {"pop": 1000000, "c": 28}], "independent_var": "pop", "dependent_var": "c", "ground_truth_verdict": "Supported"},
            {"id": "social-happy", "domain": "Psych", "claim": "Social media saturation levels correlate with happiness indices.", "dataset": [{"hr": 1, "happy": 8}, {"hr": 8, "happy": 3}], "independent_var": "hr", "dependent_var": "happy", "ground_truth_verdict": "Refuted"},
            {"id": "price-demand", "domain": "Business", "claim": "Consumer demand for luxury goods remains inelastic across price shifts.", "dataset": [{"usd": 500, "units": 100}, {"usd": 1500, "units": 95}, {"usd": 5000, "units": 98}], "independent_var": "usd", "dependent_var": "units", "ground_truth_verdict": "Inconclusive"},
            {"id": "rain-yield", "domain": "Agri", "claim": "Increased rainfall leads to non-monotonic crop yield volatility.", "dataset": [{"mm": 100, "kg": 500}, {"mm": 500, "kg": 1200}, {"mm": 1000, "kg": 700}], "independent_var": "mm", "dependent_var": "kg", "ground_truth_verdict": "Inconclusive"},
            {"id": "fuel-dist", "domain": "Physics", "claim": "Aerodynamic drag increases exponentially with terrestrial velocity.", "dataset": [{"kph": 50, "liters": 5}, {"kph": 120, "liters": 14}], "independent_var": "kph", "dependent_var": "liters", "ground_truth_verdict": "Supported"}
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
