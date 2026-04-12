import random
import uuid
import math
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, validator

EPS = 1e-6


# -----------------------------
# Safe Clamp
# -----------------------------

def safe_strict_float(value: Any, default: float = 0.5) -> float:
    """
    Convert to float and force strict (0,1) bounds.
    Handles NaN/Inf by reverting to default.
    """
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            x = float(default)
    except Exception:
        x = float(default)

    # Hard clamp to strict open interval (0, 1)
    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1.0 - EPS
    return x


# -----------------------------
# Typed Models
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


# -----------------------------
# Grader Logic
# -----------------------------

def evaluate_action(action: Dict[str, Any], task: Dict[str, Any], ground_truth=None) -> Dict[str, Any]:
    """
    Strictly returns reward in the open interval (0,1).
    """
    verdict = action.get("verdict", "")
    confidence = safe_strict_float(action.get("confidence", 0.5), default=0.5)

    evidence = task.get("dataset", [])
    dep_var = task.get("dependent_var", "y")

    y_vals = []
    pathology = []

    for i, row in enumerate(evidence):
        if dep_var not in row:
            pathology.append({"index": i, "error": "missing_key"})
            continue

        v = row[dep_var]

        if v is None:
            pathology.append({"index": i, "error": "None_value"})
        elif isinstance(v, bool):
            pathology.append({"index": i, "error": "bool_value", "value": v})
        elif isinstance(v, (int, float)):
            y_vals.append(float(v))
        elif isinstance(v, str):
            try:
                y_vals.append(float(v))
                pathology.append({"index": i, "error": "string_cast", "value": v})
            except Exception:
                pathology.append({"index": i, "error": "bad_string", "value": v})
        else:
            pathology.append({"index": i, "error": "unsupported_type", "type": type(v).__name__})

    raw_reward = 0.5

    if len(y_vals) >= 2:
        is_pos = y_vals[-1] > y_vals[0]

        if is_pos and verdict == "Supported":
            raw_reward = 0.9
        elif (not is_pos) and verdict == "Refuted":
            raw_reward = 0.9
        elif verdict == "Inconclusive":
            raw_reward = 0.5
        else:
            raw_reward = 0.2
    else:
        raw_reward = 0.5

    raw_reward = safe_strict_float(raw_reward, default=0.5)

    final_reward = (raw_reward * 0.8) + (confidence * 0.1) + 0.05
    final_reward = safe_strict_float(final_reward, default=0.5)

    print("[REWARD_TRACE] verdict:", verdict)
    print("[REWARD_TRACE] confidence:", confidence)
    print("[REWARD_TRACE] dep_var:", dep_var)
    print("[REWARD_TRACE] y_vals:", y_vals)
    print("[REWARD_TRACE] raw_reward:", raw_reward)
    print("[REWARD_TRACE] final_reward:", final_reward)
    if pathology:
        print("[REWARD_TRACE] pathology:", pathology)

    return {
        "reward": final_reward,
        "info": {
            "grader": "strictly_between_0_and_1",
            "raw_reward_before_blend": raw_reward,
            "confidence": confidence,
            "dependent_var": dep_var,
            "y_vals": y_vals,
            "pathology": pathology,
            "ground_truth": ground_truth
        }
    }


# -----------------------------
# Environment
# -----------------------------

class HypothesisEnv:
    def __init__(self):
        self.tasks = [
            {
                "id": "baseline-correlation",
                "difficulty": "easy",
                "mode": "benchmark",
                "claim": "More study hours improve marks.",
                "dataset": [
                    {"hours": 2, "marks": 60},
                    {"hours": 5, "marks": 75},
                    {"hours": 8, "marks": 85},
                    {"hours": 10, "marks": 95}
                ],
                "independent_var": "hours",
                "dependent_var": "marks",
                "ground_truth_verdict": "Supported",
                "grader": evaluate_action
            },
            {
                "id": "nonlinear-dependency",
                "difficulty": "medium",
                "mode": "benchmark",
                "claim": "Higher caffeine intake leads to less sleep.",
                "dataset": [
                    {"cups": 0, "sleep": 8.5},
                    {"cups": 1, "sleep": 8},
                    {"cups": 2, "sleep": 7.5},
                    {"cups": 4, "sleep": 5.5},
                    {"cups": 6, "sleep": 4}
                ],
                "independent_var": "cups",
                "dependent_var": "sleep",
                "ground_truth_verdict": "Supported",
                "grader": evaluate_action
            },
            {
                "id": "confounding-variables",
                "difficulty": "hard",
                "mode": "benchmark",
                "claim": "Increased rainfall always leads to higher crop yield.",
                "dataset": [
                    {"rainfall": 100, "yield": 5},
                    {"rainfall": 200, "yield": 8},
                    {"rainfall": 500, "yield": 12},
                    {"rainfall": 800, "yield": 9},
                    {"rainfall": 1000, "yield": 6}
                ],
                "independent_var": "rainfall",
                "dependent_var": "yield",
                "ground_truth_verdict": "Inconclusive",
                "grader": evaluate_action
            }
        ]

        self.benchmark_tasks = self.tasks
        self._current_state: Optional[State] = None

    def reset(self, mode: Literal["benchmark", "custom"] = "benchmark") -> Observation:
        task = random.choice(self.tasks)

        obs = Observation(
            mode_identifier=mode,
            task_id=task["id"],
            claim=task["claim"],
            evidence_block=task["dataset"],
            independent_var=task["independent_var"],
            dependent_var=task["dependent_var"]
        )

        self._current_state = State(current_task=obs)
        return obs

    def step(self, action: Action) -> Reward:
        if not self._current_state:
            raise ValueError("Env not initialized. Call /reset first.")

        obs = self._current_state.current_task

        print("[DEBUG_PRE] task_id:", obs.task_id)
        print("[DEBUG_PRE] dependent_var:", obs.dependent_var)
        print("[DEBUG_PRE] evidence_block:", obs.evidence_block)

        task = next(
            t for t in self.tasks
            if t["id"] == obs.task_id
        )

        ground_truth = None
        if obs.mode_identifier == "benchmark":
            gt_task = next((t for t in self.benchmark_tasks if t["id"] == obs.task_id), None)
            if gt_task:
                ground_truth = gt_task.get("ground_truth_verdict")

        eval_res = evaluate_action(action.model_dump(), task, ground_truth=ground_truth)

        reward_value = safe_strict_float(eval_res["reward"], default=0.5)

        print("[DEBUG_POST] reward before storage:", repr(reward_value))
        print("[DEBUG_POST] info:", eval_res["info"])

        # Strict Assertion
        assert 0.0 < reward_value < 1.0, f"Reward {reward_value} out of range (0,1)"

        reward = Reward(
            reward=reward_value,
            info=eval_res["info"],
            done=True
        )

        self._current_state.history.append(action)
        return reward

    def state(self) -> State:
        if not self._current_state:
            raise ValueError("Env not initialized")
        return self._current_state
