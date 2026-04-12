import random
import uuid
import json
import os
import re
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal

# --- OpenEnv Grader Logic (Embedded for Validation Discovery) ---

def evaluate_action(action, task, ground_truth=None):
    """
    OpenEnv Grader: Multi-mode reasoning evaluation.
    Returns reward [0.0, 0.5, 1.0].
    """
    verdict = action.get("verdict", "")
    reasoning = action.get("reasoning", "")
    
    evidence = task.get("evidence_block", task.get("dataset", []))
    ind_var = task.get("independent_var", "")
    dep_var = task["dependent_var"]
    
    # 1. Hallucination Check (Strict)
    clean_reasoning = re.sub(r"(Estimated Correlation \(r\): |Confidence Score: |r=)[-+]?\d*\.\d+", "", reasoning)
    numbers_in_reasoning = re.findall(r"[-+]?\d*\.\d+|\d+", clean_reasoning)
    evidence_numbers = []
    for d in evidence:
        evidence_numbers.extend([str(v) for v in d.values()])
    
    hallucination_detected = False
    hallucinated_points = []
    for num in numbers_in_reasoning:
        if num not in evidence_numbers and num not in ["0", "1", "2", "3", "4", "5", "6", "8", "10", "15", "100"]:
            hallucination_detected = True
            hallucinated_points.append(num)
            
    # 2. Verdict Verification
    verdict_correct = False
    if ground_truth:
        verdict_correct = (verdict == ground_truth)
    else:
        y_vals = [d[dep_var] for d in evidence if dep_var in d]
        if len(y_vals) >= 2:
            is_pos = y_vals[-1] > y_vals[0]
            if is_pos and verdict == "Supported": verdict_correct = True
            if not is_pos and verdict == "Refuted": verdict_correct = True
            if y_vals[-1] == y_vals[0] and verdict == "Inconclusive": verdict_correct = True
        else:
            verdict_correct = (verdict == "Inconclusive")

    # 3. Reward Calculation
    reward = 0.0
    breakdown = []
    
    if hallucination_detected:
        reward = 0.0
        breakdown.append({"metric": "Hallucination Check", "status": "FAIL", "points": "0.0", "reason": "Fabricated numbers not found in evidence."})
    elif verdict_correct:
        reward = 1.0
        breakdown.append({"metric": "Hallucination Check", "status": "PASS", "points": "+0.2", "reason": "No illegal numeric hallucinations found."})
        breakdown.append({"metric": "Verdict Accuracy", "status": "PASS", "points": "+0.5", "reason": "Predicted verdict matches empirical data."})
    elif not verdict_correct and not hallucination_detected:
        if ind_var in reasoning and dep_var in reasoning:
            reward = 0.5
            breakdown.append({"metric": "Logic Baseline", "status": "PASS", "points": "+0.3", "reason": "Variables tracked correctly."})
            
    return {
        "reward": reward,
        "hallucination_detected": hallucination_detected,
        "verdict_correct": verdict_correct,
        "logic_consistency": 1.0 if verdict_correct else 0.5,
        "breakdown": breakdown
    }

# --- OpenEnv Typed Models ---

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
    confidence: float
    hallucination_check: Dict[str, str]

class Reward(BaseModel):
    reward: float
    info: Dict[str, Any]
    done: bool = True

class State(BaseModel):
    current_task: Observation
    history: List[Action] = Field(default_factory=list)

# --- Environment Implementation ---

class HypothesisEnv:
    def __init__(self):
        # Aligned with openenv.yaml requirements for Phase 2
        self.benchmark_tasks = [
            {
                "id": "baseline-correlation",
                "mode": "benchmark",
                "claim": "Increased study hours lead to higher marks.",
                "dataset": [
                    {"hours": 2, "marks": 60},
                    {"hours": 5, "marks": 85},
                    {"hours": 8, "marks": 70},
                    {"hours": 12, "marks": 95}
                ],
                "independent_var": "hours",
                "dependent_var": "marks",
                "ground_truth_verdict": "Inconclusive",
                "grader": evaluate_action
            },
            {
                "id": "nonlinear-dependency",
                "mode": "benchmark",
                "claim": "Coffee consumption reduces sleep duration.",
                "dataset": [
                    {"cups": 0, "sleep": 8},
                    {"cups": 1, "sleep": 7.5},
                    {"cups": 4, "sleep": 5},
                    {"cups": 8, "sleep": 3}
                ],
                "independent_var": "cups",
                "dependent_var": "sleep",
                "ground_truth_verdict": "Supported",
                "grader": evaluate_action
            },
            {
                "id": "confounding-variables",
                "mode": "benchmark",
                "claim": "Higher temperatures decrease umbrella sales.",
                "dataset": [
                    {"temp": 15, "sales": 50},
                    {"temp": 25, "sales": 20},
                    {"temp": 35, "sales": 5}
                ],
                "independent_var": "temp",
                "dependent_var": "sales",
                "ground_truth_verdict": "Supported",
                "grader": evaluate_action
            }
        ]
        self._current_state: Optional[State] = None

    @property
    def tasks(self):
        return self.benchmark_tasks

    def reset(self, mode: Literal["benchmark", "custom"] = "benchmark", custom_data: Optional[Dict[str, Any]] = None) -> Observation:
        if mode == "benchmark":
            task_data = random.choice(self.benchmark_tasks)
        else:
            task_data = custom_data if custom_data else self.benchmark_tasks[0]
            task_data["id"] = "custom-" + str(uuid.uuid4())[:8]

        obs = Observation(
            mode_identifier=mode,
            task_id=task_data["id"],
            claim=task_data["claim"],
            evidence_block=task_data.get("dataset", []),
            independent_var=task_data["independent_var"],
            dependent_var=task_data["dependent_var"]
        )
        self._current_state = State(current_task=obs)
        return obs

    def step(self, action: Action) -> Reward:
        if not self._current_state:
            raise ValueError("Environment must be reset before calling step().")
            
        ground_truth = "Inconclusive"
        if self._current_state.current_task.mode_identifier == "benchmark":
            task = next((t for t in self.benchmark_tasks if t["id"] == self._current_state.current_task.task_id), None)
            if task:
                ground_truth = task["ground_truth_verdict"]

        eval_res = evaluate_action(action.dict(), self._current_state.current_task.dict(), ground_truth)
        
        return Reward(
            reward=eval_res["reward"],
            info=eval_res,
            done=True
        )

    def state(self) -> State:
        if not self._current_state:
            raise ValueError("Environment has not been initialized.")
        return self._current_state
