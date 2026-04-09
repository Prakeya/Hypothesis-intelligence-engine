import random
import uuid
import json
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal

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
    confidence_score: Optional[float] = 1.0

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
        # Structured Benchmarking Tasks
        self.benchmark_tasks = [
            {
                "id": "bench-01",
                "mode": "benchmark",
                "claim": "Increased study hours lead to higher marks.",
                "dataset": [
                    {"hours": 2, "marks": 60},
                    {"hours": 5, "marks": 75},
                    {"hours": 8, "marks": 85},
                    {"hours": 10, "marks": 95}
                ],
                "independent_var": "hours",
                "dependent_var": "marks",
                "ground_truth_verdict": "Supported"
            },
            {
                "id": "bench-02",
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
                "ground_truth_verdict": "Supported"
            },
            {
                "id": "bench-03",
                "mode": "benchmark",
                "claim": "Higher temperatures decrease umbrella sales.",
                "dataset": [
                    {"temp": 15, "sales": 50},
                    {"temp": 25, "sales": 20},
                    {"temp": 35, "sales": 5}
                ],
                "independent_var": "temp",
                "dependent_var": "sales",
                "ground_truth_verdict": "Supported"
            },
            {
                "id": "bench-04",
                "mode": "benchmark",
                "claim": "Eating more sugar leads to weight loss.",
                "dataset": [
                    {"sugar_g": 10, "weight": 70},
                    {"sugar_g": 50, "weight": 75},
                    {"sugar_g": 100, "weight": 82}
                ],
                "independent_var": "sugar_g",
                "dependent_var": "weight",
                "ground_truth_verdict": "Refuted"
            },
            {
                "id": "bench-05",
                "mode": "benchmark",
                "claim": "Wearing blue shoes improves running speed.",
                "dataset": [
                    {"color": "blue", "speed": 10},
                    {"color": "red", "speed": 11},
                    {"color": "blue", "speed": 9.5}
                ],
                "independent_var": "color",
                "dependent_var": "speed",
                "ground_truth_verdict": "Inconclusive"
            }
        ]
        self._current_state: Optional[State] = None

    @property
    def tasks(self):
        # To maintain compatibility with existing UI that uses .tasks
        return self.benchmark_tasks

    def reset(self, mode: Literal["benchmark", "custom"] = "benchmark", custom_data: Optional[Dict[str, Any]] = None) -> Observation:
        """Resets the environment and returns the initial observation."""
        if mode == "benchmark":
            task_data = random.choice(self.benchmark_tasks)
        else:
            if not custom_data:
                # Default custom if none provided
                task_data = {
                    "id": "custom-" + str(uuid.uuid4())[:8],
                    "claim": "Custom claim",
                    "dataset": [],
                    "independent_var": "x",
                    "dependent_var": "y",
                    "ground_truth_verdict": "Inconclusive"
                }
            else:
                task_data = custom_data
                task_data["id"] = "custom-" + str(uuid.uuid4())[:8]
                task_data["ground_truth_verdict"] = "Inconclusive" # Custom mode is open-ended

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
        """Executes a logical step (audit) and returns the reward."""
        if not self._current_state:
            raise ValueError("Environment must be reset before calling step().")
            
        from server.grader import evaluate_action
        
        # Find ground truth for benchmarking if applicable
        ground_truth = "Inconclusive"
        if self._current_state.current_task.mode_identifier == "benchmark":
            task = next((t for t in self.benchmark_tasks if t["id"] == self._current_state.current_task.task_id), None)
            if task:
                ground_truth = task["ground_truth_verdict"]

        # Grading logic
        eval_res = evaluate_action(action.dict(), self._current_state.current_task.dict(), ground_truth)
        
        reward = Reward(
            reward=eval_res["reward"],
            info=eval_res,
            done=True
        )
        
        self._current_state.history.append(action)
        return reward

    def state(self) -> State:
        """Returns the current internal state."""
        if not self._current_state:
            raise ValueError("Environment has not been initialized.")
        return self._current_state

