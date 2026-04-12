import random
import uuid
import json
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from server.grader import evaluate_action

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
        # Structured Benchmarking Tasks
        self.benchmark_tasks = [
            {
                "id": "baseline-correlation",
                "mode": "benchmark",
                "claim": "Increased study hours lead to higher marks.",
                "dataset": [
                    {"hours": 2, "marks": 60, "student": "A"},
                    {"hours": 5, "marks": 85, "student": "B"},
                    {"hours": 8, "marks": 70, "student": "C"},
                    {"hours": 12, "marks": 95, "student": "D"}
                ],
                "independent_var": "hours",
                "dependent_var": "marks",
                "ground_truth_verdict": "Inconclusive",
                "domain": "Education",
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
                "domain": "Health",
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
                "domain": "Retail",
                "grader": evaluate_action
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
                "ground_truth_verdict": "Refuted",
                "domain": "Nutrition",
                "grader": evaluate_action
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
                "ground_truth_verdict": "Inconclusive",
                "domain": "Sports",
                "grader": evaluate_action
            },
            {
                "id": "bench-06",
                "mode": "benchmark",
                "claim": "Higher interest rates reduce loan applications.",
                "dataset": [
                    {"rate": 2.0, "apps": 1000},
                    {"rate": 4.5, "apps": 700},
                    {"rate": 7.0, "apps": 350}
                ],
                "independent_var": "rate",
                "dependent_var": "apps",
                "ground_truth_verdict": "Supported",
                "domain": "Finance",
                "grader": evaluate_action
            },
            {
                "id": "bench-07",
                "mode": "benchmark",
                "claim": "Increased ad spend results in lower total revenue.",
                "dataset": [
                    {"ad_spend": 1000, "revenue": 5000},
                    {"ad_spend": 5000, "revenue": 15000},
                    {"ad_spend": 10000, "revenue": 25000}
                ],
                "independent_var": "ad_spend",
                "dependent_var": "revenue",
                "ground_truth_verdict": "Refuted",
                "domain": "Marketing",
                "grader": evaluate_action
            },
            {
                "id": "bench-08",
                "mode": "benchmark",
                "claim": "Daily meditation improves memory recall.",
                "dataset": [
                    {"meditation_mins": 10, "recall": 80},
                    {"meditation_mins": 20, "recall": 85},
                    {"meditation_mins": 30, "recall": 92}
                ],
                "independent_var": "meditation_mins",
                "dependent_var": "recall",
                "ground_truth_verdict": "Supported",
                "domain": "Psychology",
                "grader": evaluate_action
            },
            {
                "id": "bench-09",
                "mode": "benchmark",
                "claim": "Higher altitude leads to lower air pressure.",
                "dataset": [
                    {"altitude_m": 0, "pressure_hpa": 1013},
                    {"altitude_m": 1000, "pressure_hpa": 898},
                    {"altitude_m": 2000, "pressure_hpa": 795}
                ],
                "independent_var": "altitude_m",
                "dependent_var": "pressure_hpa",
                "ground_truth_verdict": "Supported",
                "domain": "Physics",
                "grader": evaluate_action
            },
            {
                "id": "bench-10",
                "mode": "benchmark",
                "claim": "More reading reduces vocabulary.",
                "dataset": [
                    {"books_per_month": 0, "vocab": 5000},
                    {"books_per_month": 2, "vocab": 7500},
                    {"books_per_month": 5, "vocab": 12000}
                ],
                "independent_var": "books_per_month",
                "dependent_var": "vocab",
                "ground_truth_verdict": "Refuted",
                "domain": "Linguistics",
                "grader": evaluate_action
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
            
        # Grading logic
        
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

