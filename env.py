import random
import uuid
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# -----------------------------
# Typed Models
# -----------------------------

class Observation(BaseModel):
    task_id: str
    step_number: int = 1
    max_steps: int = 15
    claim: str
    dataset: List[Dict[str, Any]]
    independent_var: str
    dependent_var: str
    previous_claims: List[str] = Field(default_factory=list)


class Action(BaseModel):
    hypothesis: str
    method: str
    reasoning_steps: str
    conclusion: str


class Reward(BaseModel):
    reward: float
    info: Dict[str, Any]
    done: bool = True


class State(BaseModel):
    current_task: Observation
    history: List[Action] = Field(default_factory=list)


# -----------------------------
# YOUR ORIGINAL GRADER LOGIC (UNCHANGED)
# -----------------------------

def evaluate_action(action, task, ground_truth=None):
    """
    KEEPING YOUR LOGIC EXACTLY AS-IS (no changes)
    """
    verdict = action.get("verdict", "")
    reasoning = action.get("reasoning", "")

    evidence = task.get("dataset", [])
    dep_var = task["dependent_var"]

    y_vals = [d[dep_var] for d in evidence if dep_var in d]

    if len(y_vals) >= 2:
        is_pos = y_vals[-1] > y_vals[0]
        if is_pos and verdict == "Supported":
            reward = 1.0
        elif not is_pos and verdict == "Refuted":
            reward = 1.0
        else:
            reward = 0.5
    else:
        reward = 0.5

    return {
        "reward": reward,
        "info": {"grader": "original_logic"}
    }


# -----------------------------
# ENVIRONMENT
# -----------------------------

class HypothesisEnv:
    def __init__(self):

        # ✅ KEEP YOUR ORIGINAL TASKS UNCHANGED
        self.benchmark_tasks = [
            {
                "id": "easy-01",
                "difficulty": "easy",
                "claim": "More study hours improve marks.",
                "dataset": [
                    {"hours": 2, "marks": 60},
                    {"hours": 5, "marks": 75},
                    {"hours": 8, "marks": 85},
                    {"hours": 10, "marks": 95}
                ],
                "independent_var": "hours",
                "dependent_var": "marks",
                "grader": evaluate_action
            },
            {
                "id": "medium-01",
                "difficulty": "medium",
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
                "grader": evaluate_action
            },
            {
                "id": "hard-01",
                "difficulty": "hard",
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
                "grader": evaluate_action
            }
        ]

        # ⭐ CRITICAL FIX: make tasks directly visible to validator
        self.tasks = self.benchmark_tasks

        self._current_state: Optional[State] = None

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self) -> Observation:
        task = random.choice(self.tasks)

        obs = Observation(
            task_id=task["id"],
            claim=task["claim"],
            dataset=task["dataset"],
            independent_var=task["independent_var"],
            dependent_var=task["dependent_var"]
        )

        self._current_state = State(current_task=obs)
        return obs

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action: Action) -> Reward:
        if not self._current_state:
            raise ValueError("Env not initialized")

        task = next(
            t for t in self.tasks
            if t["id"] == self._current_state.current_task.task_id
        )

        eval_res = task["grader"](action.dict(), task)

        return Reward(
            reward=eval_res["reward"],
            info=eval_res,
            done=True
        )

    # -----------------------------
    # STATE
    # -----------------------------
    def state(self) -> State:
        if not self._current_state:
            raise ValueError("Env not initialized")
        return self._current_state