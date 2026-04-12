import random
import uuid
import json
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# --- OpenEnv Typed Models ---

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

# --- Environment Implementation ---

class HypothesisEnv:
    def __init__(self):
        # 3 Structured Task Levels as per OpenEnv requirement
        self.tasks = [
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
                "dependent_var": "marks"
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
                "dependent_var": "sleep"
            },
            {
                "id": "hard-01",
                "difficulty": "hard",
                "claim": "Increased rainfall always leads to higher crop yield.",
                "dataset": [
                    {"rainfall": 100, "yield": 5},
                    {"rainfall": 200, "yield": 8},
                    {"rainfall": 500, "yield": 12},
                    {"rainfall": 800, "yield": 9},  # The "Trap": Excessive rain reduces yield
                    {"rainfall": 1000, "yield": 6}
                ],
                "independent_var": "rainfall",
                "dependent_var": "yield"
            }
        ]
        self._current_state: Optional[State] = None

    def reset(self) -> Observation:
        """Resets the environment and returns the initial observation."""
        task_data = random.choice(self.tasks)
        obs = Observation(
            task_id=task_data["id"],
            claim=task_data["claim"],
            dataset=task_data["dataset"],
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
        
        # Grading logic compliant with OpenEnv (0.0 - 1.0)
        eval_res = evaluate_action(action.dict(), self._current_state.current_task.dict())
        
        # Apply OpenEnv blueprint reward normalization
        normalized_reward = eval_res["reward"]
        
        reward = Reward(
            reward=normalized_reward,
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
