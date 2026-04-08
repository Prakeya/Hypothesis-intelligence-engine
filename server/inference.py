import os
import asyncio
import json
from typing import List, Optional
from openai import OpenAI
from env import HypothesisEnv, Action, Observation

# --- OpenEnv Variables ---
# Injected via environment settings
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("OPENAI_API_KEY", "no-key")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Logging Helpers (Strict OpenEnv Format) ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}")

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    print(f"[STEP] step={step} action={action} reward={reward} done={done} error={error}")

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}")

# --- Inference Logic ---
def get_model_message(client: OpenAI, step: int, claim: str, dataset: List, last_reward: float):
    prompt = f"""
    Step {step} | Last Reward: {last_reward}
    Audit Target: {claim}
    Data Artifacts: {dataset}
    
    Task: Conduct a high-performance logic audit.
    Output JSON with: hypothesis, method, reasoning_steps, conclusion.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}")
        return json.dumps({
            "hypothesis": "Error",
            "method": "Error",
            "reasoning_steps": str(e),
            "conclusion": "Audit failure."
        })

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = HypothesisEnv() # Local simulation for baseline
    
    MAX_STEPS = 1
    MAX_TOTAL_REWARD = 1.0
    
    # Initialize
    obs = env.reset()
    log_start(task=obs.task_id, env="hypothesis-intelligence-v4", model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        # Step Loop (One-shot audit for this env)
        for step in range(1, MAX_STEPS + 1):
            raw_action = get_model_message(client, step, obs.claim, obs.dataset, 0.0)
            action_data = json.loads(raw_action)
            
            # Map to Typed Action
            action = Action(
                hypothesis=action_data.get("hypothesis", ""),
                method=action_data.get("method", ""),
                reasoning_steps=action_data.get("reasoning_steps", ""),
                conclusion=action_data.get("conclusion", "")
            )
            
            reward_obj = env.step(action)
            reward = reward_obj.reward
            done = reward_obj.done
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=raw_action[:100], reward=reward, done=done)
            
            if done: break
            
        final_score = sum(rewards) / MAX_TOTAL_REWARD
        final_score = min(max(final_score, 0.0), 1.0)
        success = final_score >= 0.7
        
    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
