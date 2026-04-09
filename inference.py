import os
import asyncio
import json
from typing import List, Optional
from openai import OpenAI
from server.env import HypothesisEnv, Action, Observation

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
def get_model_message(client: OpenAI, step: int, claim: str, evidence: List, last_reward: float):
    prompt = f"""
    Step {step} | Last Reward: {last_reward}
    Audit Target: {claim}
    Evidence Block: {evidence}
    
    Task: Conduct a high-performance logic audit.
    Output JSON with: verdict (Supported, Refuted, or Inconclusive), reasoning, and confidence_score (0.0 to 1.0).
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
            "verdict": "Inconclusive",
            "reasoning": str(e),
            "confidence_score": 0.0
        })

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = HypothesisEnv() # Local simulation for baseline
    
    MAX_STEPS = 1
    MAX_TOTAL_REWARD = 1.0
    
    # Initialize
    obs = env.reset(mode="benchmark")
    log_start(task=obs.task_id, env="hypothesis-intelligence-v4", model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        # Step Loop (One-shot audit for this env)
        for step in range(1, MAX_STEPS + 1):
            raw_action = get_model_message(client, step, obs.claim, obs.evidence_block, 0.0)
            action_data = json.loads(raw_action)
            
            # Map to Typed Action
            action = Action(
                verdict=action_data.get("verdict", "Inconclusive"),
                reasoning=action_data.get("reasoning", ""),
                confidence_score=action_data.get("confidence_score", 0.5)
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
