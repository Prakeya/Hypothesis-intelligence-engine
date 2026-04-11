import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# --- OpenEnv Variables ---
# Injected via environment settings
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("OPENAI_API_KEY", "no-key")
HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI(title="Hypothesis Intelligence Engine - Inference API")

class PredictRequest(BaseModel):
    claim: str
    dataset: List[Any]

# --- Core Existing Logic ---
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

@app.get("/")
def read_root():
    return {"status": "running"}

@app.post("/predict")
def predict(req: PredictRequest):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    raw_action = get_model_message(client, 1, req.claim, req.dataset, 0.0)
    
    try:
        action_data = json.loads(raw_action)
    except Exception as e:
        action_data = {"error": "Invalid JSON mapping", "raw": raw_action, "details": str(e)}
        
    return action_data
