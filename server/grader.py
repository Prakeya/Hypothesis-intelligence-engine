import re

def evaluate_action(action, task):
    """
    Programmatic grader for OpenEnv.
    Scores performance on a scale of 0.0 to 1.0.
    """
    hypothesis = action.get("hypothesis", "")
    method = action.get("method", "")
    conclusion = action.get("conclusion", "")
    reasoning_steps = action.get("reasoning_steps", "")
    
    dataset = task["dataset"]
    ind_var = task["independent_var"]
    dep_var = task["dependent_var"]
    
    # 1. Programmatic Checks (Ground Truth)
    # Extract all numbers from conclusion
    numbers_in_conclusion = re.findall(r"[-+]?\d*\.\d+|\d+", conclusion)
    dataset_numbers = []
    for d in dataset:
        dataset_numbers.extend([str(v) for v in d.values()])
    
    # Hallucination Detection: Check if cited numbers exist in dataset
    hallucination_detected = False
    hallucination_reason = ""
    for num in numbers_in_conclusion:
        if num not in dataset_numbers and num not in ["0", "1", "100"]: # Allow small constants
            hallucination_detected = True
            hallucination_reason = f"Fabricated data point detected: {num}"
            break
            
    # 2. Logic Consistency (Simple Trend Analysis)
    x_vals = [d[ind_var] for d in dataset]
    y_vals = [d[dep_var] for d in dataset]
    
    # Calculate simple correlation direction
    is_positive = y_vals[-1] > y_vals[0]
    
    # Check if agent detected the 'Hard' trap (non-monotonicity)
    has_trap = False
    for i in range(len(y_vals)-1):
        if is_positive and y_vals[i+1] < y_vals[i]: has_trap = True
        if not is_positive and y_vals[i+1] > y_vals[i]: has_trap = True
        
    logic_correct = True
    if "increase" in conclusion.lower() and not is_positive: logic_correct = False
    if "decrease" in conclusion.lower() and is_positive: logic_correct = False
    if has_trap and "always" in conclusion.lower(): logic_correct = False # Missed the trap

    # 3. Scoring (Normalized to 0.0 - 1.0)
    if "error" in conclusion.lower() or "failure" in conclusion.lower() or "unknown" in conclusion.lower():
        return {"reward": 0.0, "hallucination_detected": False, "logic_correct": False, "info": "Inference Error"}

    score = 0.5 # Base score for effort
    
    if hallucination_detected:
        score -= 0.5
    else:
        if logic_correct: score += 0.3
        if len(reasoning_steps) > 50: score += 0.2 # Depth bonus
        
    # Clamp to [0, 1]
    final_score = max(0.0, min(1.0, score))
    
    return {
        "reward": final_score,
        "hallucination_detected": hallucination_detected,
        "hallucination_reason": hallucination_reason,
        "logic_correct": logic_correct,
        "breakdown": {
            "logical_consistency_score": 0.8 if logic_correct else 0.2,
            "data_usage_score": 0.0 if hallucination_detected else 1.0,
            "method_validity_score": 0.9 if len(method) > 20 else 0.3,
            "hallucination_penalty": 0.5 if hallucination_detected else 0.0
        }
    }
