import re

import re

def evaluate_action(action, task, ground_truth=None):
    """
    OpenEnv Grader: Multi-mode reasoning evaluation.
    Returns reward [0.0, 0.5, 1.0].
    """
    verdict = action.get("verdict", "")
    reasoning = action.get("reasoning", "")
    
    evidence = task["evidence_block"]
    ind_var = task["independent_var"]
    dep_var = task["dependent_var"]
    
    # 1. Hallucination Check (Strict)
    numbers_in_reasoning = re.findall(r"[-+]?\d*\.\d+|\d+", reasoning)
    evidence_numbers = []
    for d in evidence:
        evidence_numbers.extend([str(v) for v in d.values()])
    
    hallucination_detected = False
    hallucinated_points = []
    for num in numbers_in_reasoning:
        # Allow common constants, indices, and sample sizes
        if num not in evidence_numbers and num not in ["0", "1", "2", "3", "4", "5", "6", "8", "10", "15", "100"]:
            hallucination_detected = True
            hallucinated_points.append(num)
            
    # 2. Verdict Verification
    verdict_correct = False
    if ground_truth:
        verdict_correct = (verdict == ground_truth)
    else:
        # Logic analysis for custom mode (basic trend check)
        y_vals = [d[dep_var] for d in evidence if dep_var in d]
        if len(y_vals) >= 2:
            is_pos = y_vals[-1] > y_vals[0]
            if is_pos and verdict == "Supported": verdict_correct = True
            if not is_pos and verdict == "Refuted": verdict_correct = True
            if y_vals[-1] == y_vals[0] and verdict == "Inconclusive": verdict_correct = True
        else:
            verdict_correct = (verdict == "Inconclusive")

    # 3. Reward Calculation
    # Correct verdict: 1.0
    # Partially correct reasoning (but correct hallucination check): 0.5
    # Incorrect verdict OR Hallucination: 0.0
    
    reward = 0.0
    if hallucination_detected:
        reward = 0.0
    elif verdict_correct:
        reward = 1.0
    elif not verdict_correct and not hallucination_detected:
        # Check if reasoning at least identified the variables correctly
        if ind_var in reasoning and dep_var in reasoning:
            reward = 0.5
            
    return {
        "reward": reward,
        "hallucination_detected": hallucination_detected,
        "hallucinated_points": hallucinated_points,
        "verdict_correct": verdict_correct,
        "logic_consistency": 1.0 if verdict_correct else 0.5,
        "info": "Hallucination Detected" if hallucination_detected else "Logic Audited"
    }

