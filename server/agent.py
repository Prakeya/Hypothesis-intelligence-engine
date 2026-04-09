import random
import re

class HypothesisAgent:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm

    def generate_action(self, state, audit_id="SYSTEM"):
        """
        Generates an action based on the current state.
        Uses robust multi-step reasoning to produce a verdict.
        """
        claim = state["claim"]
        evidence = state["evidence_block"]
        ind_var = state["independent_var"]
        dep_var = state["dependent_var"]
        
        # 1. Signal Extraction & Hallucination Check
        x_values = [d.get(ind_var) for d in evidence if ind_var in d]
        y_values = [d.get(dep_var) for d in evidence if dep_var in d]
        
        reasoning = "Step 1: Signal Extraction\n"
        reasoning += f"- Identified independent variable: '{ind_var}' and dependent variable: '{dep_var}'.\n"
        reasoning += f"- Extracted {len(evidence)} valid data points from the evidence block.\n"
        
        if not x_values or not y_values:
            return {
                "verdict": "Inconclusive",
                "reasoning": reasoning + "Step 2: Analysis\n- Insufficient data points to perform trend analysis.",
                "confidence_score": 0.1
            }

        # 2. Contradiction Detection & Trend Analysis
        # Sort by x for trend analysis
        paired = sorted(zip(x_values, y_values))
        sorted_x, sorted_y = zip(*paired)
        
        is_increasing = all(sorted_y[i] <= sorted_y[i+1] for i in range(len(sorted_y)-1))
        is_decreasing = all(sorted_y[i] >= sorted_y[i+1] for i in range(len(sorted_y)-1))
        
        reasoning += "Step 2: Trend Analysis\n"
        if is_increasing:
            reasoning += f"- Detected a consistent positive trend: {dep_var} increases with {ind_var}.\n"
        elif is_decreasing:
            reasoning += f"- Detected a consistent negative trend: {dep_var} decreases as {ind_var} increases.\n"
        else:
            reasoning += f"- Detected a non-monotonic or complex relationship between {ind_var} and {dep_var}.\n"

        # 3. Final Synthesis & Verdict
        reasoning += "Step 3: Logical Synthesis\n"
        claim_lower = claim.lower()
        
        pos_terms = ["increase", "improve", "higher", "more", "positive", "growth"]
        neg_terms = ["decrease", "reduce", "lower", "less", "negative", "loss", "reduction"]
        
        # Determine claim direction based on the last movement term (often the outcome)
        # and checking for common structures like "X leads to Y"
        all_terms = []
        for t in pos_terms:
            idx = claim_lower.find(t)
            if idx != -1: all_terms.append((idx, 1))
        for t in neg_terms:
            idx = claim_lower.find(t)
            if idx != -1: all_terms.append((idx, -1))
            
        all_terms.sort()
        
        if not all_terms:
            claim_direction = 1 # Assume positive by default if no terms found
        else:
            # If multiple terms, look at the one that appears later (usually the Y in "X leads to Y")
            claim_direction = all_terms[-1][1]
        
        if (claim_direction == 1 and is_increasing) or (claim_direction == -1 and is_decreasing):
            verdict = "Supported"
            reasoning += f"- The claim predicts a trend that aligns perfectly with the empirical evidence."
        elif (claim_direction == 1 and is_decreasing) or (claim_direction == -1 and is_increasing):
            verdict = "Refuted"
            reasoning += f"- The claim predicts a trend that is directly contradicted by the empirical evidence."
        else:
            if len(evidence) < 3:
                verdict = "Inconclusive"
                reasoning += "- The sample size is too small to definitively support or refute the claim."
            else:
                verdict = "Inconclusive"
                reasoning += "- The data shows a relationship that does not clearly support or refute the simplified claim."


        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "confidence_score": 0.9 if verdict != "Inconclusive" else 0.4
        }
