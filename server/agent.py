import random
import re
from server.inference import run_inference, parse_inference_output

class HypothesisAgent:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm

    def generate_action(self, state, audit_id="SYSTEM"):
        """
        Generates an action based on the current state.
        Now supports audit_id for structured logging.
        """
        claim = state["claim"]
        dataset = state["dataset"]
        
        if self.use_llm:
            return self._generate_llm_action(claim, dataset, audit_id)
        else:
            return self._generate_rule_based_action(claim, dataset)

    def _generate_rule_based_action(self, claim, dataset):
        """Standard rule-based agent logic."""
        keys = list(dataset[0].keys())
        x_var = keys[0]
        y_var = keys[1]
        
        sorted_data = sorted(dataset, key=lambda d: d[x_var])
        y_values = [d[y_var] for d in sorted_data]
        is_increasing = all(y_values[i] <= y_values[i+1] for i in range(len(y_values)-1))
        
        hypothesis = f"There is a {'positive' if is_increasing else 'complex'} relationship between {x_var} and {y_var}."
        method = f"I will analyze the trend of {y_var} as {x_var} increases using the provided {len(dataset)} data points."
        reasoning_steps = (
            f"1. Identified independent variable: {x_var} and dependent variable: {y_var}.\n"
            f"2. Observed {len(dataset)} data points in the provided set.\n"
            f"3. Compared endpoints and intermediate monotonic trends."
        )
        
        if is_increasing:
            conclusion = f"The data shows a consistent positive trend. As {x_var} increases, {y_var} also increases. For example, when {x_var} is {sorted_data[0][x_var]}, {y_var} is {sorted_data[0][y_var]}."
        else:
            conclusion = f"The data does not show a strictly monotonic relationship. While {x_var} increases, {y_var} fluctuates."
            
        return {
            "hypothesis": hypothesis,
            "method": method,
            "reasoning_steps": reasoning_steps,
            "conclusion": conclusion
        }

    def _generate_llm_action(self, claim, dataset, audit_id):
        """Delegates LLM call to the centralized inference module."""
        prompt = f"""
        Task: Analyze the following claim based on the provided dataset.
        Claim: {claim}
        Dataset: {dataset}
        
        Output format:
        Hypothesis: <your hypothesis>
        Method: <your method>
        Conclusion: <your conclusion>
        
        Constraints:
        1. Be accurate.
        2. Only use numbers from the dataset.
        3. Do not make absolute claims if the data has even one exception.
        """
        
        raw_output = run_inference(prompt, audit_id)
        if raw_output:
            return parse_inference_output(raw_output)
        else:
            return {
                "hypothesis": "Error", 
                "method": "Error", 
                "reasoning_steps": "Inference failed or timed out.", 
                "conclusion": "Audit aborted due to technical failure."
            }

class HallucinatingAgent(HypothesisAgent):
    def _generate_rule_based_action(self, claim, dataset):
        """Agent designed to trigger hallucinations for testing."""
        return {
            "hypothesis": "I believe that the correlation is exactly 0.99 and guaranteed to be true.",
            "method": "I will use advanced AI magic to find hidden numbers like 555.5 and 999.",
            "reasoning_steps": "1. Used occult powers to bypass the dataset.\n2. Identified invisible numbers 555.5 and 999.",
            "conclusion": "The data definitively proves that the trend always increases without exception, and the secret value is 420."
        }
