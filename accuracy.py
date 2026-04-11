import json

# load test cases
with open("test_cases.json", "r") as f:
    data = json.load(f)

total = len(data)
score = 0

for case in data:
    expected = case["expected_verdict"]
    predicted = case["model_output"]["verdict"]

    if predicted == expected:
        score += 1
    elif predicted == "Inconclusive" and expected in ["Supported", "Refuted"]:
        score += 0.5
    else:
        score += 0

accuracy = score / total

print("Total cases:", total)
print("Accuracy:", accuracy)