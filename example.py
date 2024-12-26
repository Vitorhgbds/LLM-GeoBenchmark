from deepeval import DeepEval

# Define models to evaluate
models = [
    {"name": "Llama 3.0", "provider": "huggingface", "model_path": "llama-3.0"},
    {"name": "Gemma", "provider": "huggingface", "model_path": "gemma"}
]

# Define GPT-4o-mini as the judge
judge = {
    "name": "GPT-4o-mini",
    "provider": "openai",  # Or the relevant provider
    "model_path": "gpt-4o-mini"
}

# Initialize DeepEval
deepeval = DeepEval(models=models, judge=judge)

# Dummy evaluation test case
test_cases = [
    {"input": "What is the capital of Spain?", "expected_output": "Madrid"}
]

# Evaluate models
results = deepeval.evaluate(test_cases)
print(results)

#https://huggingface.co/meta-llama/Llama-3.2-1B