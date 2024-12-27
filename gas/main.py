from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase

from gas.models.llama import Llama3_8B

correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o-mini",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

llama = Llama3_8B()
question = "The dog chased the cat up the tree, who ran up the tree?"
actual_output = llama.generate(question)
expected_output = "The cat."

test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    expected_output=expected_output
)

print("-------------------- Correctness Metric --------------------")
print(f"Question: {test_case.input}")
print(f"Actual Output: {test_case.actual_output}")
print(f"Expected Output: {test_case.expected_output}")
#correctness_metric.measure(test_case)
#print(correctness_metric.score)
#print(correctness_metric.reason)