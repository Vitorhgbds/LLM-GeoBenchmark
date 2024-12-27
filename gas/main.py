from deepeval import evaluate
from deepeval.test_case import LLMTestCase

# Define your model
from gas.metrics import metrics
from gas.models.llama import Llama3_8B

# Dataset
qa = [
    ("How are igneous rocks formed?", "Igneous rocks form from the cooling and solidification of magma or lava."),
    ("List three causes of metamorphism.", "Heat, pressure, fluid activity.")
]

llama = Llama3_8B()

# Test Cases
test_cases = [
    LLMTestCase(
        input=qa[0][0],
        expected_output=qa[0][1],
        actual_output=llama.generate(qa[0][0]),
        retrieval_context=[qa[0][1]]
    ),
    LLMTestCase(
        input=qa[1][0],
        expected_output=qa[1][1],
        actual_output=llama.generate(qa[1][0]),
        retrieval_context=[qa[1][1]]
    )
]


# Evaluate
print("-----------------EVALUATION------------------")
evaluate(test_cases, metrics)