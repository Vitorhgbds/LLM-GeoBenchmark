from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from gas.models.llama import Llama3_8B

correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o-mini",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
)


def test_llama3_8B():
    llama = Llama3_8B()
    question = "The dog chased the cat up the tree, who ran up the tree?"
    actual_output = llama.generate(question)
    assert len(actual_output) > 0
