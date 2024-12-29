from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCaseParams

from gas.commons import GPT_JUDGE

metrics = [
    GEval(
        name="Correctness",
        model=GPT_JUDGE,
        evaluation_steps=[
            "Compare the actual output directly with the expected output to verify factual accuracy.",
            (
                "Check if all elements mentioned in the expected output are present"
                "and correctly represented in the actual output."
            ),
            (
                "Assess if there are any discrepancies"
                "in details, values, or information between the actual and expected outputs."
            ),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
    ),
    AnswerRelevancyMetric(threshold=0.7, model=GPT_JUDGE, include_reason=True),
]
