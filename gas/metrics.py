from deepeval.metrics import GEval, ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from gas.commons import GPT_JUDGE


correctness_metric = GEval(
    name="Correctness", 
    model=GPT_JUDGE, 
    criteria="Determine whether the actual output is factually correct based on the expected output.", 
    evaluation_steps=[
        "Compare the actual output with the expected output to check for factual correctness."
        ,"Ensure that the input data used to generate the actual output is consistent and relevant to the expected output."
        ,"Identify any discrepancies between the actual and expected output and determine if they stem from incorrect input or a processing error."
        ,"Document the findings detailing the relationship between the input, actual output, and expected output."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)

contextual_metric = ContextualPrecisionMetric(
    model=GPT_JUDGE,
    include_reason=True,
)

faithfulness_metric = FaithfulnessMetric(
    model=GPT_JUDGE,
    include_reason=True,
)

metrics = [
    correctness_metric, 
    contextual_metric, 
    faithfulness_metric
]

