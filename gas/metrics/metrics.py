from deepeval.metrics import AnswerRelevancyMetric, GEval, PromptAlignmentMetric, BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCaseParams

from gas.commons import GPT_JUDGE

# Subjective
## PromptAlignmentMetric -GPT
## Answer Relevancy - GPT
## Correctness - GPT
## Semantic Similarity

# Objective
## PromptAlignmentMetric -GPT
## Accuracy

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=["Reply in all uppercase"],
    model=GPT_JUDGE,
    include_reason=True
)

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=GPT_JUDGE,
    include_reason=True
)

correcness = GEval(
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

metrics = []
