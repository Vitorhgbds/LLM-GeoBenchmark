from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM

from gas.models.gpt import GPT4oMini
from gas.models.llama import Llama3_8B


class JudgeEvaluationMetric(BaseMetric):
    def __init__(self, judge_llm, threshold=0.7):
        """
        Initialize the custom metric.
        :param judge_llm: An LLM instance used as the judge.
        :param threshold: Minimum score for passing the test.
        """
        self.judge_llm: DeepEvalBaseLLM = judge_llm
        self.threshold = threshold

    def evaluate(self, test_case: LLMTestCase) -> bool:
        """
        Evaluate the test case using the judge LLM.
        :param test_case: The test case to evaluate.
        :return: True if the score meets the threshold, False otherwise.
        """
        # Generate the judgment prompt
        judgment_prompt = (
            f"Evaluate the following response based on its relevance and accuracy to the question:\n\n"
            f"Question: {test_case.input}\n"
            f"Response: {test_case.actual_output}\n"
            f"Context: {' '.join(test_case.retrieval_context)}\n\n"
            f"Score the response on a scale of 0 to 1 (1 being highly relevant and accurate):"
        )

        # Use the judge LLM to evaluate
        score = float(self.judge_llm.generate(judgment_prompt).strip())
        return score >= self.threshold


def test_llama3_with_gpt4o_judge():
    # Initialize your models
    llama3_8b = Llama3_8B()  # Replace this with the appropriate initialization for Llama3_8B
    gpt4o_mini_judge = GPT4oMini()  # Provide your OpenAI API key for GPT4oMini

    # Define a test case
    test_case = LLMTestCase(
        input="What is the process of plate tectonics?",
        # Replace with the actual output of the Llama3_8B model
        actual_output=llama3_8b.generate("What is the process of plate tectonics?"),
        retrieval_context=[
            "Plate tectonics is the movement of large plates that make up the Earth's surface.",
            "These plates float on the molten layer beneath the Earth's crust and move due to convection currents."
        ]
    )

    # Use the GPT4oMini as a judge
    judge_metric = JudgeEvaluationMetric(
        judge_llm=gpt4o_mini_judge, 
        threshold=0.7  # Define a threshold for acceptable outputs
    )

    # Run the evaluation
    assert_test(test_case, [judge_metric])
