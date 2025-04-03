from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase

from gas.logger import Logger

logger = Logger().get_logger()


class BertSimilarityMetric(BaseMetric):
    """BERT similarity metric for evaluating the performance of a model on a specific task."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure the BERT similarity score of the model's output against the expected output.
        Args:
            test_case (LLMTestCase): The test case containing the model's output and expected output.
        Returns:
            float: The BERT similarity score of the model's output.
        """
        bert_scores = self.scorer.bert_score(
            references=test_case.expected_output,
            predictions=test_case.actual_output,
            lang="en",
        )
        logger.debug(f"bert scores: {bert_scores}")
        self.score = bert_scores["bert-f1"][0]
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase) -> float:
        """
        Async measure method for evaluating the BERT similarity score of the model's output.
        Args:
            test_case (LLMTestCase): The test case containing the model's output and expected output.

        Returns:
            float: The BERT similarity score of the model's output.
        """
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if the test case was successful based on the BERT similarity score.
        Returns:
            bool: True if the test case was successful, False otherwise.
        """
        return self.success

    @property
    def __name__(self) -> str:
        """Name of the metric.
        Returns:
            str: The name of the metric.
        """
        return "Bert Similarity Metric"
