from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase

from gas.logger import Logger

logger = Logger().get_logger()


class BertSimilarityMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        bert_scores = self.scorer.bert_score(
            references=test_case.expected_output,
            predictions=test_case.actual_output,
        )
        logger.debug(f"bert scores: {bert_scores}")
        self.score = bert_scores["bert-f1"][0]
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Bert Similarity Metric"
