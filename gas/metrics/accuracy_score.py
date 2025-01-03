from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class ObjectiveAccuracyMetric(BaseMetric):
    def __init__(self):
        pass
    
    def measure(self, test_case: LLMTestCase):
        self.score = test_case.actual_output == test_case.expected_output
        self.success = self.score
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Objective Accuracy Metric"