from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ObjectiveAccuracyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        output = test_case.actual_output.strip().split("\n")[0].split(" ")[0].removesuffix(".")
        self.score = output == test_case.expected_output
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
        return "Objective Task Accuracy Metric"
