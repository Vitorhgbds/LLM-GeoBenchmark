from deepeval.metrics import AnswerRelevancyMetric
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import CustomLlama32_1B

# Instantiate your custom model
custom_llm = CustomLlama32_1B()

# Define and run the metric with your custom model
metric = AnswerRelevancyMetric(model=custom_llm)
result = metric.measure(
    references=["What are the applications of AI in geology?"],
    candidates=["AI is used in geology for predictive modeling, data analysis, and exploration."]
)

print("Evaluation Result:", result)
