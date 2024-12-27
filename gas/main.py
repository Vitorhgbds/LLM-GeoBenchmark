from deepeval import evaluate
from deepeval.test_case import LLMTestCase

# Define your model
from gas.database.geobenchmark_provider import GeobenchProvider
from gas.metrics import metrics
from gas.models.llama import Llama3_8B

provider = GeobenchProvider()
llama = Llama3_8B()
test_cases = provider.fetch_evaluation_dataset(llama, GeobenchProvider.BanchmarkType.NOUN, limit=5)
evaluate(test_cases, metrics)