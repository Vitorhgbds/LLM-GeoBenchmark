# Define your model
from gas.database.geobenchmark_provider import GeobenchProvider
from gas.models.gemma import Gemma_9B

provider = GeobenchProvider()
# model = Llama3_8B()
# model = Ministral_8B()
model = Gemma_9B()
test_cases = provider.fetch_evaluation_dataset(model, GeobenchProvider.BanchmarkType.NOUN, limit=1)

for test_case in test_cases.test_cases:
    print(f"Question: {test_case.input}")
    print(f"Expected Output: {test_case.expected_output}")
    print(f"Actual Output: {test_case.actual_output}")
    print("-------------------------------------------------")
# evaluate(test_cases, metrics)
