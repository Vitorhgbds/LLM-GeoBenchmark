# Define your model
from gas.database.geobenchmark_provider import GeobenchProvider
from gas.models.llama import Llama3_8B

provider = GeobenchProvider()
llama = Llama3_8B()
test_cases = provider.fetch_evaluation_dataset(llama, GeobenchProvider.BanchmarkType.NOUN, limit=1)

for test_case in test_cases.test_cases:
    print(f"Question: {test_case.input}")
    print(f"Expected Output: {test_case.expected_output}")
    print(f"Actual Output: {test_case.actual_output}")
    print("-------------------------------------------------")
# evaluate(test_cases, metrics)
