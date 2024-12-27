from enum import Enum
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
import json
from pathlib import Path

class GeobenchProvider:
    """
    A class to provide the GeoBenchmark dataset for evaluation.
    """

    class BanchmarkType(Enum):
        """
        An enumeration of the different types of benchmarks in the GeoBenchmark dataset.
        """
        NOUN = 1
        CHOICE = 2
        COMPLETION = 3
        TF = 4
        QA = 5
        DISCUSSION = 6
    
    def __init__(self):
        # Load the JSON content into a dictionary
        json_path = Path("geobenchmark_npee.json")
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        data = json.loads()

        # Create separate dictionaries for each key
        self.noun_dict = {"question": data["noun"]["question"], "answer": data["noun"]["answer"]}
        self.choice_dict = {"question": data["choice"]["question"], "answer": data["choice"]["answer"]}
        self.completion_dict = {"question": data["completion"]["question"], "answer": data["completion"]["answer"]}
        self.tf_dict = {"question": data["tf"]["question"], "answer": data["tf"]["answer"]}
        self.qa_dict = {"question": data["qa"]["question"], "answer": data["qa"]["answer"]}
        self.discussion_dict = {"question": data["discussion"]["question"], "answer": data["discussion"]["answer"]}

    def fetch_evaluation_dataset(self, model: DeepEvalBaseLLM, banchmark_type: BanchmarkType) -> EvaluationDataset:
        """
        Fetches the evaluation dataset for the specified benchmark type.
        Args:
            model: The model to evaluate.
            banchmark_type: The type of benchmark to fetch.
        Returns:
            An EvaluationDataset containing the test cases for the specified benchmark type.
        """
        test_cases = []
        banchmark_dict = {}
        match banchmark_type:
            case self.BanchmarkType.NOUN:
                banchmark_dict = self.noun_dict
            case self.BanchmarkType.CHOICE:
                banchmark_dict = self.choice_dict
            case self.BanchmarkType.COMPLETION:
                banchmark_dict = self.completion_dict
            case self.BanchmarkType.TF:
                banchmark_dict = self.tf_dict
            case self.BanchmarkType.QA:
                banchmark_dict = self.qa_dict
            case self.BanchmarkType.DISCUSSION:
                banchmark_dict = self.discussion_dict
            case _:
                raise ValueError("Invalid benchmark type")
        
        for i in enumerate(banchmark_dict):
            input = banchmark_dict.get("question")[i]
            expected_output = banchmark_dict.get("answer")[i]
            test_case = LLMTestCase(
                input=input,
                expected_output=expected_output,
                actual_output=model.generate(input),
                retrieval_context=""
            )
            test_cases.append(test_case)
        return EvaluationDataset(test_cases=test_cases)