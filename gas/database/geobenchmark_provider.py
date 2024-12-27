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
        json_path = Path(__file__).parent / "geobenchmark_npee.json"
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Create separate dictionaries for each key
        self.noun_dict = {"question": data["noun"]["question"], "answer": data["noun"]["answer"]}
        self.choice_dict = {"question": data["choice"]["question"], "answer": data["choice"]["answer"]}
        self.completion_dict = {"question": data["completion"]["question"], "answer": data["completion"]["answer"]}
        self.tf_dict = {"question": data["tf"]["question"], "answer": data["tf"]["answer"]}
        self.qa_dict = {"question": data["qa"]["question"], "answer": data["qa"]["answer"]}
        self.discussion_dict = {"question": data["discussion"]["question"], "answer": data["discussion"]["answer"]}

    def fetch_evaluation_dataset(self, model: DeepEvalBaseLLM, banchmark_type: BanchmarkType, limit: int|None=None) -> EvaluationDataset:
        """
        Fetches the evaluation dataset for the specified benchmark type.
        Args:
            model: The model to evaluate.
            banchmark_type: The type of benchmark to fetch.
        Returns:
            An EvaluationDataset containing the test cases for the specified benchmark type.
        """
        banchmark_dict = self._fetch_task_dict(banchmark_type)
        test_cases = []
        for i in range(len(banchmark_dict["question"])):
            print(f"Generating test case {i+1}")
            if limit is not None and i >= limit:
                break
            question = banchmark_dict["question"][i]
            expected_output = banchmark_dict["answer"][i]
            prompt = self._fetch_prompt(banchmark_type, question)

            actual_output = model.generate(prompt)
            test_case = LLMTestCase(
                input=question,
                expected_output=expected_output,
                actual_output=actual_output,
            )
            test_cases.append(test_case)
        
        return EvaluationDataset(test_cases=test_cases)
    
    def _fetch_task_dict(self, banchmark_type: BanchmarkType) -> dict:
        """
        Fetches the dictionary corresponding to the specified benchmark type.
        Args:
            banchmark_type: The type of benchmark to fetch.
        Returns:
            A dictionary containing the questions and answers for the specified benchmark type.
        """
        match banchmark_type:
            case self.BanchmarkType.NOUN:
                return self.noun_dict
            case self.BanchmarkType.CHOICE:
                return self.choice_dict
            case self.BanchmarkType.COMPLETION:
                return self.completion_dict
            case self.BanchmarkType.TF:
                return self.tf_dict
            case self.BanchmarkType.QA:
                return self.qa_dict
            case self.BanchmarkType.DISCUSSION:
                return self.discussion_dict
            case _:
                raise ValueError("Invalid benchmark type")

    def _fetch_prompt(self, banchmark_type: BanchmarkType, input: str) -> str:
        """
        Fetches the prompt for the specified benchmark type and input.
        Args:
            banchmark_type: The type of benchmark to fetch.
            input: The input to fetch the prompt for.
        Returns:
            The prompt for the specified benchmark type and input.
        """
        context = ""
        role = ""
        structure = ""
        match banchmark_type:
            case self.BanchmarkType.NOUN:
                role = "You are a geoscience expert with a deep understanding of geological and geographical terminology."
                context = "Using your expertise, explain the meaning of the following concept in clear and concise language. The explanation should reflect definitions commonly found in geoscience dictionaries and encyclopedias, such as Signal G7 and Wikipedia."
                structure = f"Concept: {input}. Answer:"
            case self.BanchmarkType.CHOICE:
                role = "You are a geoscience expert tasked with answering multiple-choice questions based on geological and geographical knowledge derived from dictionaries, encyclopedias, and related materials."
                context = "Below is a question with options. Select the correct answer from the given choices."
                structure = f"Question: {input}. Answer:"
            case self.BanchmarkType.COMPLETION:
                role = "You are a geoscience expert tasked with completing sentences based on geological and geographical knowledge derived from dictionaries, encyclopedias, and related materials."
                context = "Fill in the blanks or incomplete sentences with the most accurate and relevant terms or phrases."
                structure = f"Sentence: {input} Answer:"
            case self.BanchmarkType.TF:
                role = "You are a geoscience expert tasked with determining whether the following statements are True or False based on geological and geographical knowledge derived from dictionaries, encyclopedias, and related resources."
                context = "Provide a clear and concise judgment for each statement."
                structure = f"Statement: {input}. Answer:"
            case self.BanchmarkType.QA:
                role = "You are a geoscience expert answering detailed questions about geological and geographical processes."
                context = "Provide comprehensive, accurate, and contextually relevant answers based on geoscience dictionaries, encyclopedias, and related materials."
                structure = f"Question: {input}. Answer:"
            case self.BanchmarkType.DISCUSSION:
                role = "You are a geoscience expert leading a detailed discussion about geological and geographical topics."
                context = "Provide a thorough, structured, and contextually accurate explanation based on your knowledge of geoscience dictionaries, encyclopedias, and related academic materials."
                structure = f"Topic: {input}. Answer:"
            case _:
                raise ValueError("Invalid benchmark type")
            
        return f"{role}\n\n{context}\n\n{structure}"