import json
from enum import Enum
from pathlib import Path

from deepeval.dataset import EvaluationDataset
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from gas.logger import Logger

logger = Logger().get_logger()


class BanchmarkType(Enum):
    """
    An enumeration of the different types of benchmarks in the GeoBenchmark dataset.
    """

    NOUN = "noun"
    CHOICE = "choice"
    COMPLETION = "completion"
    TF = "tf"
    QA = "qa"
    DISCUSSION = "discussion"


class GeobenchProvider:
    """
    A class to provide the GeoBenchmark dataset for evaluation.
    """

    def __init__(self):
        # Load the JSON content into a dictionary
        json_path = Path(__file__).parent / "geobenchmark_npee.json"
        with Path.open(json_path, encoding="utf-8") as json_file:
            data = json.load(json_file)

        # Create separate dictionaries for each key
        self.noun_dict = {
            "question": data["noun"]["question"],
            "answer": data["noun"]["answer"],
        }
        self.choice_dict = {
            "question": data["choice"]["question"],
            "answer": data["choice"]["answer"],
        }
        self.completion_dict = {
            "question": data["completion"]["question"],
            "answer": data["completion"]["answer"],
        }
        self.tf_dict = {
            "question": data["tf"]["question"],
            "answer": data["tf"]["answer"],
        }
        self.qa_dict = {
            "question": data["qa"]["question"],
            "answer": data["qa"]["answer"],
        }
        self.discussion_dict = {
            "question": data["discussion"]["question"],
            "answer": data["discussion"]["answer"],
        }

    def fetch_evaluation_dataset(
        self,
        model: DeepEvalBaseLLM,
        banchmark_type: BanchmarkType,
        limit: int | None = None,
    ) -> EvaluationDataset:
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
            if limit is not None and i >= limit:
                break
            logger.debug(f"Generating test case {i+1}")
            question = banchmark_dict["question"][i]
            expected_output = banchmark_dict["answer"][i]
            prompt = self._fetch_prompt(banchmark_type, question)
            actual_output = model.generate(prompt)
            logger.debug(f"prompt: {prompt}")
            logger.debug(f"Expected output: {expected_output}")
            logger.debug(f"Generated output: {actual_output}")
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
            case BanchmarkType.NOUN:
                return self.noun_dict
            case BanchmarkType.CHOICE:
                return self.choice_dict
            case BanchmarkType.COMPLETION:
                return self.completion_dict
            case BanchmarkType.TF:
                return self.tf_dict
            case BanchmarkType.QA:
                return self.qa_dict
            case BanchmarkType.DISCUSSION:
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
            case BanchmarkType.NOUN:
                role = (
                    "You are a geoscience expert with a deep understanding of geological and "
                    "geographical terminology."
                )
                context = (
                    "You are asked to provide a concise definition of a specific geoscience concept. "
                    "Your answer should be short and to the point. "
                    "Your response should have technical specifications and relevant numerical details if applicable."
                )
                structure = f"Concept: {input}. Answer:"
            case BanchmarkType.CHOICE:
                context = (
                    "The question below is about college geoscience. "
                    "The question below have answer options (associated with alphabet letters) "
                    "out of which only one is correct. "
                    "Answer the question by selecting the correct option. "
                    'Answer should be " A", " B", " C", or " D". Without any additional information.'
                )
                structure = f"Question:{input}.\nAnswer:"
            case BanchmarkType.COMPLETION:
                context = (
                    "The sentence below is about college geoscience. "
                    "The sentence below is incomplete and has a blank spaces. "
                    'Blank spaces are represented by "_" and "()". '
                    "Answer the sentence by filling in the blank space. "
                    "Answer should be just the missing words that fits in each blank. "
                    "Without any additional information."
                )
                structure = f"Sentence: {input}.\nAnswer:"
            case BanchmarkType.TF:
                context = (
                    "The statement below is about college geoscience. "
                    "The statement below is either true or false. "
                    "Answer if the statement is true or false. "
                    'Answer should be either "True" or "False". '
                    "Without any additional information."
                )
                structure = f"Statement: {input}. Answer:"
            case BanchmarkType.QA:
                role = (
                    "You are a geoscience expert answering detailed questions about geological and "
                    "geographical processes."
                )
                context = (
                    "You are asked to provide a detailed answer to a specific geoscience question. "
                    "Your answer should be thorough and detailed, providing all relevant information. "
                    "Your response should have technical specifications and relevant numerical details if applicable. "
                    "Your answer should be short and to the point. "
                )
                structure = f"Question: {input}. Answer:"
            case BanchmarkType.DISCUSSION:
                role = (
                    "You are a geoscience expert leading a detailed discussion about geological and "
                    "geographical topics."
                )
                context = (
                    "Provide a thorough, structured, and contextually accurate explanation based on your "
                    "knowledge of geoscience dictionaries, encyclopedias, and related academic materials."
                )
                structure = f"Topic: {input}. Answer:"
            case _:
                raise ValueError("Invalid benchmark type")

        return f"{role}\n\n{context}\n\n{structure}"
