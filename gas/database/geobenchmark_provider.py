import json
from enum import Enum
from pathlib import Path

from gas.logger import Logger

logger = Logger().get_logger()


class BenchmarkType(Enum):
    """
    An enumeration of the different types of benchmarks in the GeoBenchmark dataset.
    """

    NOUN = "NOUN"
    CHOICE = "CHOICE"
    COMPLETION = "COMPLETION"
    TF = "TF"
    QA = "QA"


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
        self.benchmark_datasets_dict: dict[BenchmarkType, dict[str, str]] = {}
        for type in BenchmarkType:
            self.benchmark_datasets_dict[type] = {
                "question": data[type.value.lower()]["question"],
                "answer": data[type.value.lower()]["answer"],
            }

    def fetch_task_instruction(self, banchmark_type: BenchmarkType) -> str:
        """
        Fetches the task instruction for the specified benchmark type.
        Args:
            banchmark_type: The type of benchmark.
        Returns:
            The task instruction for the specified benchmark type.
        """
        match banchmark_type:
            case BenchmarkType.NOUN:
                return (
                    "You are asked to provide a concise definition of a specific geoscience concept. "
                    "Your answer should be short and to the point. "
                    "Your response should have technical specifications and relevant numerical details if applicable."
                )
            case BenchmarkType.CHOICE:
                return (
                    "The question below is about college geoscience. "
                    "The question below have answer options (associated with alphabet letters) "
                    "out of which only one is correct. "
                    "Answer the question by selecting the correct option. "
                    'Response should be just "A", "B", "C", or "D". '
                    "Do not add any explanations."
                )
            case BenchmarkType.COMPLETION:
                return (
                    "The sentence below is about college geoscience. "
                    "The sentence below is incomplete and has a blank spaces. "
                    'Blank spaces are represented by "_" and "()". '
                    "Answer the sentence by filling in the blank space. "
                    "Answer should be just the missing words that fits in each blank "
                    "without any additional information."
                )
            case BenchmarkType.TF:
                return (
                    "The statement below is about college geoscience. "
                    "The statement below is either true or false. "
                    "Answer if the statement is true or false. "
                    'Response should be just "True" or "False" '
                    "without any additional information."
                )
            case BenchmarkType.QA:
                return (
                    "You are asked to provide a detailed answer to a specific geoscience question. "
                    "Your answer should be thorough and detailed, providing all relevant information. "
                    "Your response should have technical specifications and relevant numerical details if applicable. "
                    "Your answer should be short and to the point. "
                )
            case _:
                raise ValueError("Invalid benchmark type")
