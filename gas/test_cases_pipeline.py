import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, BaseMetric, GEval, PromptAlignmentMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from rich.progress import BarColumn, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn

from gas.commons import (
    DO_SAMPLE,
    GPT_JUDGE,
    MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    PENALTY_ALPHA,
    SEED,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from gas.database.geobenchmark_provider import BenchmarkType, GeobenchProvider
from gas.logger import Logger, MyProgress
from gas.metrics import (
    accuracy_score,
)
from gas.metrics.bert_score import BertSimilarityMetric
from gas.models import model_name_class_map

logger = Logger().get_logger()


def build_llm_test_cases(
    model: DeepEvalBaseLLM,
    prompt_instruction: str,
    dataset: dict[str, list[str]],
    limit: int | None,
    obj_task: BenchmarkType,
) -> EvaluationDataset:
    """
    Build an evaluation dataset with LLM Test Cases.
    Args:
        model (DeepEvalBaseLLM): The model to evaluate.
        prompt_instruction (str): prompt with task instruction
        dataset (dict[str,str]): dataset with input and expected output
        limit (int): total tests limit
    Returns:
        An EvaluationDataset containing the LLM test cases with generated output.
    """
    test_cases = []
    total = limit if limit else len(dataset["question"])
    with MyProgress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(bar_width=None),
        "[progress.completed]({task.completed}/{task.total})",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
        console=Logger().console,
    ) as progress:
        task = progress.add_task(f"[bold bright_green]Generating {total} Test Cases:[/bold bright_green]", total=total)
        for i in range(total):
            question = dataset["question"][i]
            expected_output = dataset["answer"][i]

            input_prompt = f"### Input:\n{question}\n\n" "### Response:\n"
            prompt = f"{prompt_instruction}" f"{input_prompt}"

            if not (obj_task == BenchmarkType.CHOICE or obj_task == BenchmarkType.TF):
                prompt = model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )

            actual_output = model.generate(prompt)
            test_case = LLMTestCase(
                input=question,
                expected_output=expected_output,
                actual_output=actual_output,
            )
            test_cases.append(test_case)

            if Logger().get_level() == "DEBUG":
                Logger().print_information_table_panel(
                    [{"Input:": question, "Expected:": expected_output, "Generated:": actual_output}],
                    title=f"[bold bright_green]DEBUG Test Case {i + 1} Result[/bold bright_green]",
                    border_style="green",
                    justify="full",
                )
            progress.update(task, advance=1)

    return EvaluationDataset(test_cases=test_cases)

def save_evaluation_dataset(dataset: EvaluationDataset, file_name: str, **kwargs):
    full_file_path = Path().cwd() / file_name
    direcotry_path = kwargs.get("directory_full_path", None)
    if kwargs.get("directory_full_path", None):
        full_file_path = f"{direcotry_path}/{file_name}" 
    with open(full_file_path, "w", encoding="utf-8") as file:
        json_data = [
            {
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "expected_output": test_case.expected_output,
            }
            for test_case in dataset
        ]
        json.dump(json_data, file, indent=4, ensure_ascii=False)
        

class testCasesPipeline:
    def __init__(self, model: str, task: str, limit: int | None):
        self.model_name: str = model
        self.task: BenchmarkType = BenchmarkType(task)
        self.limit: int | None = limit

    def _show_general_information(self):
        """
        Shows the current benchmark setup.
        """

        information = [
            {
                "Geoscience benchmark task:": self.task.name,
                "Tests limit:": self.limit,
                "Model for evaluation:": self.model_name,
                "LOG LEVEL:": f"{Logger().get_level()}",
                "VERBOSE:": Logger().is_verbose,
                "SEED: ": SEED,
                "MIN NEW TOKENS: ": MIN_NEW_TOKENS,
                "MAX NEW TOKENS: ": MAX_NEW_TOKENS,
                "DO SAMPLE: ": DO_SAMPLE,
                "TEMPERATURE: ": TEMPERATURE,
                "TOP P: ": TOP_P,
                "TOP K: ": TOP_K,
                "PENALTY ALPHA: ": PENALTY_ALPHA,
            }
        ]

        Logger().print_information_table_panel(
            information,
            title="[bold medium_orchid]Benchmark General Information[/bold medium_orchid]",
            border_style="purple",
            expand=False,
            justify="center",
        )
        

    def run(self, **kwargs: dict[str, Any]):
        """
        Method to execute the benchmark.


        kwargs (dict[str,any]): model kwargs
        """
        self._show_general_information()
        logger.info("Fetching Geobench dataset...")
        provider: GeobenchProvider = GeobenchProvider()
        geobench_data = provider.benchmark_datasets_dict.get(self.task, {})
        logger.info("Done.")
        logger.info("Fetching task instruction...")
        task_instruction = provider.fetch_task_instruction(self.task)
        PROMPT_INSTRUCTION = (
            "Below is an instruction that describes a task, "
            "paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Human:\n{task_instruction}\n\n"
        )
        logger.info("Done.")
        Logger().print_panel(
            f"[yellow][align=left]{str(PROMPT_INSTRUCTION)}[/align][/yellow]",
            title="[bold bright_yellow]Prompt Instruction[/bold bright_yellow]",
            border_style="yellow",
        )
        logger.info("Trying to initialize model")
        model_instance = model_name_class_map.get(self.model_name, DeepEvalBaseLLM)(**kwargs)
        logger.info("Done.")
        logger.info("Building evaluation dataset...")
        evaluation_dataset = build_llm_test_cases(
            model_instance, PROMPT_INSTRUCTION, geobench_data, self.limit, self.task
        )
        path = save_evaluation_dataset(evaluation_dataset, f"{model_instance.get_model_name()}_{self.task.value}.json", **kwargs)
        logger.info("Done.")
        logger.info(f"Evaluation dataset stored in: {path}")
