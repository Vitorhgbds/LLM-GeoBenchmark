import csv
import glob
import json
import os
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

def get_latest_file(directory: str, file_pattern: str = "*"):
    """
    Get the latest file based on timestamp in the name or modification time.
    
    Args:
        directory (str): Directory path where files are stored.
        file_pattern (str): Pattern to match filenames (e.g., "*.txt", "*.log").
        
    Returns:
        str: The path of the latest file.
    """
    # List files matching the pattern
    files = glob.glob(os.path.join(directory, file_pattern))
    
    if not files:
        print("No files found matching the pattern.")
        return None
    
    # Sort files by their modification time
    latest_file = max(files, key=os.path.getmtime)
    
    print(f"Latest file: {latest_file}")
    return latest_file


def fetch_evaluation_dataset(file_name: str, **kwargs) -> EvaluationDataset:
    
    full_file_path = Path().cwd() / file_name
    direcotry_path = kwargs.get("directory_full_path", None)
    if kwargs.get("directory_full_path", None):
        full_file_path = f"{direcotry_path}/{file_name}" 
        
    dataset = EvaluationDataset()
    
    dataset.add_test_cases_from_json_file(
        file_path=full_file_path,
        input_key_name="input",
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output"
    )
    
    return dataset

def fetch_results(model: str, task_type: str) -> list[dict[str, Any]]:
    """fetch benchmark results

    Args:
        model (str): model name
        task_type (str): task type

    Returns:
        list[dict[str,any]]: records with benchmark results
    """
    result_folder = os.environ.get("DEEPEVAL_RESULTS_FOLDER", None)
    if not result_folder:
        raise Exception("Not Found")
    result_folder_path = Path().cwd() / result_folder
    
    latest_file = get_latest_file(result_folder_path)
    
    benchmark_records: list[dict[str, Any]] = []
    with Path.open(latest_file, encoding="utf-8") as json_file:
        data = json.load(json_file)
        
    test_cases: list[dict[str, Any]] = data["testCases"]

    i = 0
    total = len(test_cases)
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
        task = progress.add_task(
            f"[bold bright_green]Fetching {total} test cases results:[/bold bright_green]", total=total
        )
        for test_case in test_cases:
            base_record = {
                "model": model,
                "task": task_type,
                "testId": i,
                "input": test_case.get("input", "None"),
                "expected": test_case.get("expectedOutput", "None"),
                "output": test_case.get("actualOutput", "None"),
            }
            metrics: list[dict[str, Any]] = test_case.get("metricsData", [])
            for metric in metrics:
                record = {
                    **base_record,
                    "metric": metric.get("name", "None"),
                    "score": metric.get("score", "None"),
                    "reason": metric.get("reason", "None"),
                    "threshold": metric.get("threshold", "None"),
                    "success": metric.get("success", "False"),
                    "evaluationCost": metric.get("evaluationCost", 0),
                }
                benchmark_records.append(record)
            i += 1
            progress.update(task, advance=1)
    return benchmark_records


def save_records(records: list[dict[str, Any]], output_file_name: str = "benchmark_cache.csv") -> str:
    """Save benchmark results on CSV file

    Args:
        model (str): evaluated model name
        task (str): benchmark task
        task (str): benchmark task
        output_file_name (str, optional): output file name. Defaults to "benchmark_cache.csv".
    Returns:
        str: output csv file path
    """
    df = pd.DataFrame(records)
    df["input"] = df["input"].str.replace("\n", r"\\n", regex=False)
    df["expected"] = df["expected"].str.replace("\n", r"\\n", regex=False)
    df["output"] = df["output"].str.replace("\n", r"\\n", regex=False)
    out_file_path = Path.cwd() / output_file_name
    file_exists = out_file_path.exists()

    df.to_csv(
        out_file_path,
        index=False,  # Don't write index
        mode="a" if file_exists else "w",  # Append mode
        encoding="utf-8",
        lineterminator="\n",
        sep=";",
        doublequote=True,
        quoting=csv.QUOTE_NONNUMERIC,  # csv.QUOTE_ALL to quote all fields
        header=not file_exists,
    )
    return str(out_file_path)


class evaluationPipeline:
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

    def _show_benchmark_summary(self, benchmark_records: list[dict[str, Any]]):
        totals: dict[str, dict[str, float]] = {}
        for record in benchmark_records:
            metric_name = record["metric"]
            score = record["score"]
            success = record["success"]
            if metric_name not in totals:
                totals[metric_name] = {"total_score": 0.0, "total_success": 0.0, "count": 0.0}
            totals[metric_name]["total_score"] += score
            totals[metric_name]["total_success"] += 1 if success else 0
            totals[metric_name]["count"] += 1

        summary: dict[str, str] = {}

        for metric, values in totals.items():
            summary[metric] = (
                f"average score: {values['total_score'] / values['count']}\n"
                f"success rate: {values['total_success'] / values['count'] * 100}\n"
                f"total tests: {values['count']}"
            )

        Logger().print_information_table_panel(
            [summary], title="[bold bright_green]Benchmark Summary[/bold bright_green]", border_style="green"
        )

    def _fetch_metrics(self, prompt_instruction: str) -> list[BaseMetric]:
        """
        Fetches the evaluation metrics for the specified benchmark type.

        NOTE: Metrics for subjective tasks
        - PromptAlignmentMetric -GPT
        - Answer Relevancy - GPT
        - Correctness - GPT
        - Semantic Similarity
        NOTE: Metrics for objective tasks
        - Accuracy

        args:
            prompt_instruction (str): prompt instruction for prompt alignment metric.
        returns:
            list[BaseMetric]: list of metrics for the specified benchmark
        """
        if self.task == BenchmarkType.CHOICE or self.task == BenchmarkType.TF:
            return [accuracy_score.ObjectiveAccuracyMetric()]
        else:
            return [
                # PromptAlignmentMetric(
                #     prompt_instructions=[prompt_instruction], include_reason=True, model=GPT_JUDGE, threshold=0.5
                # ),
                # AnswerRelevancyMetric(threshold=0.5, model=GPT_JUDGE, include_reason=True),
                # GEval(
                #     name="Correctness",
                #     model=GPT_JUDGE,
                #     evaluation_steps=[
                #         "Compare the actual output directly with the expected output to verify factual accuracy.",
                #         (
                #             "Check if all elements mentioned in the expected output are present"
                #             "and correctly represented in the actual output."
                #         ),
                #         (
                #             "Assess if there are any discrepancies"
                #             "in details, values, or information between the actual and expected outputs."
                #         ),
                #     ],
                #     evaluation_params=[
                #         LLMTestCaseParams.INPUT,
                #         LLMTestCaseParams.ACTUAL_OUTPUT,
                #         LLMTestCaseParams.EXPECTED_OUTPUT,
                #     ],
                # ),
                BertSimilarityMetric(threshold=0.5),
            ]

    def run(self, **kwargs: dict[str, Any]):
        """
        Method to execute the benchmark.


        kwargs (dict[str,any]): model kwargs
        """
        self._show_general_information()
        logger.info("Fetching task instruction...")
        provider: GeobenchProvider = GeobenchProvider()
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
        logger.info("Fetching benchmark metrics")
        metrics = self._fetch_metrics(PROMPT_INSTRUCTION)
        log_metrics = "\n".join([m.__name__ for m in metrics])
        Logger().print_panel(
            log_metrics,
            title="[bold medium_orchid]Benchmark Metrics[/bold medium_orchid]",
            border_style="purple",
            justify="center",
        )
        logger.info("Done.")
        logger.info("Trying to initialize model")
        model_instance = model_name_class_map.get(self.model_name, DeepEvalBaseLLM)(**kwargs)
        logger.info("Done.")
        logger.info("Fetching evaluation dataset...")
        evaluation_dataset = fetch_evaluation_dataset(
            f"{model_instance.get_model_name()}_{self.task.value}.json", **kwargs
        )
        logger.info("Done.")
        logger.info("Starting Deepeval Evaluation...")
        evaluate(evaluation_dataset, metrics, write_cache=True, print_results=False, max_concurrent=30)
        logger.info("Done.")
        logger.info("Fetching benchmark results...")
        results = fetch_results(model_instance.get_model_name(), self.task.value)
        logger.info("Done.")
        self._show_benchmark_summary(results)
        logger.info("Saving benchmark into CSV file...")
        save_records(results)
