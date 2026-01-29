import json
import os
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, BaseMetric, GEval, PromptAlignmentMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval.evaluate.configs import CacheConfig, AsyncConfig, ErrorConfig, DisplayConfig

from gas.commons import TaskType
from gas.logger import Logger
from gas.metrics import BertSimilarityMetric, ObjectiveAccuracyMetric
from gas.pipelines.pipeline import Pipeline
from gas.providers import PromptProvider, TestCasesProvider

logging = Logger()
logger = logging.get_logger()


class EvaluationPipeline(Pipeline):
    def __init__(self, model_judge: str, results_path: str, *args, **kwargs):
        super().__init__(**kwargs)
        self.model_judge = model_judge
        self.results_path = Path(results_path)

    def _create_summary(self) -> list[dict[str, Any]]:
        """
        Create a summary of the evaluation results.
        This method reads the evaluation results from a JSON file, processes the data,
        and generates a summary report.
        It calculates the total score, success rate, and other metrics for each test case.
        The summary is then printed in a formatted table.

        NOTE: The summary includes the following metrics:
            - average_score: The average score for the test case.
            - success_rate: The success rate for the test case.
            - total_score: The total score for the test case.
            - total_success: The total number of successful evaluations.
            - total_tests: The total number of tests conducted.
            - total_cost: The total cost of the evaluation.

        Raises:
            FileNotFoundError: If the evaluation results file is not found.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the summary of the evaluation results.
        """
        deepEval_result_folder = os.environ.get("DEEPEVAL_RESULTS_FOLDER", None)
        if not deepEval_result_folder:
            logger.error("File Not Found")
            raise FileNotFoundError()

        de_result_folder_path = Path(deepEval_result_folder)
        last_result_file = self.data_provider.find_last_file(de_result_folder_path)

        de_result_path = Path(last_result_file)
        tc_result_path = (
            self.test_cases_path / "result" / f"result_{self.model.get_model_name()}_{self.task.value}.json"
        )
        tc_result_path.parent.mkdir(parents=True, exist_ok=True)
        if tc_result_path.exists():
            tc_result_path.unlink()
        de_result_path = de_result_path.rename(tc_result_path)
        logger.debug(f"Renamed to {tc_result_path}")
        with Path.open(tc_result_path, encoding="utf-8") as json_file:
            data = json.load(json_file)

        test_cases: list[dict[str, Any]] = data["testCases"]
        logger.debug(f"Loaded {len(test_cases)} test cases from {tc_result_path}")
        totals: dict[str, dict[str, float]] = {}
        for test_case in test_cases:
            metrics: list[dict[str, Any]] = test_case.get("metricsData", [])
            #logger.debug(f"Processing test case with metrics: {metrics}")
            for metric in metrics:
                #logger.debug(f"Processing metric: {metric}")
                name = metric.get("name", None)
                score = metric.get("score", None)
                success = metric.get("success", None)
                cost = metric.get("evaluationCost", 0.0)
                if name not in totals:
                    totals[name] = {"total_score": 0.0, "total_success": 0.0, "total_cost": 0.0, "total_tests": 0.0}
                if any(x == None for x in [name, score, success]):
                    logger.warning(f"Skipping metric with missing data: {metric}")
                    continue
                totals[name]["total_score"] += score
                totals[name]["total_success"] += 1 if success else 0
                totals[name]["total_tests"] += 1
                totals[name]["total_cost"] += cost

        def format_metric_values(values: dict[str, float]) -> str:
            """
            Format the metric values for display.
            args:
                values (dict[str, float]): The metric values to format.
            returns:
                str: The formatted metric values.
            """
            details = "\n".join([f"{k}: {v}" for k, v in values.items()])
            avg_score = f"average score: {values['total_score'] / values['total_tests']:.2f}"
            success_rate = f"success rate: {values['total_success'] / values['total_tests'] * 100:.2f}%"
            return f"{details}\n{avg_score}\n{success_rate}"

        logging.print_table_panel(
            {metric: format_metric_values(values) for metric, values in totals.items()},
            title="[bold bright_green]Benchmark Summary[/bold bright_green]",
            border_style="green",
        )

        summary: list[dict[str, Any]] = []
        for m, values in totals.items():
            summary.append(
                {
                    "model": self.model.get_model_name(),
                    "judge": self.model_judge,
                    "task": self.task.value,
                    "metric": m,
                    **values,
                    "average_score": values["total_score"] / values["total_tests"],
                    "success_rate": values["total_success"] / values["total_tests"],
                }
            )

        return summary

    def _fetch_metrics(self) -> list[BaseMetric]:
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
        if self.task == TaskType.CHOICE or self.task == TaskType.TF:
            return [ObjectiveAccuracyMetric()]
        else:
            return [
                BertSimilarityMetric(threshold=0.5),
                PromptAlignmentMetric(
                    prompt_instructions=[PromptProvider.fetch_instruction(self.task)],
                    include_reason=True,
                    model=self.model_judge,
                    threshold=0.5,
                ),
                AnswerRelevancyMetric(threshold=0.5, model=self.model_judge, include_reason=True),
                GEval(
                    name="Correctness",
                    model=self.model_judge,
                    evaluation_steps=[
                        "Compare the actual output directly with the expected output to verify factual accuracy.",
                        (
                            "Check if all elements mentioned in the expected output are present"
                            "and correctly represented in the actual output."
                        ),
                        (
                            "Assess if there are any discrepancies"
                            "in details, values, or information between the actual and expected outputs."
                        ),
                    ],
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                        LLMTestCaseParams.EXPECTED_OUTPUT,
                    ],
                ),
            ]

    def run(self, *args, **kwargs):
        """
        Run the evaluation pipeline.
        args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        logger.info("Fetching benchmark metrics")
        metrics = self._fetch_metrics()
        log_metrics = "\n".join([m.__name__ for m in metrics])
        Logger().print_panel(
            log_metrics,
            title="[bold medium_orchid]Benchmark Metrics[/bold medium_orchid]",
            border_style="purple",
            justify="center",
        )
        logger.info("Done.")
        logger.info("Fetching generated test cases...")
        tests_path = Path(self.test_cases_path) / f"{self.model.get_model_name()}_{self.task.value}.json"
        dataset = TestCasesProvider.fetch(tests_path)
        logger.info("Done.")
        logger.info("Starting Deepeval Evaluation...")
        done = False
        use_cache = False
        while not done:
            try:
                evaluate(dataset.test_cases, metrics, identifier=f"{self.task.value}_{self.model.get_model_name()}", cache_config=CacheConfig(write_cache=True, use_cache=use_cache), async_config=AsyncConfig(run_async=True, max_concurrent=5), error_config=ErrorConfig(ignore_errors=False), display_config=DisplayConfig(verbose_mode=False))
                done = True
            except KeyboardInterrupt:
                logger.info("Evaluation interrupted by user.")
                return
            except Exception as e:
                logger.error(f"Deepeval Evaluation failed: {e.with_traceback(None)}")
                continue
            finally:
                use_cache = True
        logger.info("Done.")
        logger.info("Creating benchmark summary...")
        summary = self._create_summary()
        logger.info("Done.")
        logger.info("Saving benchmark into CSV file...")
        outpath = Path(self.results_path) / f"summary_{self.model.get_model_name()}_{self.task.value}.json"
        self.data_provider.save(summary, outpath, "json")
        logger.info("Done.")
