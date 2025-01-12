import json
import os
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, BaseMetric, GEval, PromptAlignmentMetric
from deepeval.test_case import LLMTestCaseParams

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

        with Path.open(de_result_path, encoding="utf-8") as json_file:
            data = json.load(json_file)

        test_cases: list[dict[str, Any]] = data["testCases"]

        totals: dict[str, dict[str, float]] = {}
        for test_case in test_cases:
            metrics: list[dict[str, Any]] = test_case.get("metricsData", [])
            for metric in metrics:
                name = metric.get("name", "None")
                score = metric.get("score", "None")
                success = metric.get("success", "False")
                cost = metric.get("evaluationCost", 0)
                if name not in totals:
                    totals[name] = {"total_score": 0.0, "total_success": 0.0, "total_cost": 0.0, "total_tests": 0.0}
                totals[name]["total_score"] += score
                totals[name]["total_success"] += 1 if success else 0
                totals[name]["total_tests"] += 1
                totals[name]["total_cost"] += cost

        def format_metric_values(values):
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
        for metric, values in totals.items():
            summary.append(
                {
                    "model": self.model.get_model_name(),
                    "judge": self.model_judge,
                    "task": self.task.value,
                    "metric": metric,
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
        evaluate(dataset, metrics, write_cache=True, print_results=False, max_concurrent=30)
        logger.info("Done.")
        logger.info("Creating benchmark summary...")
        summary = self._create_summary()
        logger.info("Done.")
        logger.info("Saving benchmark into CSV file...")
        outpath = Path(self.results_path) / f"summary_{self.model.get_model_name()}_{self.task.value}.json"
        self.data_provider.save(summary, outpath, "json")
        logger.info("Done.")
