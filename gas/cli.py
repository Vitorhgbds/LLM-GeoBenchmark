import argparse

from deepeval import evaluate
from gas import metrics
from gas.logger import Logger

from gas.database.geobenchmark_provider import BanchmarkType, GeobenchProvider
from gas.models import __all__ as models
from gas.models import model_name_class_map

logger = Logger().get_logger()
def run(model: str, task: str, limit: int | None = None) -> None:
    """
    Run the Geobenchmark evaluation suite.
    Args:
        model: The model to evaluate.
        task: The task to evaluate.
        limit: The total number of test cases to generate.
    """
    logger.info(f"Running Geobenchmark evaluation suite for model: {model}")
    logger.info(f"Task: {task}")
    logger.info(f"Test Cases Limit: {limit}")
    provider: GeobenchProvider = GeobenchProvider()
    model_instance = model_name_class_map.get(model)()
    list_evaluations = []
    if task == "all":
        for task_type in BanchmarkType:
            evaluation_dict = provider.fetch_evaluation_dataset(model_instance, task_type, limit)
            list_evaluations.append(evaluation_dict)
    else:
        list_evaluations = [provider.fetch_evaluation_dataset(model_instance, BanchmarkType(task), limit)]
    """
    for evaluation in list_evaluations:
        for test_case in evaluation.test_cases:
            logger.debug(f"Question: {test_case.input}")
            logger.debug(f"Expected Output: {test_case.expected_output}")
            logger.debug(f"Actual Output: {test_case.actual_output}")
    """
    for evaluation in list_evaluations:
        evaluate(evaluation, metrics.metrics)                

def cli() -> None:
    """Main entry point for the gas command line interface."""
    parser = argparse.ArgumentParser(description='GAS: A command line tool for running the Geobenchmark evaluation suite.')
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        dest="model",
        help="Models to evaluate. (default: %(default)s) (required: %(required)s)",
        choices=models,
        default="Llama3_8B",
    )
    parser.add_argument(
        "--task",
        type=str,
        dest="task",
        help="Task for . (default: %(default)s)",
        choices=["noun", "choice", "completion", "tf", "qa", "discussion", "all"],
        default="all",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        dest="limit",
        help="Total number of test cases to generate. (default: %(default)s)",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. (default: %(default)s)",
        default="INFO",
    )
    parser.set_defaults(func=run)

    try:
        args = parser.parse_args()
        if hasattr(args, "func"):
            args_dict = vars(args).copy()
            Logger().set_level(args_dict.pop("log_level"))
            args_dict.pop("func", None)
            args.func(**args_dict)
        else:
            parser.print_help()
    except Exception as e:
        print(e)
        parser.print_help()
    
if __name__ == "__main__":
    cli()