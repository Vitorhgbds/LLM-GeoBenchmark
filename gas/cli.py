import argparse

from deepeval.models import DeepEvalBaseLLM
from rich.console import Console
from rich.panel import Panel

from gas.database.geobenchmark_provider import BanchmarkType, GeobenchProvider
from gas.logger import Logger
from gas.models import __all__ as models, model_name_class_map

logger = Logger().get_logger()


def print_help(parser: argparse.ArgumentParser) -> None:
    """Print the parser's help message beautifully."""
    console = Console()
    help_text = parser.format_help()
    console.print(Panel(help_text, title="[bold cyan]Help[/bold cyan]", border_style="blue"))


def run(model: str, task: str, limit: int | None = None, path: str | None = None) -> None:
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
    if path:
        model_instance = model_name_class_map.get("PEFTModel", DeepEvalBaseLLM)(path)
    else:
        model_instance = model_name_class_map.get(model, DeepEvalBaseLLM)()
    list_evaluations = []
    if task == "all":
        for task_type in BanchmarkType:
            evaluation_dict = provider.fetch_evaluation_dataset(model_instance, task_type, limit)
            list_evaluations.append(evaluation_dict)
    else:
        list_evaluations = [provider.fetch_evaluation_dataset(model_instance, BanchmarkType(task), limit)]

    # for evaluation in list_evaluations:
    #    evaluate(evaluation, metrics.metrics)


def cli() -> None:
    """Main entry point for the gas command line interface."""
    parser = argparse.ArgumentParser(
        description="GAS: A command line tool for running the Geobenchmark evaluation suite."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="<MODEL>",
        dest="model",
        help="Models to evaluate. Choices: [%(choices)s] (default: %(default)s)",
        choices=models,
        default="Llama3_8B",
    )
    parser.add_argument(
        "--task",
        type=str,
        dest="task",
        metavar="<TASK>",
        help="Task to evaluate. Choices: [%(choices)s] (default: %(default)s)",
        choices=["noun", "choice", "completion", "tf", "qa", "discussion", "all"],
        default="all",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        dest="limit",
        metavar="<LIMIT>",
        help="Total number of test cases to generate. (default: %(default)s)",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="<LEVEL>",
        help="Set the logging level. Choices: [%(choices)s] (default: %(default)s) ",
        default="INFO",
    )
    parser.add_argument(
        "--path",
        type=str,
        dest="path",
        help="Path to the model checkpoint.",
    )
    parser.print_help = lambda: print_help(parser)  # type: ignore[method-assign, misc, assignment]
    parser.set_defaults(func=run)
    try:
        args = parser.parse_args()
        if hasattr(args, "func"):
            args_dict = vars(args).copy()
            Logger().set_level(args_dict.pop("log_level"))
            args_dict.pop("func", None)
            args.func(**args_dict)
        else:
            print_help(parser)
    except Exception as e:
        print(e)
        print_help(parser)


if __name__ == "__main__":
    cli()
