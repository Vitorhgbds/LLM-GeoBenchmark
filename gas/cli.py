import argparse
import os

from dotenv import load_dotenv

from gas.evaluation_pipeline import evaluationPipeline
from gas.logger import Logger
from gas.models import __all__ as models
from gas.test_cases_pipeline import testCasesPipeline

logger = Logger().get_logger()


def execute_benchmark(model: str, task: str, limit: int | None, dotenv_path: str | None, **kwargs) -> None:
    """
    Run the Geobenchmark evaluation suite.
    Args:
        model: The model to evaluate.
        task: The task to evaluate.
        limit: The total number of test cases to generate.
    """
    load_dotenv(dotenv_path=dotenv_path)
    key = os.environ.get("OPENAI_API_KEY")
    logger.debug(key)

    command = kwargs.get("command")
    if command == "test_cases":
        pipeline = testCasesPipeline(model, task, limit)
        pipeline.run(**kwargs)
    else:
        test_pipeline = evaluationPipeline(model, task, limit)
        test_pipeline.run(**kwargs)


def cli() -> None:
    """Main entry point for the gas command line interface."""
    parser = argparse.ArgumentParser(
        description="GAS: A command line tool for running the Geobenchmark evaluation suite.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        metavar="<MODEL>",
        dest="model",
        help="Model to evaluate.\nChoices: [%(choices)s]\n\n",
        choices=models,
        default="Llama3_8B",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        dest="task",
        metavar="<TASK>",
        help="Task to evaluate.\nChoices: [%(choices)s]\nDefault: %(default)s\n\n",
        choices=["NOUN", "CHOICE", "COMPLETION", "TF", "QA"],
        default="TF",
    )
    parser.add_argument(
        "-l",
        "--test-limit",
        type=int,
        dest="limit",
        metavar="<LIMIT>",
        help=(
            "Total number of test cases for evaluation.\n"
            "Type: %(type)s\nDefault: generates all test cases available\n\n"
        ),
        default=None,
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        type=str,
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="<LEVEL>",
        help="Set the logging level.\nChoices: [%(choices)s]\nDefault: %(default)s\n\n",
        default="INFO",
    )
    parser.add_argument(
        "-c",
        "--cache_dir",
        type=str,
        dest="cache_dir",
        help="cache directory to keep downloaded models from huggingFace\nDefault: %(default)s\n\n",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--path-cache",
        type=str,
        dest="path",
        help="Path to the model checkpoint.\nDefault: %(default)s\n\n",
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        dest="verbose",
        help="More logs will be added. Only works if log-level is DEBUG\nDefault: %(default)s\n\n",
        default=False,
    )
    parser.add_argument(
        "-e",
        "--dot-env",
        type=str,
        dest="dotenv_path",
        help="dot env path\nDefault: %(default)s\n\n",
        default=None,
    )
    parser.add_argument(
        "-cm",
        "--command",
        type=str,
        dest="command",
        choices=["test_cases", "evaluate"],
        metavar="<COMMAND>",
        help="Command to generate test cases or evaluate.\nChoices: [%(choices)s]\nDefault: %(default)s\n\n",
        default="test_cases",
        required=True,
    )
    parser.add_argument(
        "-tp",
        "--tests-path",
        type=str,
        dest="directory_full_path",
        help="path to tests cases to be stored or consumed\nDefault: %(default)s\n\n",
        default=None,
    )
    parser.print_help = lambda: Logger.print_help(parser)  # type: ignore[method-assign, misc, assignment]
    parser.set_defaults(func=execute_benchmark)
    try:
        args = parser.parse_args()
        if hasattr(args, "func"):
            args_dict = vars(args).copy()
            Logger().set_level(args_dict.pop("log_level"), args_dict.pop("verbose"))
            args_dict.pop("func", None)
            args.func(**args_dict)
        else:
            Logger.print_help(parser)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    cli()
