import argparse
import os
from typing import Any

import toml
from dotenv import load_dotenv

from gas.logger import Logger
from gas.models.base import BaseModel
from gas.pipelines import EvaluationPipeline, GenerationPipeline

logging = Logger()
logger = logging.get_logger()


def show_info(
    model: BaseModel, task: str, seed: int | None, generation_params: dict[str, Any], limit: int | None = None, **kwargs
):
    info = {
        "Geoscience benchmark task:": task,
        "Model for evaluation:": model.get_model_name(),
        "LOG LEVEL:": f"{logging.get_level()}",
        "Tests limit:": limit,
        "SEED: ": seed,
        **generation_params,
    }
    logging.print_table_panel(
        info,
        title="[bold medium_orchid]Benchmark General Information[/bold medium_orchid]",
        border_style="purple",
        expand=False,
        justify="center",
    )


def generate(model: BaseModel, task: str, limit: int | None, test_cases_path: str, **kwargs):
    GenerationPipeline(model=model, task=task, test_cases_path=test_cases_path, limit=limit, **kwargs).run()


def evaluate(model, task: str, model_judge: str, test_cases_path: str, results_path: str, **kwargs):
    EvaluationPipeline(
        model=model,
        task=task,
        test_cases_path=test_cases_path,
        model_judge=model_judge,
        results_path=results_path,
        **kwargs,
    ).run()


def make_shared_commands(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        dest="task",
        metavar="<TASK>",
        help="Task to evaluate.\nChoices: [%(choices)s]\n\n",
        choices=["NOUN", "CHOICE", "COMPLETION", "TF", "QA"],
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
        "--config",
        type=str,
        dest="config_path",
        metavar="<PATH>",
        help="Path to config file.\nDefault: %(default)s\n\n",
        default=None,
    )
    return parser


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GAS: A command line tool for running the Geobenchmark evaluation suite.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = make_shared_commands(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)

    parser_generate = subparsers.add_parser(
        "generate",
        help="Generate the test cases to evaluate.",
        description="Generate test cases for a provided model.",
        parents=[parent_parser],
    )
    parser_generate.add_argument(
        "-l",
        "--limit",
        type=int,
        dest="limit",
        metavar="<LIMIT>",
        help=(
            "Total number of test cases to generate for evaluation.\n" "Default: generates all test cases available\n\n"
        ),
        default=None,
    )
    parser_evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate the generated test cases.",
        description="Evaluate the generated test cases.",
        parents=[parent_parser],
    )
    parser.print_help = lambda: Logger.print_help(parser)  # type: ignore[method-assign, misc, assignment]
    parser_evaluate.print_help = lambda: Logger.print_help(parser_evaluate)  # type: ignore[method-assign, misc, assignment]
    parser_generate.print_help = lambda: Logger.print_help(parser_generate)  # type: ignore[method-assign, misc, assignment]
    return parser


def cli() -> None:
    """Main entry point for the gas command line interface."""
    parser = build_argparser()

    args = parser.parse_args()
    args_dict = vars(args).copy()
    logging.set_level(args_dict.pop("log_level"))

    config_path = args_dict.get("config_path", "./config.toml")
    config = toml.load(config_path)
    tc_path = config["environment"]["test_cases_path"]
    dotenv_path = config["environment"].get("dotenv_path", None)
    load_dotenv(dotenv_path=dotenv_path) if dotenv_path else load_dotenv()

    key = os.environ.get("OPENAI_API_KEY")
    logger.debug(key)

    generation_params = config["generation"]
    model_params = config["model"]
    seed = config["environment"].get("seed", None)
    model = BaseModel(model_params, generation_params, seed=seed)

    show_info(model=model, seed=seed, generation_params=generation_params, **args_dict)
    match args.command:
        case "generate":
            generate(model=model, test_cases_path=tc_path, **args_dict)
        case "evaluate":
            evaluation_params = config["evaluation"]
            evaluate(model=model, test_cases_path=tc_path, **evaluation_params, **args_dict)
        case _:
            parser.print_help()


if __name__ == "__main__":
    cli()
