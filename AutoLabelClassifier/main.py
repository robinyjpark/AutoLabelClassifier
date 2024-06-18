"""
Implements argparser
"""

import importlib
import argparse


def import_and_run_func(filename):
    """
    Wrapper around a function that imports it and then runs it -
    this makes the cli much faster since it doesn't need to load
    in all the relevant modules for the scripts just to run
    commands that only need a specific subset e.g. -h.
    """
    # load in main function from file

    def func(args):
        module = importlib.import_module(filename)
        # remove command from args
        del args.command
        del args.func
        return module.main(**vars(args))

    return func


def parser():
    from AutoLabelClassifier import __version__

    parser = argparse.ArgumentParser("auto_label_classifier")
    subparsers = parser.add_subparsers(
        title="subcommands", description="", help="additional help", dest="command"
    )
    subparsers.required = True

    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s {}".format(__version__)
    )

    label_parser = subparsers.add_parser(
        "label",
        help="Label reports using LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    label_parser.set_defaults(func=import_and_run_func("AutoLabelClassifier.label"))
    label_parser.add_argument(
        "--condition", type=str, required=True, help="The condition to label for"
    )
    label_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="The data to label, this should be a csv file with a column called report that contains the text to label and a column called pat_id which has the patient identifier. See example_reports.csv for an example of the format",
    )
    label_parser.add_argument(
        "--definition",
        type=str,
        required=True,
        help="A definition of the condition to label for, this should be a string that describes the condition.",
    )
    label_parser.add_argument(
        "--output", type=str, required=True, help="The path to save the output to"
    )
    label_parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
        help="Name of the model to use, this should be a HuggingFace model (see https://huggingface.co/docs/transformers/en/main_classes/model for more details)",
    )
    label_parser.add_argument(
        "--transformers_cache",
        type=str,
        help="Path to the transformers cache",
        default="~/.cache",
    )
    label_parser.add_argument(
        "--device", type=str, help="Device to run the model on", default="cuda:0"
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Prepare reports for labelling format. This is not yet implemented.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    preprocess_parser.set_defaults(
        func=import_and_run_func("AutoLabelClassifier.preprocess")
    )

    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Summary finetune existing LLM on an existing corpus of text. This is not yet implemented",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    finetune_parser.set_defaults(
        func=import_and_run_func("AutoLabelClassifier.finetune")
    )

    return parser


def main():
    this_parser = parser()
    args = this_parser.parse_args()
    args.func(args)
