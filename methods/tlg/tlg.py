import argparse

from ablation import run_ablation
from constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_EXTRACTED_FACTS_PATH, DatasetType


def main():
    parser = argparse.ArgumentParser(description="CLI for running TLG ablation studies")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ablation_parser = subparsers.add_parser(
        "run_ablation", help="Run an ablation study"
    )
    ablation_parser.add_argument(
        "--encoder-checkpoint-path",
        type=str,
        help="Path to the encoder checkpoint file",
        default=DEFAULT_CHECKPOINT_PATH,
    )
    ablation_parser.add_argument(
        "--extracted-facts-path",
        type=str,
        help="Path to the extracted facts file",
        default=DEFAULT_EXTRACTED_FACTS_PATH,
    )
    ablation_parser.add_argument(
        "--dataset-type",
        type=DatasetType,
        choices=list(DatasetType),
        help="Type of dataset: 'weird' or 'whoops'",
        default=DatasetType.WHOOPS,
    )

    args = parser.parse_args()

    if args.command == "run_ablation":
        run_ablation(
            encoder_checkpoint_path=args.encoder_checkpoint_path,
            extracted_facts_path=args.extracted_facts_path,
            dataset_type=args.dataset_type,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
