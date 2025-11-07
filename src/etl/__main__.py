import argparse
from etl.etl_pipeline import main


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run the ETL pipeline.")
    parser.add_argument("--dataset", "-d", default="rajpurkar/squad")
    parser.add_argument("--split", "-s", default="validation")
    args = parser.parse_args()

    main(args.dataset, args.split)


if __name__ == "__main__":
    cli()
