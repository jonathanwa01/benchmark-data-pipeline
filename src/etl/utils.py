from pathlib import Path


def get_raw_path(
    dataset_name: str,
    split: str = "train",
    raw_dir: Path | None = None,
) -> Path:
    """
    Return path to save/load raw dataset parquet for a given split.

    Args:
        dataset_name: Hugging Face dataset name or slug.
        split: Dataset split (e.g. 'train', 'validation').
        raw_dir: Base directory for raw data. Defaults to 'data/raw'.
    """
    base = Path(raw_dir or Path("data/raw"))
    subfolder = base / split
    subfolder.mkdir(parents=True, exist_ok=True)
    filename = dataset_name.replace("/", "_") + "_raw.pq"
    return subfolder / filename


def get_processed_path(
    dataset_name: str,
    split: str = "train",
    suffix: str = "processed",
    processed_dir: Path | None = None,
) -> Path:
    """
    Return path to save/load processed dataset parquet for a given split.

    Args:
        dataset_name: Hugging Face dataset name or slug.
        split: Dataset split (e.g. 'train', 'validation').
        suffix: File suffix, defaults to 'processed'.
        processed_dir: Base directory for processed data. Defaults to 'data/processed'.
    """
    base = Path(processed_dir or Path("data/processed"))
    subfolder = base / split
    subfolder.mkdir(parents=True, exist_ok=True)
    filename = dataset_name.replace("/", "_") + f"_{suffix}.pq"
    return subfolder / filename


__all__ = [
    "get_raw_path",
    "get_processed_path",
]
