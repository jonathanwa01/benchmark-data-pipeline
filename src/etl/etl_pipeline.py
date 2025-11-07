import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from typing import Iterator

from etl.utils import get_processed_path, get_raw_path

# Base directories
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def download_hf_dataset(
    dataset_name: str, split: str = "train", save_path: Path | None = None
) -> pd.DataFrame:
    """
    Downloads a dataset from Hugging Face, converts it to pandas, and saves as parquet.

    Args:
        dataset_name (str): Hugging Face dataset ID (e.g., 'scikit-learn/adult-census-income').
        split (str): Dataset split to use if a DatasetDict. Defaults to 'train'.
        save_path (Path | None): Optional path to save parquet. Defaults to data/raw/{dataset_name}_raw.pq.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.

    Raises:
        ValueError: If requested split is missing in a DatasetDict.
        TypeError: If dataset is streaming (IterableDataset / IterableDatasetDict).
    """
    logger.info(f"Downloading dataset '{dataset_name}' from Hugging Face...")
    ds = load_dataset(dataset_name)

    if isinstance(ds, DatasetDict):
        logger.info(f"DatasetDict detected. Using split '{split}'.")
        if split not in ds:
            raise ValueError(
                f"Split '{split}' not found. Available splits: {list(ds.keys())}"
            )
        ds_sel = ds[split]
        df_or_iter = ds_sel.to_pandas()

    elif isinstance(ds, Dataset):
        logger.info("Dataset detected. Converting directly to pandas.")
        df_or_iter = ds.to_pandas()
    else:
        logger.error(
            f"Unsupported dataset type: {type(ds)}. Streaming datasets cannot be converted to pandas."
        )
        raise TypeError(
            f"Cannot convert object of type {type(ds)} to pandas DataFrame."
        )

    # Handle iterator if returned
    if isinstance(df_or_iter, Iterator):
        logger.info("Iterator detected. Concatenating chunks into a single DataFrame.")
        df = pd.concat(list(df_or_iter), ignore_index=True)
    else:
        df = df_or_iter

    if save_path is None:
        save_path = get_raw_path(dataset_name, split=split, raw_dir=RAW_DIR)

    df.to_parquet(save_path, index=False)
    logger.info(f"Raw dataset saved to {save_path} with shape {df.shape}")
    return df


def drop_duplicates_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops duplicates while ignoring columns with unhashable types.
    """
    hashable_cols = [
        c
        for c in df.columns
        if df[c]
        .apply(lambda x: isinstance(x, (int, float, str, bool)) or pd.isna(x))
        .all()
    ]
    return df.drop_duplicates(subset=hashable_cols)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dataset: drop duplicates/nulls, trim strings, normalize numerics.

    Returns:
        pd.DataFrame: Cleaned dataset (in memory, not saved).
    """
    logger.info(f"Starting cleaning dataset with shape {df.shape}")

    # Trim string columns
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].str.strip()

    # Drop duplicates and nulls
    df = df.drop_duplicates().dropna()
    logger.info(f"After dropping duplicates/nulls, shape: {df.shape}")

    logger.info("Cleaning completed")
    return df


def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example transformation: ensure target column is string and create numeric normalized features.

    Returns:
        pd.DataFrame: Transformed dataset (in memory, not saved).
    """
    logger.info(f"Starting transformation on dataset with shape {df.shape}")

    # Z-score numeric columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for c in num_cols:
        std = df[c].std()
        if std and std != 0:
            df[f"{c}_zscore"] = (df[c] - df[c].mean()) / std
        else:
            df[f"{c}_zscore"] = 0.0

    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for c in numeric_cols:
        df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

    logger.info("Transformation completed")
    return df


def load_dataset_from_file(
    dataset_name: str, split: str, stage: str = "processed"
) -> pd.DataFrame:
    """
    Loads a dataset from a Parquet file. If stage is 'processed' and the file does not exist,
    it will process the raw dataset and save it automatically.

    Args:
        dataset_name (str): The name of the dataset.
        stage (str): One of ['raw', 'processed'] specifying which version to load.
                     Defaults to 'processed'.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    stage = stage.lower()
    if stage == "raw":
        path = get_raw_path(dataset_name, split=split, raw_dir=RAW_DIR)
        if not path.exists():
            raise FileNotFoundError(f"No raw file found at {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded raw dataset from {path} with shape {df.shape}")
        return df

    elif stage == "processed":
        path = get_processed_path(
            dataset_name, split=split, processed_dir=PROCESSED_DIR
        )
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"Loaded processed dataset from {path} with shape {df.shape}")
            return df

        # If processed file does not exist, process raw dataset
        raw_path = get_raw_path(dataset_name, split=split, raw_dir=RAW_DIR)
        if not raw_path.exists():
            raise FileNotFoundError(f"No raw file found at {raw_path} to process")

        df = pd.read_parquet(raw_path)
        df = clean_dataset(df)
        df = transform_dataset(df)

        # Save processed dataset
        df.to_parquet(path, index=False)
        logger.info(f"Processed dataset saved to {path} with shape {df.shape}")
        return df

    else:
        raise ValueError("stage must be one of ['raw', 'processed']")


if __name__ == "__main__":
    dataset_name = "rajpurkar/squad"
    split = "validation"

    # Download raw dataset (saved)
    download_hf_dataset(dataset_name, split=split)

    # Load processed dataset (saved only here)
    processed_df = load_dataset_from_file(dataset_name, stage="processed", split=split)

    logger.info("ETL pipeline completed successfully!")

__all__ = [
    "download_hf_dataset",
    "drop_duplicates_safe",
    "clean_dataset",
    "transform_dataset",
    "load_dataset_from_file",
]
