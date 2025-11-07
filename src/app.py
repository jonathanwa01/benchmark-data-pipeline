from __future__ import annotations

import shutil
import time
from pathlib import Path
import altair as alt

import pandas as pd
import streamlit as st

from etl.etl_pipeline import (
    PROCESSED_DIR,
    RAW_DIR,
    download_hf_dataset,
    get_processed_path,
    get_raw_path,
)

st.set_page_config(
    page_title="HF Dataset ETL Demo",
    page_icon="🧪",
    layout="wide",
)

# ---- Sidebar controls -----------------------------------------------------
st.sidebar.header("⚙️ Controls")


DEFAUT_DATASET = "scikit-learn/iris"


with st.sidebar:
    dataset_name = st.text_input(
        "Hugging Face dataset ID",
        value=DEFAUT_DATASET,
        placeholder="e.g. scikit-learn/adult-census-income",
        help="Use the full dataset slug as on huggingface.co/datasets/...",
    )

    split = st.text_input(
        "Split",
        value="train",
        help="If the dataset is a DatasetDict, provide a valid split name (e.g., train, validation, test).",
    )

    colb1, colb2 = st.columns(2)
    with colb1:
        run_btn = st.button(
            "▶️ Process dataset", use_container_width=True, type="primary"
        )
    with colb2:
        clean_cache_btn = st.button("🧹 Clear data folder", use_container_width=True)

    st.divider()
    if dataset_name.strip():
        st.markdown(
            f"Open on 🤗: [huggingface.co/datasets/{dataset_name}](https://huggingface.co/datasets/{dataset_name})"
        )

# ---- Main layout ----------------------------------------------------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.title("🧪 ETL Dashboard — Hugging Face → Pandas → Parquet")
    st.caption(
        "Enter a HF dataset slug, choose a split, then use **Process** to ensure the raw file exists and load the processed file.\n"
        "Use **Clear data folder** to remove all cached data (data/raw and data/processed)."
    )
    placeholder_status = st.empty()
    placeholder_tabs = st.empty()

with right:
    st.subheader("Paths")
    if dataset_name.strip():
        raw_path = get_raw_path(dataset_name, split=split or "train", raw_dir=RAW_DIR)
        processed_path = get_processed_path(
            dataset_name, split=split or "train", processed_dir=PROCESSED_DIR
        )
        st.code(f"Raw:       {raw_path}", language="bash")
        st.code(f"Processed: {processed_path}", language="bash")
    else:
        st.info("Enter a dataset slug to see resolved paths.")

# ---- Helpers --------------------------------------------------------------


def _ensure_raw(dataset: str, split: str) -> Path:
    raw_path = get_raw_path(dataset, split=split, raw_dir=RAW_DIR)
    if Path(raw_path).exists():
        st.info(f"Found existing raw parquet. Loading from disk: {raw_path}")
        return raw_path
    with st.spinner("Downloading raw dataset and saving Parquet…"):
        download_hf_dataset(dataset, split=split, save_path=raw_path)
    return raw_path


def _build_processed_from_raw(dataset: str, split: str) -> pd.DataFrame:
    """
    Build and persist a processed DataFrame from a raw parquet file for a given dataset and split.

    It cleans the data and transforms it into a processed DataFrame before persisting it to disk.

    Args:
        dataset (str): HF dataslug.
        split (str): Data split to use (e.g., "train", "validation", "test"). If falsy, "train" is used by default.

    Returns:
        pd.DataFrame: The processed DataFrame that was written to the processed parquet path.

    Raises:
        FileNotFoundError: If the raw parquet file does not exist at the resolved raw path.
    """
    raw_path = get_raw_path(dataset, split=split or "train", raw_dir=RAW_DIR)
    processed_path = get_processed_path(
        dataset, split=split or "train", processed_dir=PROCESSED_DIR
    )
    if not Path(raw_path).exists():
        raise FileNotFoundError(
            f"Raw parquet not found at {raw_path}. Click 'Process dataset' first to download it."
        )
    from etl.etl_pipeline import clean_dataset, transform_dataset

    df_raw = pd.read_parquet(raw_path)
    df_clean = clean_dataset(df_raw.copy())
    df_proc = transform_dataset(df_clean)
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_parquet(processed_path, index=False)
    return df_proc


def _load_or_build_processed(dataset: str, split: str) -> pd.DataFrame:
    """
    Return processed DataFrame for a dataset split. Load existing processed file or build from raw.

    Args:
        dataset (str): Dataset identifier.
        split (str): Split name (defaults to "train" when falsy).

    Returns:
        pd.DataFrame: The processed DataFrame for the requested split.
    """
    processed_path = get_processed_path(
        dataset, split=split or "train", processed_dir=PROCESSED_DIR
    )
    if Path(processed_path).exists():
        return pd.read_parquet(processed_path)
    return _build_processed_from_raw(dataset, split)


# ---- Actions --------------------------------------------------------------
if run_btn:
    if not dataset_name.strip():
        st.error("Please enter a valid dataset slug (e.g. `scikit-learn/iris`).")
        st.stop()

    start = time.time()
    raw_path = _ensure_raw(dataset_name, split)

    with st.spinner("Loading processed dataset (or building from raw if missing)…"):
        try:
            df_processed = _load_or_build_processed(dataset_name, split)
        except Exception as e:
            st.exception(e)
            st.stop()

    total = time.time() - start

    with placeholder_status.container():
        st.success("Pipeline ready!")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Processed rows", f"{len(df_processed):,}")
        with m2:
            st.metric("Processed cols", f"{df_processed.shape[1]:,}")
        with m3:
            st.metric("Elapsed", f"{total:0.2f}s")

    with placeholder_tabs:
        tab1, tab2, tab3 = st.tabs(["Preview", "Schema", "Download files"])

        with tab1:
            st.dataframe(df_processed.head(100), use_container_width=True)

        with tab2:
            info = pd.DataFrame(
                {
                    "column": df_processed.columns,
                    "dtype": [str(dt) for dt in df_processed.dtypes],
                    "non_null_count": [
                        int(df_processed[c].notna().sum()) for c in df_processed.columns
                    ],
                }
            )
            st.dataframe(info, use_container_width=True)

        with tab3:
            raw_path = get_raw_path(
                dataset_name, split=split or "train", raw_dir=RAW_DIR
            )
            processed_path = get_processed_path(
                dataset_name, split=split or "train", processed_dir=PROCESSED_DIR
            )

            if Path(raw_path).exists():
                st.download_button(
                    label="⬇️ Download RAW parquet",
                    data=Path(raw_path).read_bytes(),
                    file_name=Path(raw_path).name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                st.warning("Raw file not found on disk.")

            if Path(processed_path).exists():
                st.download_button(
                    label="⬇️ Download PROCESSED parquet",
                    data=Path(processed_path).read_bytes(),
                    file_name=Path(processed_path).name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                st.warning("Processed file not found on disk.")

if clean_cache_btn:
    data_dir = Path("data")
    if data_dir.exists():
        with st.spinner("Clearing all cached data (data/raw and data/processed)…"):
            try:
                shutil.rmtree(data_dir)
                st.success("Data folder removed successfully!")
            except Exception as e:
                st.exception(e)
    else:
        st.info("No data folder found to remove.")

# ---- Helpful footer -------------------------------------------------------
st.divider()
st.caption(
    "Tip: **Process dataset** will reuse an existing raw parquet from `data/raw/` if present,"
    " otherwise it downloads it. **Clear data folder** deletes the entire `data/` cache."
)


# =========================
# Browse existing processed datasets + example plot
# =========================


st.divider()
st.header("📂 Browse processed datasets")


def _discover_processed() -> list[dict[str, str | Path]]:
    """
    Scan data/processed/{split}/*.pq and return a list of entries.
    """
    entries: list[dict[str, str | Path]] = []
    if not PROCESSED_DIR.exists():
        return entries

    for split_dir in sorted([p for p in PROCESSED_DIR.iterdir() if p.is_dir()]):
        split = split_dir.name
        for pq in sorted(split_dir.glob("*.pq")):
            stem = pq.stem
            # Try to split into <dataset_key>_<suffix>
            if "_" in stem:
                dataset_key, suffix = stem.rsplit("_", 1)
            else:
                dataset_key, suffix = stem, ""
            label = f"{dataset_key} [{split}]{' · ' + suffix if suffix else ''}"
            entries.append(
                {
                    "label": label,
                    "split": split,
                    "dataset_key": dataset_key,
                    "suffix": suffix,
                    "path": pq,
                }
            )
    return entries


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot a correlation heatmap between all standardized (z-score) numeric features.

    Each cell shows the Pearson correlation coefficient between two features:
        - +1 → features increase together (strong positive relationship)
        - -1 → one increases as the other decreases (strong negative relationship)
        - 0  → no linear relationship

    Interpretation:
        - Bright red = strong positive correlation (features redundant or similar)
        - Bright blue = strong negative correlation (inverse relationship)
        - White/neutral = weak or no correlation

    Insight:
        - Reveals multicollinearity among features.
        - Identifies clusters of correlated variables (latent groups).
        - Highlights redundant engineered columns (e.g., norm vs zscore).

    Args:
        df (pd.DataFrame): Dataset containing z-score columns (columns ending with "_zscore").
    """
    z_cols = [c for c in df.columns if c.endswith("_zscore")]
    if len(z_cols) < 2:
        st.info("Need at least two z-scored columns for correlation heatmap.")
        return

    corr = df[z_cols].corr()
    corr_long = corr.stack().reset_index()
    corr_long.columns = ["x", "y", "corr"]

    chart = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("x:N", sort=z_cols, title=None),
            y=alt.Y("y:N", sort=z_cols, title=None),
            color=alt.Color(
                "corr:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1))
            ),
            tooltip=["x", "y", alt.Tooltip("corr:Q", format=".2f")],
        )
        .properties(title="Correlation heatmap (z-score features)")
    )
    st.altair_chart(chart, use_container_width=True)


def plot_zscore_distributions(df: pd.DataFrame) -> None:
    """
    Plot overlaid density curves of all standardized (z-score) numeric features.

    Each line represents the distribution of a feature's z-scores across all samples.

    Interpretation:
        - Because all features are standardized (mean≈0, std≈1), distributions are comparable.
        - The closer the curves overlap around 0, the more similar the features’ scales and spread.
        - Wide curves indicate more variability; narrow curves show low variance.
        - Skewed or multi-peaked lines suggest non-normal or heterogeneous distributions.

    Insight:
        - Confirms successful standardization and helps detect features with abnormal variance.
        - Highlights skewness, multimodality, or potential outliers even after scaling.
        - Useful for sanity-checking numeric features before modeling.

    Args:
        df (pd.DataFrame): Dataset containing z-score columns (columns ending with "_zscore").
    """
    z_cols = [c for c in df.columns if c.endswith("_zscore")]
    if not z_cols:
        st.info("No z-score columns found.")
        return

    melted = df[z_cols].melt(var_name="feature", value_name="zscore")

    chart = (
        alt.Chart(melted)
        .transform_density(
            "zscore",
            groupby=["feature"],
            as_=["zscore", "density"],
        )
        .mark_line()
        .encode(
            x=alt.X("zscore:Q"),
            y=alt.Y("density:Q"),
            color="feature:N",
        )
        .properties(title="Distribution of standardized (z-score) features")
    )
    st.altair_chart(chart, use_container_width=True)


# UI: selector + load button
processed_entries = _discover_processed()
if not processed_entries:
    st.info(
        "No processed datasets found yet. Run **Process dataset** above to create some."
    )
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        choice = st.selectbox(
            "Select a processed dataset",
            options=processed_entries,
            format_func=lambda e: e["label"],
            key="processed_selector",
        )
    with col2:
        load_processed_btn = st.button("Load selection", use_container_width=True)

    if load_processed_btn:
        try:
            df_sel = pd.read_parquet(choice["path"])
            st.success(f"Loaded: `{choice['label']}` — shape {df_sel.shape}")
            # quick preview
            with st.expander("Preview (head)"):
                st.dataframe(df_sel.head(200), use_container_width=True)
            # example plot
            plot_correlation_heatmap(df_sel)
            plot_zscore_distributions(df_sel)
        except Exception as e:
            st.exception(e)
