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


def _example_plot(df: pd.DataFrame) -> None:
    """
    Schema-agnostic example plot:
      - If numeric columns exist: histogram of the first numeric column.
      - Else if categorical exists: top-20 frequency bar chart of the first categorical column.
      - Else: show a friendly message.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Sample for speed with large datasets
    dfp = df
    if len(df) > 50_000:
        dfp = df.sample(50_000, random_state=42)

    if num_cols:
        col = num_cols[0]
        st.markdown(f"**Numeric distribution** — `{col}`")
        bins = st.slider("Bins", 10, 100, 30, key=f"bins_{col}")
        chart = (
            alt.Chart(dfp)
            .mark_bar()
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins)),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip(f"{col}:Q", format=".3f"), "count()"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        return

    if cat_cols:
        col = cat_cols[0]
        st.markdown(f"**Category frequency** — `{col}` (top 20)")
        freq = (
            dfp[col]
            .astype("object")
            .value_counts(dropna=False)
            .reset_index()
            .rename(columns={"index": col, col: "count"})
            .head(20)
        )
        chart = (
            alt.Chart(freq)
            .mark_bar()
            .encode(
                x=alt.X("count:Q"),
                y=alt.Y(f"{col}:N", sort="-x"),
                tooltip=[f"{col}:N", "count:Q"],
            )
            .properties(height=min(30 * len(freq), 600))
        )
        st.altair_chart(chart, use_container_width=True)
        return

    st.info("No numeric or categorical columns detected to plot.")


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
            _example_plot(df_sel)
        except Exception as e:
            st.exception(e)
