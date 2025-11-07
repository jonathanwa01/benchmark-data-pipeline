# 🧪 Benchmark Data Pipeline

A simple, modular **ETL pipeline** built in Python that downloads datasets from [🤗 Hugging Face Datasets](https://huggingface.co/datasets), processes them with **pandas**, and stores both **raw** and **processed** versions as Parquet files.  

Includes a **Streamlit dashboard** for interactive exploration and visualization of processed data.

---

## 🚀 Features

- 📦 Download datasets directly from Hugging Face (`datasets` library)
- 🧹 Clean, normalize, and z-score numerical columns automatically
- 🧠 Store raw and processed data locally (`data/raw` and `data/processed`)
- 📊 Visualize and explore datasets in a Streamlit dashboard
- ⚙️ Configurable, reusable ETL pipeline structure

---

## ⚙️ Prerequisites

- **Python 3.10+** installed  
- **curl** (for installing UV)
- macOS, Linux, or Windows (via WSL or PowerShell)

---

## 1️⃣ Install [UV](https://github.com/astral-sh/uv)

UV is a next-generation, Rust-based Python package and environment manager — a faster, modern alternative to `pip` and `venv`.

You can install UV globally using `curl`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal and confirm installation:
```bash
uv --version
```

🧰 2️⃣ Create and sync your development environment

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/benchmark-data-pipeline.git
cd benchmark-data-pipeline
```

Then use UV to set up a virtual environment and install all dependencies:

```bash
uv sync
```

💡 This will:

Create a .venv in your project directory

Install all required libraries defined in pyproject.toml

Handle Python versioning automatically

---

🧩 3️⃣ Install your local package (editable mode)

Because your project uses a src/ layout (src/etl/), you need to install it so that the etl package is importable:


```bash
uv pip install -e .
```

That registers your local package in editable mode, meaning any code changes inside src/etl/ take effect immediately — no reinstall needed.


```bash
uv pip show benchmark-data-pipeline
```

Expected output:

```bash
Name: benchmark-data-pipeline
Editable project location: /path/to/benchmark-data-pipeline
```

---

🧪 4️⃣ Run the ETL pipeline

Now you can run the ETL module directly:

```bash
uv run -m etl
```

This will:

Download the default Hugging Face dataset (rajpurkar/squad)

Clean and process it

Save both raw and processed data in data/raw/ and data/processed/

You can also specify your own dataset and split:

```bash
uv run -m etl --dataset scikit-learn/iris --split train
```

--- 

🖥️ 5️⃣ Run the Streamlit Dashboard

After you’ve processed a dataset, launch the dashboard:

```bash
uv run streamlit run src/app.py
```

This provides:

Interactive dataset preview

Schema inspection

Download links for processed data

Visualization of numerical distributions and correlations

--- 


🧹 6️⃣ Code quality & checks (optional)

Pre-commit hooks are included for:

Ruff (lint + formatting)

Mypy (type checking)

Deptry (dependency hygiene)

```bash
uv run pre-commit install
pre-commit run --all-files
```