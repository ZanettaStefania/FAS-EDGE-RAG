# First Aid Question Answering with RAG and Small Language Models

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system based on Small Language Models (SLMs) designed to answer first aid related questions. The system retrieves relevant information from a curated First Aid Knowledge Base and generates answers with Qwen2.5 and Phi-3 models suitable for edge devices. Answers are automatically evaluated with [Ragas](https://docs.ragas.io/).

## Knowledge Base File

The knowledge base is formed by the following files (stored in `PDF_files/`):

1. Manual of St. John Ambulance Canada
   - <https://sja.ca/en/digital_first_aid_reference_guide_preview>
2. Study Material of the National CPR Association
   - <https://www.nationalcprassociation.com/free-cpr-study-guide>
3. First Aid Reference Notes of Actual First Aid Training
   - <https://actualfirstaid.com/first-aid-notes--feedback.html>

## File Description

| File / Folder | Description |
|---|---|
| `main.py` | Entry point for the general RAG pipeline (embedding `BAAI/bge-small-en-v1.5`) |
| `main_med.py` | Entry point for the medical RAG pipeline (embedding `abhinand/MedEmbed-small-v0.1`) |
| `new_config_main_med.py` | Medical pipeline with `max_new_tokens=500` and `temperature=0.5` |
| `launcher.py` | Runs every model × pipeline × dataset combination sequentially |
| `ragas_evaluation.py` | Ragas evaluation using `medgemma:27b` as judge via Ollama |
| `results_io.py` | Reading and writing of result files |
| `retrieval_logging.py` | Logging of retrieval results |
| `utils.py` / `utils_med.py` | Utility functions used by the general / medical pipeline |
| `config.py` / `config_med.py` | Configuration parameters of the general / medical pipeline |
| `prompt_config.py` | Prompt templates used for the language models |
| `imports.py` | Centralized imports |
| `DB/` | Chroma vector databases (chunk size 500/1000/1500, general and `_med`) and the `new_create.py` build script |
| `EvalSet/` | Evaluation datasets (`Question`, `ground_truths` columns) |
| `EvalResults/` | Generation and retrieval results already produced |
| `PDF_files/` | Source documents used to build the knowledge base |

## Requirements

- Python **3.10 or later**
- NVIDIA GPU with CUDA (models are loaded with `device_map='cuda'`)
- [Ollama](https://ollama.com/) running on `http://localhost:11434` with the judge model `medgemma:27b` (only needed for Ragas evaluation, i.e. when using `--test` or the launcher)
- Internet access on first run, to download models from Hugging Face

## Installation

Install the dependencies from **`requirements.txt`**.

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) Install PyTorch with the CUDA build matching your GPU first (see https://pytorch.org/get-started/locally/), e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) Then install the remaining dependencies
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check that the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Ragas judge (Ollama)

```powershell
ollama pull medgemma:27b
ollama serve   # if it is not already running as a service
```

The model and host can be changed via `JUDGE_MODEL` and `OLLAMA_HOST` in `ragas_evaluation.py`.

### Vector databases

The databases in `DB/` use the legacy Chroma format (`chroma-*.parquet` files + `index/`), created with chromadb 0.3.x. With `chromadb==1.5.9` from `requirements.txt` they must be migrated or rebuilt:

- **Rebuild** (recommended): in `DB/new_create.py` set `chunk_size`, the embedding model and `persist_directory`, then run:
  ```powershell
  cd DB
  python new_create.py
  cd ..
  ```
- **Migrate** with the official Chroma tool: `pip install chroma-migrate`, then run `chroma-migrate`.

The database used by each pipeline is set by `PERSIST_DIRECTORY_DB` in `config.py` (default `DB/new1500_db`) and in `config_med.py` (default `DB/new1000_db_med`).

## Supported models

`--model_name` takes the model's **short name**. The Hugging Face prefix (`Qwen/` or `microsoft/`) is added automatically:

- `Qwen2.5-0.5B-Instruct`
- `Qwen2.5-1.5B-Instruct`
- `Qwen2.5-3B-Instruct`
- `Phi-3.5-mini-instruct`
- `Phi-3-mini-128k-instruct`

To use models already downloaded locally, set `model_path` in `prompt_config.py` to a folder containing the `Qwen/` and `microsoft/` subfolders.

## Running a single pipeline

Always run commands from the project root.

```powershell
# Interactive mode (type "exit" to quit)
python main.py --model_name Qwen2.5-0.5B-Instruct

# On a dataset: the file name is also looked up in EvalSet/; Ragas evaluation runs at the end
python main.py --model_name Qwen2.5-0.5B-Instruct --test CPR_open.csv
python main_med.py --model_name Phi-3.5-mini-instruct --test First_Aid_open.csv
```

Output is written to `CSV_Result_med/`, or to the folder set by the `THESIS_RAG_OUTPUT_DIR` environment variable.

## Using the launcher

`launcher.py` runs every **model × pipeline × dataset** combination in a separate process, so GPU memory is released between experiments. Options:

| Option | Default | Description |
|---|---|---|
| `--models` | all 5 models | Short Qwen/Phi names (not paths) |
| `--pipelines` | `main main_med new_config_main_med` | Pipelines to run |
| `--datasets` | every CSV in `EvalSet/` | CSV file names (name only, no path) |
| `--dry-run` | — | Print the plan without running anything |
| `--stop-on-error` | — | Stop at the first failed experiment |

### Examples

**1. Check the plan without running anything (recommended first step):**

```powershell
python launcher.py --dry-run
```

**2. Quick test: one small model, one pipeline, one dataset:**

```powershell
python launcher.py --models Qwen2.5-0.5B-Instruct --pipelines main --datasets CPR_open.csv
```

**3. Compare the general and medical pipelines on the same model:**

```powershell
python launcher.py --models Qwen2.5-1.5B-Instruct --pipelines main main_med --datasets First_Aid_open.csv Infant_CPR_open.csv
```

**4. Both Phi models on all datasets, stopping at the first error:**

```powershell
python launcher.py --models Phi-3.5-mini-instruct Phi-3-mini-128k-instruct --pipelines main_med --stop-on-error
```

**5. Full experiment (5 models × 3 pipelines × all datasets; takes a long time):**

```powershell
python launcher.py
```

### Launcher output

Each run creates a timestamped folder:

```
CSV_Result_med/experiments/<timestamp>/
├── run.json                          # summary: status, timing and errors for each experiment
└── <model>/<pipeline>/
    ├── <db><model><suffix>_<dataset>.json   # answers, contexts and Ragas scores
    └── <dataset>.log                        # full process log
```

Exit codes: `0` if everything succeeded, `1` if at least one experiment failed, `130` if interrupted with Ctrl+C.

## Troubleshooting

- **`Database directory missing`**: the path in `config.py` / `config_med.py` does not exist. Check `PERSIST_DIRECTORY_DB`.
- **No context retrieved / generic answers**: the database is in the legacy format. Rebuild or migrate it (see [Vector databases](#vector-databases)).
- **`Ragas evaluation did not finish`**: Ollama is not running or `medgemma:27b` has not been pulled. Check with `ollama list`.
- **CUDA out of memory**: use smaller models (`Qwen2.5-0.5B-Instruct`) or run one experiment at a time.
- **An experiment fails**: read the `.log` file referenced by the `log_file` field in `run.json`.
