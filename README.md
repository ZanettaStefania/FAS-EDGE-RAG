# First Aid Question Answering with RAG and Small Language Models

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system based on Small Language Models (SLMs) designed to answer first aid related questions. The default pipeline retrieves passages from a curated First Aid Knowledge Base and generates answers with Qwen2.5 and Phi-3 models. Dataset runs evaluate the answers with [Ragas](https://docs.ragas.io/) by default.

## Knowledge Base File

The knowledge base uses three sources, stored in `PDF_files/`. The St. John Ambulance guide is included as two PDF files.

1. Manual of St. John Ambulance Canada
   - <https://sja.ca/en/digital_first_aid_reference_guide_preview>
2. Study Material of the National CPR Association
   - <https://www.nationalcprassociation.com/free-cpr-study-guide>
3. First Aid Reference Notes of Actual First Aid Training
   - [First Aid Training Notes (PDF)](PDF_files/Actual_First_Aid_Training_Notes.pdf)

## File Description

```text
FAS-EDGE-RAG/
├── PDF_files/
├── DB/
│   ├── new_create.py
│   ├── new500_db/
│   ├── new500_db_med/
│   ├── new1000_db/
│   ├── new1000_db_med/
│   ├── new1500_db/
│   └── new1500_db_med/
├── EvalSet/
├── EvalResults/
│   ├── Generation/
│   └── Retrieval/
└── CSV_Result_med/
    └── experiments/
```

| File / Folder | Description |
|---|---|
| `main.py` | Entry point for the general RAG pipeline (embedding `BAAI/bge-small-en-v1.5`) |
| `main_med.py` | Entry point for the medical RAG pipeline (embedding `abhinand/MedEmbed-small-v0.1`) |
| `new_config_main_med.py` | Medical pipeline with `max_new_tokens=500` and `temperature=0.5` |
| `launcher.py` | Runs every model × pipeline × dataset combination sequentially |
| `ragas_evaluation.py` | Ragas evaluation using `medgemma:27b` as judge via Ollama |
| `results_io.py` | Reading and writing of result files |
| `utils.py` / `utils_med.py` | Utility functions used by the general / medical pipeline |
| `config.py` / `config_med.py` | Configuration parameters of the general / medical pipeline |
| `prompt_config.py` | Prompt templates used for the language models |
| `imports.py` | Centralized imports |
| `DB/` | Chroma vector databases (chunk size 500/1000/1500, general and `_med`) and the `new_create.py` build script |
| `EvalSet/` | Evaluation datasets (`Question`, `ground_truths` columns) |
| `EvalResults/` | Generation and retrieval results already produced |
| `CSV_Result_med/experiments/` | Launcher outputs grouped by run, model and pipeline |
| `PDF_files/` | Source documents used to build the knowledge base |

## Requirements

- Python **3.10–3.12** for the pinned dependencies. On Python 3.13 and later, `langchain-community==0.3.31` requires NumPy 2.1 or later, while this project pins `numpy==1.26.4`.
- NVIDIA GPU with CUDA (models are loaded with `device_map='cuda'`)
- [Ollama](https://ollama.com/) running on `http://localhost:11434` with the judge model `medgemma:27b` for Ragas evaluation. Use `--skip-ragas` to skip this step.
- Internet access to download model files that are missing from the local machine

## Installation

The examples below use Python 3.12. In the activated environment, install a CUDA build of PyTorch compatible with your system before installing the remaining dependencies. Select the installation command for your platform in the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Run the selected PyTorch installation command here, then install the remaining dependencies.
python -m pip install -r requirements.txt
```

```bash
# Linux with CUDA
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Run the selected PyTorch installation command here, then install the remaining dependencies.
python -m pip install -r requirements.txt
```

Check that the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Ragas judge (Ollama)

Start Ollama if it is not already running (`ollama serve` in a separate terminal). Then download the judge model:

```powershell
ollama pull medgemma:27b
```

The model and host can be changed via `JUDGE_MODELS` and `OLLAMA_HOST` in `ragas_evaluation.py`.

### Vector databases

Each database in `DB/` includes the SQLite database and vector index used by the current pipeline, alongside the original `chroma-*.parquet` files and `index/` directory. Use the supplied databases to reproduce retrieval. Rebuilding from the PDFs with `DB/new_create.py` can change the chunks.

The database used by each pipeline is set by `PERSIST_DIRECTORY_DB` in `config.py` (default `DB/new1500_db`) and in `config_med.py` (default `DB/new1000_db_med`).

## Supported models

`--model_name` takes the model's **short name**. The Hugging Face prefix (`Qwen/` or `microsoft/`) is added automatically:

- `Qwen2.5-0.5B-Instruct`
- `Qwen2.5-1.5B-Instruct`
- `Qwen2.5-3B-Instruct`
- `Phi-3.5-mini-instruct`
- `Phi-3-mini-128k-instruct`

To use models already downloaded locally, set `model_path` in `prompt_config.py` to a folder containing the `Qwen/` and `microsoft/` subfolders. Include the trailing slash.

The Qwen prompt asks the model to support its answer with the most relevant passages and to say when it is unsure. The Phi prompt asks for a concise answer of at most three sentences and tells the model to say when it does not know. The Phi prompt for generation without retrieval keeps these instructions and supplies only the question.

## Running a single pipeline

Always run commands from the project root.

```powershell
# Interactive mode (type "exit" to quit)
python main.py --model_name Qwen2.5-0.5B-Instruct

# On a dataset: the file name is also looked up in EvalSet/; Ragas evaluation runs at the end
python main.py --model_name Qwen2.5-0.5B-Instruct --test CPR_open.csv
python main_med.py --model_name Phi-3.5-mini-instruct --test First_Aid_open.csv
```

Interactive mode prints answers to the terminal. With `--test`, the pipeline also writes the result JSON to `CSV_Result_med/`, or to the folder set by `THESIS_RAG_OUTPUT_DIR`.

## Using the launcher

`launcher.py` runs every **model × pipeline × dataset** combination in a separate process, so GPU memory is released between experiments. Options:

| Option | Default | Description |
|---|---|---|
| `--models` | all 5 models | Short Qwen/Phi names (not paths) |
| `--pipelines` | `main main_med new_config_main_med` | Pipelines to run |
| `--datasets` | every CSV in `EvalSet/` | CSV file names (name only, no path) |
| `--dry-run` | — | Print the plan without running anything |
| `--stop-on-error` | — | Stop at the first failed experiment |
| `--no-rerank` | Off | Use up to two passages with the highest similarity scores that meet the 0.6 threshold |
| `--no-retrieval` | Off | Answer without retrieved context. Requires a Phi model and `--pipelines main_med` |
| `--skip-ragas` | Off | Save answers and timings and skip Ragas evaluation |

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

**6. Compare retrieval modes on `Final.csv` using the same model and pipeline:**

```powershell
python launcher.py --models Phi-3.5-mini-instruct --pipelines main_med --datasets Final.csv
python launcher.py --models Phi-3.5-mini-instruct --pipelines main_med --datasets Final.csv --no-rerank
python launcher.py --models Phi-3.5-mini-instruct --pipelines main_med --datasets Final.csv --no-retrieval
```

### Launcher output

Each launcher run creates a timestamped folder unless `--dry-run` is used:

```
CSV_Result_med/experiments/<timestamp>/
├── run.json                          # summary: status, timing and errors for each experiment
├── timings.csv                       # per-question total, time to first token and stage times
└── <model>/<pipeline>/
    ├── <db><model><suffix>_<dataset>.json   # answers, contexts and Ragas scores
    └── <dataset><mode>.log                  # mode suffix for ablation runs
```

For an experiment batch, the launcher returns `0` when all experiments pass its output checks, `1` on a recorded experiment failure and `130` when it catches an interruption during an experiment. A completed Ragas evaluation can still contain missing scores. Check the per-question `metric_errors` in the result JSON.

## Troubleshooting

- **`Database directory missing`**: the path in `config.py` / `config_med.py` does not exist. Check `PERSIST_DIRECTORY_DB`.
- **No context retrieved / generic answers**: check the selected database and embedding model. The dataset log shows the candidate scores and how many passed the threshold. An empty context is expected with `--no-retrieval`.
- **Ragas evaluation fails**: check that Ollama is running and that `ollama list` includes `medgemma:27b`. For a missing score, check `metric_errors` in the result JSON.
- **CUDA out of memory**: check the experiment log to see which model was loading. For a retrieval-enabled run, try a smaller generator such as `Qwen2.5-0.5B-Instruct`. The launcher already runs one experiment at a time.
- **An experiment fails**: read the `.log` file referenced by the `log_file` field in `run.json`.
