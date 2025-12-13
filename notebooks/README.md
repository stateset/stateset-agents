# Notebooks

These Jupyter notebooks are practical, runnable walkthroughs for using StateSet Agents locally.

## Prereqs

- Create and activate a venv, then install the repo in editable mode:
  - `pip install -e ".[training,trl]"`
- Install Jupyter tooling:
  - `pip install jupyterlab ipykernel`
- For GPU training, install a CUDA-enabled PyTorch build that matches your NVIDIA driver.

## Notebooks

- `notebooks/00_environment_setup.ipynb`: sanity checks + stub-mode quick start.
- `notebooks/01_qwen_support_agent_gspo.ipynb`: GSPO + LoRA fine-tuning for a Qwen technical support agent.
- `notebooks/02_qwen_sales_agent_gspo.ipynb`: GSPO + LoRA fine-tuning for a Qwen sales agent.

## Run

From the repo root:

- `jupyter lab`

