# Environments

Two conda environments, exactly as frozen on the machine that produced the paper numbers (RTX 5090, driver >= 570,
CUDA 12.8 wheels; `torch 2.11.0+cu128`). Any Linux x86_64 + NVIDIA GPU (sm_75 or newer) is expected to work. The
`trace_gen` freeze was taken on Python 3.10, the `maniskill` freeze on Python 3.11 (`docs/KNOWN_ISSUES.md`, item 12).

| env | python | used by | freeze |
|---|---|---|---|
| `trace_gen` | 3.10 | planner: `msgen.*` (training, prediction, labels) | `trace_gen_freeze.txt` |
| `maniskill` | 3.11 | simulator, teachers, distillation, evaluation: `msppo.*`, `tools/*` | `maniskill_freeze.txt` |

```bash
conda create -n trace_gen python=3.10 -y && conda run -n trace_gen pip install -r requirements/trace_gen_freeze.txt
conda create -n maniskill python=3.11 -y && conda run -n maniskill pip install -r requirements/maniskill_freeze.txt
export PG=$(conda run -n trace_gen which python) PM=$(conda run -n maniskill which python)
```

Both freeze files start with `--extra-index-url https://download.pytorch.org/whl/cu128`; pip needs that index for the
`+cu128` wheels (`torch`, `torchvision`, and `torchaudio` in `trace_gen`), which are not on PyPI. The freezes contain a
few packages unrelated to this repository (they are the full environments); pip will install them but nothing here
imports them. The `hf` command-line tool used to download the weights is provided by `huggingface-hub`, which both
freezes contain; the planner stages additionally need an authenticated session (`hf auth login`) because the frozen
encoders are fetched from the Hugging Face Hub (see `third_party/README.md`). `tests/test_sam2marker.py` needs `pytest`
in addition (`pip install pytest`; it is not part of the freezes).
