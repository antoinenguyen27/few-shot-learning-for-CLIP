# Dependencies

The repo uses a single shared Python environment for the common pipeline.

## Recommended Setup

Use Python 3.10 or 3.11 if possible:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

GPU-specific PyTorch installs can vary by machine. If the default PyPI install is not suitable, install the right PyTorch build for your CUDA or CPU setup first, then run:

```bash
pip install -e .
```

## Core Libraries

- `torch` and `torchvision`: model execution and image transforms.
- `open_clip_torch`: shared CLIP implementation.
- `kagglehub`: Kaggle dataset downloads.
- `Pillow`: image loading.
- `scipy`: Stanford Cars `.mat` annotation parsing.
- `numpy`, `pandas`, `scikit-learn`: numerical utilities and method compatibility.
- `ftfy`, `regex`, `tqdm`, `timm`, `huggingface_hub`, `safetensors`: OpenCLIP and prompt-method compatibility.

## Paper Repo Dependencies

Several method repos were originally built around Dassl and OpenAI CLIP internals. Do not add those as shared dependencies by default. Add method-specific dependencies only when the adapted method truly needs them.
