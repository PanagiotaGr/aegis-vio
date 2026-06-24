# Installation

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Development checks

```bash
pytest
```

## Dataset setup

Place EuRoC MAV sequences locally under:

```text
datasets/euroc/MH_01_easy/mav0/
```

Datasets are not committed to GitHub.
