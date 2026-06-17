# pyrecall

[![PyPI version](https://img.shields.io/pypi/v/pyrecall.svg)](https://pypi.org/project/pyrecall/)
[![CI](https://github.com/Arths17/Pyrecall/actions/workflows/ci.yml/badge.svg)](https://github.com/Arths17/Pyrecall/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pyrecall?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pyrecall)

**Keep your models balanced.**  
Continuous fine-tuning with automatic forgetting detection and skill rollback.

Fine-tune your model on new data and it quietly loses skills it already had — coding ability, reasoning, safety guardrails. pyrecall catches this before it ships.

---

## Install

```bash
pip install pyrecall
```

Supports Python 3.11–3.14. Works on CPU-only hardware.

---

## Quickstart

```python
from pyrecall import Model

model = Model("meta-llama/Llama-3.2-1B")

model.snapshot("before_fine_tune")
model.learn("data.jsonl", epochs=3)

report = model.check()
if not report.is_healthy:
    model.rollback(to="before_fine_tune")
```

Or via CLI:

```bash
pyrecall init --model meta-llama/Llama-3.2-1B
pyrecall snapshot before_v1
pyrecall learn train.jsonl --epochs 5 --snapshot-after after_v1
pyrecall check --before before_v1 --after after_v1
# exit 0 → ship it   exit 2 → pyrecall rollback before_v1
```

---

## How it works

Snapshots run 180 benchmark prompts across 9 skill categories (reasoning, coding, safety, math, multilingual, and more) scored by log-likelihood. After training, `check` diffs the scores and flags any category that drops past your threshold (default 10%). Only the LoRA adapter is stored per snapshot — a few hundred MB, not the full model.

---

## Supported models

Any causal LM on HuggingFace Hub — Llama, Mistral, Phi, Gemma, Qwen, Falcon, GPT-2/Neo/J, and more. LoRA targets are auto-detected.

---

## Docs

Full CLI reference, Python API, experiment tracker integrations, custom benchmarks, and more:  
**[pyrecall.github.io/Pyrecall](https://pyrecall.github.io/Pyrecall/)**

---

## Contributing

```bash
git clone https://github.com/Arths17/Pyrecall
pip install -e ".[dev]"
pytest
```

Open an issue before large changes.

---

MIT — see [LICENSE](LICENSE).
