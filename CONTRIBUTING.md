# Contributing

Thank you for your interest in improving this repository.

## Scope

This project is a research prototype for neurological risk stratification with calibrated uncertainty. Contributions should preserve that framing.

- Do not position the repository as a clinical diagnostic system.
- Keep uncertainty visible in model outputs and interfaces.
- Prefer small, reviewable pull requests with clear rationale.

## Local setup

```bash
cd /home/agi/deeplearning.py
source .venv/bin/activate
```

Optional local runtime directories:

```bash
source scripts/common_env.sh
export_project_env
```

## Validation before proposing changes

Run the lightweight test and build loop:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

```bash
cd jsviz
npm run build
```

If you touch the training or inference workflow, also run:

```bash
python scripts/run_neuro_risk_mvp.py --epochs 6 --mc-samples 8 --device cpu
```

## Contribution guidelines

- Keep code modular and typed where practical.
- Avoid heavyweight dependencies unless there is a strong technical reason.
- Prefer reproducible, low-cost validation over aggressive training.
- Document any medically sensitive claims carefully and conservatively.
- If you change public-facing metrics or viewer payloads, update the README summary accordingly.
