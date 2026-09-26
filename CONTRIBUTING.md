# Contributing

Thank you for your interest in this project. Contributions of every size are welcome: bug reports, documentation
fixes, new metrics or segmenters for the `hintauc` library, and reproduction reports.

## Reporting issues

- Use the GitHub issue tracker. Say what you ran (command line or code), what you expected, what happened, and
  your environment (OS, Python, PyTorch/CUDA versions, GPU or CPU).
- For a reproduction that does not match the paper or the authors' reference numbers, attach the metric JSON or the
  console output of the script you used (`replicability/run.sh`, `reproduce/examples/run_examples.sh`, ...).

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your change. Keep the library (`hintauc/`) free of hard-coded paths and of dependencies that are not
   available under a free licence.
3. Add or update tests in `tests/` for behaviour changes in `hintauc/`, and run them:
   ```bash
   pip install -e ".[perceptual]" pytest
   pytest tests
   ```
4. If the change affects the deterministic hint generator or the metrics, state in the pull request whether the
   released numbers change (`python reproduce/scripts/A1_tables_from_released_metrics.py` and the example suite are
   the reference checks).
5. Update the documentation that describes what you changed (`README.md`, `docs/`, or the README of the component).
6. Open the pull request with a short description of the motivation and of how you tested it.

## Style

- Python 3.9 compatible, PEP 8, `snake_case`; docstrings for public functions of `hintauc`.
- No files that reveal private data, credentials or machine-specific paths.

## Licence

By contributing you agree that your contributions are licensed under the MIT License of this repository (see
`LICENSE`). Third-party code must keep its own licence notice and be listed in `THIRD_PARTY_NOTICES.md`.
