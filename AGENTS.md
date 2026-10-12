# Repository Guidelines

## Project Structure & Module Organization

- `sub_trans.py` contains the CLI, subtitle parsing/cleaning, translation workflow, caching, and SRT/ASS output logic.
- `prompts.py` contains the reusable prompt templates sent to the language model.
- `pyproject.toml` defines the package metadata, Python requirement (3.12+), and runtime dependencies.
- `README.md` documents user-facing behavior and packaging examples. Runtime files such as `config.yaml`, input subtitles, cache data, and generated output belong outside the source modules and should not be committed.

## Build, Test, and Development Commands

Create an environment and install the project with your preferred Python workflow (for example, `uv sync`). Run the CLI directly:

```powershell
python sub_trans.py -i input.srt -o output.srt
python sub_trans.py -i input.srt --format ass --reflect --bilingual
```

Use `-c path\to\config.yaml` and, when needed, `-a NAME` to select an API entry. Build artifacts are documented in `README.md`; the supported examples use PyInstaller or Nuitka.

Before submitting changes, compile-check the modules:

```powershell
python -m compileall sub_trans.py prompts.py
```

No automated test suite is currently checked in. For behavior changes, manually exercise representative SRT input, malformed/empty lines, configured caching, and the requested output format.

## Coding Style & Naming Conventions

Follow standard Python style: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for module constants (for example, `BLANK_RE`). Keep type annotations consistent with the existing code and use UTF-8 text handling. Preserve prompt placeholders such as `${source_language}` and `${target_language}` exactly when editing templates. Keep CLI flags and help text synchronized with `argparse` definitions.

## Testing Guidelines

There is no configured pytest or coverage target. Add focused tests only when introducing substantial parsing or translation logic, using `test_*.py` names and standard-library `unittest` or pytest if a dependency is intentionally added. Always include a reproducible manual CLI command in the change description.

## Commit & Pull Request Guidelines

Keep commits small and imperative (for example, `Fix ASS timestamp conversion`). Pull requests should explain the user-visible behavior, list configuration or prompt changes, include validation commands and sample input/output details, and call out any API, token, concurrency, or output-format impact. Never include API keys, private subtitle files, caches, or generated binaries.

## Security & Configuration

Store API credentials only in local, ignored configuration such as `config.yaml` or environment variables. Review concurrency and cache settings before sharing logs or artifacts, since requests and translated subtitle content may contain sensitive material.
