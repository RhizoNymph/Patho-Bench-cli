# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Patho-Bench-cli is a unified downloader for [Patho-Bench](https://github.com/mahmoodlab/Patho-Bench) pathology datasets. It downloads whole slide images (WSIs) from various medical imaging archives, verifies them with OpenSlide, and generates embeddings using TRIDENT.

## Common Commands

```bash
# Install
uv pip install -e .
uv pip install -e ".[dev]"  # with dev dependencies

# Run tests
pytest

# CLI usage
patho-bench-cli tasks                           # Download task definitions from HuggingFace
patho-bench-cli list                            # List all providers
patho-bench-cli list <provider>                 # List datasets for a provider
patho-bench-cli download <provider> -o ./slides # Dry-run (creates manifest)
patho-bench-cli download <provider> --download  # Actually download
patho-bench-cli verify ./slides                 # Verify WSI files can be opened
patho-bench-cli embed <provider> --slides-dir ./slides --embeddings-dir ./embeddings
```

## Architecture

### Provider System

The core abstraction is `DatasetProvider` (in `patho_bench_cli/providers/base.py`), an abstract base class that each data source implements:

- **Required properties**: `name`, `description`, `datasets`
- **Required methods**: `list_tasks()`, `get_slide_ids_for_tasks()`, `download_slides()`, `download_full()`
- **Optional override**: `get_storage_directories()` for providers with subdirectory organization

Providers are registered in `patho_bench_cli/providers/registry.py` via `_auto_register()`. The registry provides `get_provider(name)` and `list_providers()`.

Current providers:
- `cptac` - TCIA PathDB API (multiple cancer types)
- `panda` - Kaggle API
- `idr` - BioImage Archive HTTP
- `bioimage` - EBI BioImage Archive
- `ovarian_bevacizumab`, `post_nat_brca` - TCIA collections
- `imp`, `bcnb`, `hancock`, `boehmk` - Various sources

### CLI Structure

`patho_bench_cli/cli.py` contains all subcommand handlers:
- `cmd_list()` - List providers/datasets
- `cmd_download()` - Download slides with optional verification and symlink creation
- `cmd_tasks()` - Download Patho-Bench task definitions from HuggingFace
- `cmd_verify()` - Validate WSI files using OpenSlide
- `cmd_embed()` - Generate embeddings using TRIDENT subprocess

### Key Patterns

**Task files**: Patho-Bench tasks are stored as TSV files at `tasks/{dataset}/{task}/k=all.tsv` with columns including `slide_id` and `case_id`.

**Symlink organization**: The `--create-symlinks` flag creates `by_task/{dataset}/{task}/` directories with symlinks to actual slide files, avoiding duplication when slides appear in multiple tasks.

**Slide verification**: `verify_slides_in_parallel()` uses ThreadPoolExecutor to check WSIs can be opened with OpenSlide, validates MPP/magnification metadata, and performs trial reads at multiple random locations to detect corruption.

**Retry logic**: `patho_bench_cli/utils.py` provides `download_file_with_retry()` and `retry_on_timeout()` decorator using tenacity for transient network errors.

### Adding a New Provider

1. Create `patho_bench_cli/providers/<name>.py` implementing `DatasetProvider`
2. Register in `patho_bench_cli/providers/registry.py` by importing and calling `register_provider()`
3. Provider must handle mapping between Patho-Bench slide_ids and the source's file identifiers

### External Dependencies

- **TRIDENT**: Git submodule at `./TRIDENT/` used for embedding generation via `run_batch_of_slides.py`
- **Patho-Bench**: Git submodule at `./Patho-Bench/` for future `bench` command that will run benchmarking with an embeddings directory argument
