# Patho-Bench-dl

Unified downloader for [Patho-Bench](https://github.com/mahmoodlab/Patho-Bench) datasets.

## Installation

```bash
# Using uv
uv pip install -e .

# Or using pip
pip install -e .
```

## Usage

### Download task definitions

First, download the Patho-Bench task definitions from HuggingFace:

```bash
patho-bench-dl tasks
```

### List available datasets

```bash
# List all providers
patho-bench-dl list

# List datasets for a specific provider
patho-bench-dl list cptac
patho-bench-dl list panda
```

### Download slides

```bash
# Download CPTAC slides needed for Patho-Bench (dry-run, creates manifest)
patho-bench-dl download cptac

# Actually download
patho-bench-dl download cptac --download

# Download specific datasets
patho-bench-dl download cptac --datasets cptac_ccrcc cptac_brca --download

# Download with per-task symlinks
patho-bench-dl download cptac --download --create-symlinks

# Download PANDA slides
patho-bench-dl download panda --download
```

### Full dataset download

Download entire datasets (not just Patho-Bench slides):

```bash
patho-bench-dl download cptac --full --datasets cptac_ccrcc
patho-bench-dl download panda --full
```

## Data Sources

| Provider | Source | Authentication |
|----------|--------|----------------|
| `cptac` | [TCIA (The Cancer Imaging Archive)](https://www.cancerimagingarchive.net/) | None |
| `panda` | [Kaggle Competition](https://www.kaggle.com/c/prostate-cancer-grade-assessment) | `~/.kaggle/kaggle.json` |

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

## Adding New Providers

Create a new provider by implementing `DatasetProvider` in `patho_bench_dl/providers/`:

```python
from patho_bench_dl.providers.base import DatasetProvider

class MyProvider(DatasetProvider):
    @property
    def name(self) -> str:
        return "my_provider"
    
    # ... implement other methods
```

Then register it in `patho_bench_dl/providers/registry.py`.