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
patho-bench-cli tasks
```

### List available datasets

```bash
# List all providers
patho-bench-cli list

# List datasets for a specific provider
patho-bench-cli list cptac
patho-bench-cli list panda
patho-bench-cli list ovarian_bevacizumab
patho-bench-cli list post_nat_brca
patho-bench-cli list idr
```

### Download slides

```bash
# Download CPTAC slides needed for Patho-Bench (dry-run, creates manifest)
patho-bench-cli download cptac

# Actually download
patho-bench-cli download cptac --download

# Download specific datasets
patho-bench-cli download cptac --datasets cptac_ccrcc cptac_brca --download

# Download with per-task symlinks
patho-bench-cli download cptac --download --create-symlinks

# Download PANDA slides
patho-bench-cli download panda --download

# Download Ovarian Bevacizumab Response slides
patho-bench-cli download ovarian_bevacizumab --download

# Download POST-NAT-BRCA slides
patho-bench-cli download post_nat_brca --download

# Download IDR slides (via BioImage Archive)
patho-bench-cli download idr --download
```

### Full dataset download

Download entire datasets (not just Patho-Bench slides):

```bash
patho-bench-cli download cptac --full --datasets cptac_ccrcc
patho-bench-cli download panda --full
patho-bench-cli download ovarian_bevacizumab --full
patho-bench-cli download post_nat_brca --full
```

### Verify and clean slides

Recursively check if WSI files can be opened with OpenSlide:

```bash
# Verify slides in a directory
patho-bench-cli verify /path/to/slides

# Verify and delete invalid slides
patho-bench-cli verify /path/to/slides --delete

# Use specific number of parallel jobs
patho-bench-cli verify /path/to/slides --jobs 8
```

## Data Sources

| Provider | Source | Authentication |
|----------|--------|----------------|
| `cptac` | [TCIA (The Cancer Imaging Archive)](https://www.cancerimagingarchive.net/) | None |
| `panda` | [Kaggle Competition](https://www.kaggle.com/c/prostate-cancer-grade-assessment) | `~/.kaggle/kaggle.json` |
| `imp` | [INESCTEC Open Datasets](https://open-datasets.inesctec.pt/NQ3sxFMZ/IMP-CRS2024-Dataset/) | None |
| `ovarian_bevacizumab` | [TCIA Ovarian Bevacizumab Response](https://www.cancerimagingarchive.net/collection/ovarian-bevacizumab-response/) | None |
| `post_nat_brca` | [TCIA POST-NAT-BRCA](https://www.cancerimagingarchive.net/collection/post-nat-brca/) | None |
| `idr` | [Image Data Resource (OpenMicroscopy)](https://idr.openmicroscopy.org/) via [BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) | None |

### IDR Datasets

IDR slides are downloaded from the EBI BioImage Archive via direct HTTP. No additional dependencies required.

Available IDR datasets:
- `ucla_lung` - Lung carcinoma in-situ lesions from [idr0082](https://idr.openmicroscopy.org/webclient/?show=project-1251) ([S-BIAD509](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD509))

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

## Adding New Providers

Create a new provider by implementing `DatasetProvider` in `patho_bench_cli/providers/`:

```python
from patho_bench_cli.providers.base import DatasetProvider

class MyProvider(DatasetProvider):
    @property
    def name(self) -> str:
        return "my_provider"
    
    # ... implement other methods
```

Then register it in `patho_bench_cli/providers/registry.py`.