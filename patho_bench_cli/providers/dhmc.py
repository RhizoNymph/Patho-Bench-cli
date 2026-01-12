"""DHMC dataset provider for manual download datasets."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from patho_bench_cli.providers.base import DatasetProvider

logging.basicConfig()
logger = logging.getLogger(__name__)

# Mapping of Patho-Bench dataset names to folder names in the slides directory
DHMC_DATASETS = {
    "dhmc_luad": "LungCancer",
    "cptac_ccrcc_dhmc": "KidneyCancer",
}


class DHMCProvider(DatasetProvider):
    """
    Provider for DHMC datasets (Lung and Kidney cancer).

    Note: This provider does not support automated downloading. The datasets
    must be manually obtained and placed in the slides directory with the
    appropriate folder names (LungCancer, KidneyCancer).
    """

    @property
    def name(self) -> str:
        return "dhmc"

    @property
    def description(self) -> str:
        return "DHMC: Dartmouth-Hitchcock Medical Center datasets (Manual download only)"

    @property
    def datasets(self) -> list[str]:
        return list(DHMC_DATASETS.keys())

    def get_storage_directories(self, output_dir: Path, datasets: list[str] | None = None) -> list[Path]:
        """Get the DHMC subdirectories."""
        if datasets:
            return [output_dir / DHMC_DATASETS[d] for d in datasets if d in DHMC_DATASETS]
        return [output_dir / folder for folder in DHMC_DATASETS.values()]

    def _get_all_tsv_files(self, tasks_dir: Path, dataset_name: str) -> list[Path]:
        """Find all k=all.tsv files for a specific DHMC dataset."""
        dataset_path = tasks_dir / dataset_name
        if not dataset_path.exists():
            return []
        return list(dataset_path.glob("**/k=all.tsv"))

    def _extract_slide_ids_from_tsv(self, tsv_path: Path) -> pd.DataFrame:
        """Extract case_id and slide_id from a TSV file."""
        df = pd.read_csv(tsv_path, sep="\t")
        if "slide_id" in df.columns and "case_id" in df.columns:
            return df[["case_id", "slide_id"]].drop_duplicates()
        return pd.DataFrame(columns=["case_id", "slide_id"])

    def list_tasks(self, tasks_dir: Path, datasets: list[str] | None = None) -> list[dict[str, Any]]:
        """List all available DHMC tasks."""
        tasks = []

        for dataset_name in DHMC_DATASETS.keys():
            # Filter to requested datasets
            if datasets and dataset_name not in datasets:
                continue

            for tsv_path in self._get_all_tsv_files(tasks_dir, dataset_name):
                task_name = tsv_path.parent.name
                df = self._extract_slide_ids_from_tsv(tsv_path)

                tasks.append({
                    "dataset": dataset_name,
                    "task": task_name,
                    "n_slides": len(df),
                    "n_cases": df["case_id"].nunique() if "case_id" in df.columns else 0,
                })
        return tasks

    def get_slide_ids_for_tasks(
        self,
        tasks_dir: Path,
        datasets: list[str] | None = None
    ) -> dict[str, set[str]]:
        """Get slide IDs needed for DHMC Patho-Bench tasks."""
        result: dict[str, set[str]] = {}

        for dataset_name in DHMC_DATASETS.keys():
            # Filter to requested datasets
            if datasets and dataset_name not in datasets:
                continue

            for tsv_path in self._get_all_tsv_files(tasks_dir, dataset_name):
                task_name = tsv_path.parent.name
                slide_df = self._extract_slide_ids_from_tsv(tsv_path)

                key = f"{dataset_name}/{task_name}"
                if key not in result:
                    result[key] = set()
                result[key].update(slide_df["slide_id"].unique())

        return result

    def download_slides(
        self,
        slide_ids: set[str],
        output_dir: Path,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Organize manual DHMC downloads via symlinks if requested."""
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir)
            return

        raise RuntimeError(
            "The DHMC datasets cannot be downloaded automatically. "
            "Please obtain the datasets manually and place them in your slides directory:\n"
            "  - LungCancer/ for dhmc_luad\n"
            "  - KidneyCancer/ for cptac_ccrcc_dhmc"
        )

    def download_full(
        self,
        output_dir: Path,
        datasets: list[str] | None = None,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Organize manual DHMC downloads via symlinks if requested."""
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir, datasets)
            return

        self.download_slides(set(), output_dir, **kwargs)

    def _create_symlinks(
        self,
        tasks_dir: Path,
        slides_dir: Path,
        datasets: list[str] | None = None,
    ) -> None:
        """Create per-task symlink directories for DHMC slides."""
        extensions = {'.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn', '.bif', '.vms', '.vmu', '.jpg', '.jpeg', '.png'}

        for dataset_name, folder_name in DHMC_DATASETS.items():
            # Filter to requested datasets
            if datasets and dataset_name not in datasets:
                continue

            dataset_dir = slides_dir / folder_name
            if not dataset_dir.exists():
                logger.warning(f"DHMC {dataset_name} slides directory not found at {dataset_dir}. Skipping symlink creation.")
                continue

            for tsv_path in self._get_all_tsv_files(tasks_dir, dataset_name):
                task_name = tsv_path.parent.name

                # Get slide IDs needed for this specific task
                slide_df = self._extract_slide_ids_from_tsv(tsv_path)
                task_slide_ids = set(str(sid) for sid in slide_df["slide_id"].unique())

                task_dir = slides_dir / "by_task" / dataset_name / task_name
                task_dir.mkdir(parents=True, exist_ok=True)

                symlink_count = 0

                for img_file in dataset_dir.glob("*"):
                    if img_file.is_file() and img_file.suffix.lower() in extensions:
                        # slide_id is usually the filename stem
                        if img_file.stem in task_slide_ids:
                            symlink_path = task_dir / img_file.name
                            if not symlink_path.exists():
                                symlink_path.symlink_to(img_file.resolve())
                                symlink_count += 1

                if symlink_count > 0:
                    logger.info(f"  {dataset_name}/{task_name}: {symlink_count} symlinks")
