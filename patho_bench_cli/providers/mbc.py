"""MBC dataset provider."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from patho_bench_cli.providers.base import DatasetProvider

logging.basicConfig()
logger = logging.getLogger(__name__)

MBC_DATASETS = ["mbc_"]


class MBCProvider(DatasetProvider):
    """
    Provider for the MBC dataset.

    Note: This provider does not support automated downloading because the dataset
    must be manually downloaded. It expects the dataset to be placed
    manually in the slides directory.
    """

    @property
    def name(self) -> str:
        return "mbc"

    @property
    def description(self) -> str:
        return "MBC: Metastatic Breast Cancer dataset (Manual download only)"

    @property
    def datasets(self) -> list[str]:
        return MBC_DATASETS

    def get_storage_directories(self, output_dir: Path, datasets: list[str] | None = None) -> list[Path]:
        """Get the MBC subdirectory."""
        return [output_dir / "MBC"]

    def _get_all_tsv_files(self, tasks_dir: Path) -> list[Path]:
        """Find all k=all.tsv files in the MBC tasks directory."""
        mbc_tasks_path = tasks_dir / "mbc_"
        if not mbc_tasks_path.exists():
            return []
        return list(mbc_tasks_path.glob("**/k=all.tsv"))

    def _extract_slide_ids_from_tsv(self, tsv_path: Path) -> pd.DataFrame:
        """Extract case_id and slide_id from a TSV file."""
        df = pd.read_csv(tsv_path, sep="\t")
        if "slide_id" in df.columns and "case_id" in df.columns:
            return df[["case_id", "slide_id"]].drop_duplicates()
        return pd.DataFrame(columns=["case_id", "slide_id"])

    def list_tasks(self, tasks_dir: Path, datasets: list[str] | None = None) -> list[dict[str, Any]]:
        """List all available MBC tasks."""
        tasks = []

        if datasets and "mbc_" not in datasets:
            return tasks

        for tsv_path in self._get_all_tsv_files(tasks_dir):
            task_name = tsv_path.parent.name
            df = self._extract_slide_ids_from_tsv(tsv_path)

            tasks.append({
                "dataset": "mbc_",
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
        """Get slide IDs needed for MBC Patho-Bench tasks."""
        result: dict[str, set[str]] = {}

        if datasets and "mbc_" not in datasets:
            return result

        for tsv_path in self._get_all_tsv_files(tasks_dir):
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)

            if "mbc_" not in result:
                result["mbc_"] = set()
            result["mbc_"].update(slide_df["slide_id"].unique())

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
        """Organize manual MBC downloads via symlinks if requested."""
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir)
            return

        raise RuntimeError(
            "The MBC dataset cannot be downloaded automatically. "
            "You must download the dataset manually and place the 'MBC' folder "
            "into your slides directory (e.g., slides/MBC/)."
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
        """Organize manual MBC downloads via symlinks if requested."""
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir)
            return

        self.download_slides(set(), output_dir, **kwargs)

    def _create_symlinks(
        self,
        tasks_dir: Path,
        slides_dir: Path,
    ) -> None:
        """Create per-task symlink directories for MBC slides."""
        mbc_dir = slides_dir / "MBC"
        if not mbc_dir.exists():
            logger.warning(f"MBC slides directory not found at {mbc_dir}. Skipping symlink creation.")
            return

        for tsv_path in self._get_all_tsv_files(tasks_dir):
            task_name = tsv_path.parent.name

            # Get slide IDs needed for this specific task
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            task_slide_ids = set(slide_df["slide_id"].unique())

            task_dir = slides_dir / "by_task" / "mbc_" / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            symlink_count = 0
            # MBC slides are SVS files
            extensions = {'.svs'}

            # Convert slide IDs to strings for robust comparison with filenames
            task_slide_ids_str = {str(sid) for sid in task_slide_ids}

            for img_file in mbc_dir.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in extensions:
                    # MBC slide_id is the filename stem
                    if img_file.stem in task_slide_ids_str:
                        symlink_path = task_dir / img_file.name
                        if not symlink_path.exists():
                            symlink_path.symlink_to(img_file.resolve())
                            symlink_count += 1

            if symlink_count > 0:
                logger.info(f"  mbc_/{task_name}: {symlink_count} symlinks")
