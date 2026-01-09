"""BCNB dataset provider."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from patho_bench_cli.providers.base import DatasetProvider

logging.basicConfig()
logger = logging.getLogger(__name__)

BCNB_DATASETS = ["bcnb"]

class BCNBProvider(DatasetProvider):
    """
    Provider for the BCNB dataset.
    
    Note: This provider does not support automated downloading because the dataset
    must be manually requested via email. It expects the dataset to be placed
    manually in the slides directory.
    """
    
    @property
    def name(self) -> str:
        return "bcnb"
    
    @property
    def description(self) -> str:
        return "BCNB: Breast Cancer Nile Blue dataset (Manual download only)"
    
    @property
    def datasets(self) -> list[str]:
        return BCNB_DATASETS

    def get_storage_directories(self, output_dir: Path, datasets: list[str] | None = None) -> list[Path]:
        """Get the BCNB subdirectory."""
        return [output_dir / "BCNB"]
    
    def _get_all_tsv_files(self, tasks_dir: Path) -> list[Path]:
        """Find all k=all.tsv files in the BCNB tasks directory."""
        bcnb_tasks_path = tasks_dir / "bcnb"
        if not bcnb_tasks_path.exists():
            return []
        return list(bcnb_tasks_path.glob("**/k=all.tsv"))
    
    def _extract_slide_ids_from_tsv(self, tsv_path: Path) -> pd.DataFrame:
        """Extract case_id and slide_id from a TSV file."""
        df = pd.read_csv(tsv_path, sep="\t")
        if "slide_id" in df.columns and "case_id" in df.columns:
            return df[["case_id", "slide_id"]].drop_duplicates()
        return pd.DataFrame(columns=["case_id", "slide_id"])
    
    def list_tasks(self, tasks_dir: Path, datasets: list[str] | None = None) -> list[dict[str, Any]]:
        """List all available BCNB tasks."""
        tasks = []
        
        # BCNB only has one dataset name in Patho-Bench
        if datasets and "bcnb" not in datasets:
            return tasks
            
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            task_name = tsv_path.parent.name
            df = self._extract_slide_ids_from_tsv(tsv_path)
            
            tasks.append({
                "dataset": "bcnb",
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
        """Get slide IDs needed for BCNB Patho-Bench tasks."""
        result: dict[str, set[str]] = {}
        
        # BCNB only has one dataset name in Patho-Bench
        if datasets and "bcnb" not in datasets:
            return result
            
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            
            if "bcnb" not in result:
                result["bcnb"] = set()
            result["bcnb"].update(slide_df["slide_id"].unique())
        
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
        """Raise error as BCNB requires manual download."""
        raise RuntimeError(
            "The BCNB dataset cannot be downloaded automatically. "
            "You must request the dataset via email (see Patho-Bench documentation) "
            "and place the 'BCNB' folder into your slides directory (e.g., slides/BCNB/)."
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
        """Raise error as BCNB requires manual download."""
        self.download_slides(set(), output_dir, **kwargs)
