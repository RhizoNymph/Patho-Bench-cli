"""Boehmk_ dataset provider."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from patho_bench_cli.providers.base import DatasetProvider

logging.basicConfig()
logger = logging.getLogger(__name__)

BOEHMK_DATASETS = ["boehmk_"]

class BoehmkProvider(DatasetProvider):
    """
    Provider for the boehmk_ dataset.
    
    Note: This provider does not support automated downloading. 
    It expects the dataset to be placed manually in the slides directory.
    """
    
    @property
    def name(self) -> str:
        return "boehmk_"
    
    @property
    def description(self) -> str:
        return "Boehmk: Precision medicine for metastatic breast cancer dataset (Manual download only)"
    
    @property
    def datasets(self) -> list[str]:
        return BOEHMK_DATASETS

    def get_storage_directories(self, output_dir: Path, datasets: list[str] | None = None) -> list[Path]:
        """Get the boehmk_ subdirectory."""
        return [output_dir / "boehmk_"]
    
    def _get_all_tsv_files(self, tasks_dir: Path) -> list[Path]:
        """Find all k=all.tsv files in the boehmk_ tasks directory."""
        boehmk_tasks_path = tasks_dir / "boehmk_"
        if not boehmk_tasks_path.exists():
            return []
        return list(boehmk_tasks_path.glob("**/k=all.tsv"))
    
    def _extract_slide_ids_from_tsv(self, tsv_path: Path) -> pd.DataFrame:
        """Extract case_id and slide_id from a TSV file."""
        df = pd.read_csv(tsv_path, sep="\t")
        if "slide_id" in df.columns and "case_id" in df.columns:
            return df[["case_id", "slide_id"]].drop_duplicates()
        return pd.DataFrame(columns=["case_id", "slide_id"])
    
    def list_tasks(self, tasks_dir: Path, datasets: list[str] | None = None) -> list[dict[str, Any]]:
        """List all available boehmk_ tasks."""
        tasks = []
        
        if datasets and "boehmk_" not in datasets:
            return tasks
            
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            task_name = tsv_path.parent.name
            df = self._extract_slide_ids_from_tsv(tsv_path)
            
            tasks.append({
                "dataset": "boehmk_",
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
        """Get slide IDs needed for boehmk_ Patho-Bench tasks."""
        result: dict[str, set[str]] = {}
        
        if datasets and "boehmk_" not in datasets:
            return result
            
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            
            if "boehmk_" not in result:
                result["boehmk_"] = set()
            result["boehmk_"].update(slide_df["slide_id"].unique())
        
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
        """ Organize manual boehmk_ downloads via symlinks if requested."""
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir)
            return

        raise RuntimeError(
            "The boehmk_ dataset cannot be downloaded automatically. "
            "You must download the dataset manually and place the 'boehmk_' folder "
            "into your slides directory (e.g., slides/boehmk_/)."
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
        """Organize manual boehmk_ downloads via symlinks if requested."""
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir)
            return
            
        self.download_slides(set(), output_dir, **kwargs)

    def _create_symlinks(
        self,
        tasks_dir: Path,
        slides_dir: Path,
    ) -> None:
        """Create per-task symlink directories for boehmk_ slides."""
        boehmk_dir = slides_dir / "boehmk_"
        if not boehmk_dir.exists():
            logger.warning(f"Boehmk_ slides directory not found at {boehmk_dir}. Skipping symlink creation.")
            return

        for tsv_path in self._get_all_tsv_files(tasks_dir):
            task_name = tsv_path.parent.name
            
            # Get slide IDs needed for this specific task
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            task_slide_ids = set(slide_df["slide_id"].unique())
            
            task_dir = slides_dir / "by_task" / "boehmk_" / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            symlink_count = 0
            # Search for files with common WSI extensions and standard image formats
            extensions = {'.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn', '.bif', '.vms', '.vmu', '.jpg', '.jpeg', '.png'}
            
            # Convert slide IDs to strings for robust comparison with filenames
            task_slide_ids_str = {str(sid) for sid in task_slide_ids}
            
            for img_file in boehmk_dir.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in extensions:
                    # Boehmk slide_id is usually a substring or the whole stem
                    if img_file.stem in task_slide_ids_str:
                        symlink_path = task_dir / img_file.name
                        if not symlink_path.exists():
                            symlink_path.symlink_to(img_file.resolve())
                            symlink_count += 1
            
            if symlink_count > 0:
                logger.info(f"  boehmk_/{task_name}: {symlink_count} symlinks")
