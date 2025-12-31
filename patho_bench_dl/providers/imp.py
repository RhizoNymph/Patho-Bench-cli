"""IMP dataset provider for direct HTTP download."""

import logging
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd

from patho_bench_dl.providers.base import DatasetProvider
from patho_bench_dl.utils import download_file_urllib_with_retry, DEFAULT_MAX_RETRIES

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Create SSL context that doesn't verify certificates (for expired certs)
_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# Base URL for the IMP-CRS2024 dataset
BASE_URL = "https://open-datasets.inesctec.pt/NQ3sxFMZ/IMP-CRS2024-Dataset/"

# Subdirectories containing slides
SLIDE_FOLDERS = ["CRS1/slides", "CRS2/slides", "CRS_Test/slides"]

# Slide file extension (SVS format on the server)
SLIDE_EXTENSION = ".svs"


class IMPProvider(DatasetProvider):
    """Provider for IMP-CRS2024 dataset via direct HTTP download."""
    
    @property
    def name(self) -> str:
        return "imp"
    
    @property
    def description(self) -> str:
        return "IMP-CRS2024 colorectal cancer dataset from INESCTEC"
    
    @property
    def datasets(self) -> list[str]:
        return ["imp"]
    
    def _get_all_tsv_files(self, tasks_dir: Path) -> list[Path]:
        """Find all k=all.tsv files in the tasks directory."""
        return list(tasks_dir.glob("**/k=all.tsv"))
    
    def _extract_slide_ids_from_tsv(self, tsv_path: Path) -> pd.DataFrame:
        """Extract case_id and slide_id from a TSV file."""
        df = pd.read_csv(tsv_path, sep="\t")
        if "slide_id" in df.columns and "case_id" in df.columns:
            return df[["case_id", "slide_id"]].drop_duplicates()
        return pd.DataFrame(columns=["case_id", "slide_id"])
    
    def list_tasks(self, tasks_dir: Path) -> list[dict[str, Any]]:
        """List all available IMP tasks."""
        tasks = []
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            if dataset_name != "imp":
                continue
            
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
        """Get slide IDs needed for IMP Patho-Bench tasks."""
        result: dict[str, set[str]] = {}
        
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            
            if dataset_name != "imp":
                continue
            
            task_name = tsv_path.parent.name
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            
            key = f"imp/{task_name}"
            if key not in result:
                result[key] = set()
            result[key].update(slide_df["slide_id"].unique())
        
        return result
    
    def _download_file(
        self,
        url: str,
        target_path: Path,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> bool:
        """
        Download a single file from URL with retry on timeout.
        
        Bypasses SSL verification for expired certificates.
        """
        logger.info(f"Downloading {url}")
        return download_file_urllib_with_retry(
            url,
            target_path,
            max_retries=max_retries,
            ssl_context=_SSL_CONTEXT,
        )
    
    def _find_slide_in_folders(self, slide_id: str) -> str | None:
        """
        Determine which folder a slide might be in.
        
        Slides are named like CRC_XXXX.ndpi and could be in any of the folders.
        We'll try each folder in sequence.
        """
        filename = f"{slide_id}{SLIDE_EXTENSION}"
        for folder in SLIDE_FOLDERS:
            yield urljoin(BASE_URL, f"{folder}/{filename}")
    
    def download_slides(
        self,
        slide_ids: set[str],
        output_dir: Path,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Download specific IMP slides."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        skipped = 0
        failed = 0
        
        for slide_id in sorted(slide_ids):
            filename = f"{slide_id}{SLIDE_EXTENSION}"
            target_path = output_dir / filename
            
            if target_path.exists():
                skipped += 1
                continue
            
            # Try each folder until we find the slide
            success = False
            for url in self._find_slide_in_folders(slide_id):
                if self._download_file(url, target_path):
                    success = True
                    downloaded += 1
                    break
            
            if not success:
                failed += 1
                logger.warning(f"Could not find slide: {slide_id}")
        
        logger.info(f"Download complete. Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
        
        # Create symlinks
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir)
    
    def download_full(
        self,
        output_dir: Path,
        datasets: list[str] | None = None,
        **kwargs
    ) -> None:
        """Download complete IMP dataset."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading full IMP-CRS2024 dataset...")
        logger.info(f"Note: This will download all slides from {BASE_URL}")
        logger.info("Folders: CRS1/slides, CRS2/slides, CRS_Test/slides")
        
        # For full download, we need to list the directory contents
        # Since the server may not support directory listing, we'll use
        # the task file to get all known slides and download those
        logger.warning(
            "Full download mode for IMP requires knowing all slide IDs. "
            "Consider running with task files to get the slide list, "
            "or manually download from the website."
        )
        
        # Try to use tasks dir if available in the same parent
        tasks_dir = output_dir.parent / "tasks"
        if tasks_dir.exists():
            slide_ids_by_task = self.get_slide_ids_for_tasks(tasks_dir)
            all_slides = set()
            for ids in slide_ids_by_task.values():
                all_slides.update(ids)
            
            if all_slides:
                logger.info(f"Found {len(all_slides)} slides from task files")
                self.download_slides(all_slides, output_dir)
                return
        
        logger.error(
            "Cannot determine full slide list. Please download manually from:\n"
            f"  {BASE_URL}"
        )
    
    def _create_symlinks(self, tasks_dir: Path, slides_dir: Path) -> None:
        """Create per-task symlink directories."""
        task_slide_ids = self.get_slide_ids_for_tasks(tasks_dir)
        
        for task_key, slide_ids in task_slide_ids.items():
            task_dir = slides_dir / "by_task" / task_key
            task_dir.mkdir(parents=True, exist_ok=True)
            
            symlink_count = 0
            for slide_id in slide_ids:
                source_file = slides_dir / f"{slide_id}{SLIDE_EXTENSION}"
                if source_file.exists():
                    symlink_path = task_dir / source_file.name
                    if not symlink_path.exists():
                        symlink_path.symlink_to(source_file.resolve())
                        symlink_count += 1
            
            if symlink_count > 0:
                logger.info(f"  {task_key}: {symlink_count} symlinks")
