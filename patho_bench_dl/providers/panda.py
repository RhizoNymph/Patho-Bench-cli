"""PANDA dataset provider using Kaggle API."""

import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from patho_bench_dl.providers.base import DatasetProvider
from patho_bench_dl.utils import DEFAULT_MAX_RETRIES

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

COMPETITION_NAME = "prostate-cancer-grade-assessment"
SLIDE_EXTENSION = ".tiff"


class PANDAProvider(DatasetProvider):
    """Provider for PANDA dataset from Kaggle."""
    
    @property
    def name(self) -> str:
        return "panda"
    
    @property
    def description(self) -> str:
        return "PANDA dataset from Kaggle prostate-cancer-grade-assessment competition"
    
    @property
    def datasets(self) -> list[str]:
        return ["panda"]
    
    def _get_kaggle_executable(self) -> str:
        """Find the kaggle executable."""
        kaggle_path = shutil.which("kaggle")
        if kaggle_path:
            return kaggle_path
        
        # Check in venv
        python_dir = Path(sys.executable).parent
        kaggle_path = python_dir / "kaggle"
        if kaggle_path.exists():
            return str(kaggle_path)
        
        return "kaggle"
    
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
        """List all available PANDA tasks."""
        tasks = []
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            if dataset_name != "panda":
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
        """Get slide IDs needed for PANDA Patho-Bench tasks."""
        result: dict[str, set[str]] = {}
        
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            
            if dataset_name != "panda":
                continue
            
            task_name = tsv_path.parent.name
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            
            key = f"panda/{task_name}"
            if key not in result:
                result[key] = set()
            result[key].update(slide_df["slide_id"].unique())
        
        return result
    
    def _download_competition_zip(
        self,
        output_dir: Path,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> Path:
        """Download the full competition zip file with retry on failure."""
        kaggle_cmd = self._get_kaggle_executable()
        zip_name = f"{COMPETITION_NAME}.zip"
        zip_path = output_dir / zip_name
        
        if zip_path.exists():
            logger.info(f"Zip file already exists at {zip_path}")
            return zip_path
        
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _do_download():
            logger.info(f"Downloading competition zip to {output_dir}...")
            subprocess.run(
                [kaggle_cmd, "competitions", "download",
                 "-c", COMPETITION_NAME,
                 "-p", str(output_dir)],
                check=True
            )
        
        _do_download()
        return zip_path
    
    def _extract_slides_from_zip(
        self,
        zip_path: Path,
        slide_ids: set[str] | None,
        output_dir: Path
    ) -> None:
        """Extract slides from the competition zip."""
        logger.info(f"Opening zip file: {zip_path}")
        
        extracted_count = 0
        skipped_count = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_files = set(zf.namelist())
            
            if slide_ids is None:
                # Extract all slides
                slides_to_extract = [
                    f for f in all_files
                    if f.endswith(SLIDE_EXTENSION)
                ]
                logger.info(f"Extracting all {len(slides_to_extract)} slides...")
            else:
                # Extract specific slides
                slides_to_extract = []
                for slide_id in slide_ids:
                    filename = f"{slide_id}{SLIDE_EXTENSION}"
                    possible_paths = [f"train_images/{filename}", filename]
                    for p in possible_paths:
                        if p in all_files:
                            slides_to_extract.append(p)
                            break
                logger.info(f"Extracting {len(slides_to_extract)} specified slides...")
            
            for i, source_path in enumerate(slides_to_extract):
                filename = Path(source_path).name
                target_path = output_dir / filename
                
                if target_path.exists():
                    skipped_count += 1
                    continue
                
                source = zf.open(source_path)
                with open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                extracted_count += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(slides_to_extract)} slides")
        
        logger.info(f"Extraction complete. Extracted: {extracted_count}, Skipped: {skipped_count}")
    
    def download_slides(
        self,
        slide_ids: set[str],
        output_dir: Path,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        cleanup_zip: bool = True,
        **kwargs
    ) -> None:
        """Download specific PANDA slides from Kaggle."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download zip
        zip_path = self._download_competition_zip(output_dir)
        
        # Extract specific slides
        self._extract_slides_from_zip(zip_path, slide_ids, output_dir)
        
        # Cleanup
        if cleanup_zip and zip_path.exists():
            logger.info(f"Removing zip file: {zip_path}")
            zip_path.unlink()
        
        # Create symlinks
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir)
    
    def download_full(
        self,
        output_dir: Path,
        datasets: list[str] | None = None,
        cleanup_zip: bool = True,
        **kwargs
    ) -> None:
        """Download complete PANDA dataset from Kaggle."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download zip
        zip_path = self._download_competition_zip(output_dir)
        
        # Extract ALL slides
        self._extract_slides_from_zip(zip_path, None, output_dir)
        
        # Cleanup
        if cleanup_zip and zip_path.exists():
            logger.info(f"Removing zip file: {zip_path}")
            zip_path.unlink()
    
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
