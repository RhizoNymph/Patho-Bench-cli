"""Bioimage dataset provider for Biostudies FTP download."""

import asyncio
import logging
import ssl
import time
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import pandas as pd
from aiolimiter import AsyncLimiter

from patho_bench_cli.providers.base import DatasetProvider

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Base URLs for Biostudies
SR386_BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs/"
BRAF_BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/407/S-BIAD1407/Files/BRAF/"
VALENTINO_BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/407/S-BIAD1407/Files/VALENTINO/"

# Download settings
DEFAULT_CONCURRENT_DOWNLOADS = 10
DEFAULT_RATE_LIMIT = 5  # requests per second
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1MB chunks

class BioimageProvider(DatasetProvider):
    """Provider for Bioimage datasets (SR386, CRC Outcomes) from Biostudies FTP."""
    
    def __init__(
        self,
        concurrent_downloads: int = DEFAULT_CONCURRENT_DOWNLOADS,
        rate_limit: int = DEFAULT_RATE_LIMIT,
    ):
        self.concurrent_downloads = concurrent_downloads
        self.rate_limit = rate_limit
    
    @property
    def name(self) -> str:
        return "bioimage"
    
    @property
    def description(self) -> str:
        return "Bioimage datasets (SR386, CRC Outcomes) from Biostudies"
    
    @property
    def datasets(self) -> list[str]:
        return ["sr386_", "crc_outcomes"]
    
    def _get_all_tsv_files(self, tasks_dir: Path) -> list[Path]:
        """Find all k=all.tsv files in the tasks directory."""
        return list(tasks_dir.glob("**/k=all.tsv"))
    
    def _extract_slide_ids_from_tsv(self, tsv_path: Path) -> pd.DataFrame:
        """Extract case_id and slide_id from a TSV file."""
        df = pd.read_csv(tsv_path, sep="\t")
        if "slide_id" in df.columns:
            case_col = "case_id" if "case_id" in df.columns else "slide_id"
            return df[[case_col, "slide_id"]].drop_duplicates()
        return pd.DataFrame(columns=["case_id", "slide_id"])
    
    def list_tasks(self, tasks_dir: Path, datasets: list[str] | None = None) -> list[dict[str, Any]]:
        """List all available Bioimage tasks."""
        tasks = []
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            if dataset_name not in self.datasets:
                continue
            
            if datasets and dataset_name not in datasets:
                continue
                
            task_name = tsv_path.parent.name
            df = self._extract_slide_ids_from_tsv(tsv_path)
            
            tasks.append({
                "dataset": dataset_name,
                "task": task_name,
                "n_slides": len(df),
                "n_cases": df.iloc[:, 0].nunique() if not df.empty else 0,
            })
        return tasks
    
    def get_slide_ids_for_tasks(
        self,
        tasks_dir: Path,
        datasets: list[str] | None = None
    ) -> dict[str, set[str]]:
        """Get slide IDs needed for Bioimage Patho-Bench tasks."""
        result: dict[str, set[str]] = {}
        
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            if dataset_name not in self.datasets:
                continue
            
            if datasets and dataset_name not in datasets:
                continue
            
            task_name = tsv_path.parent.name
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            
            key = f"{dataset_name}/{task_name}"
            if key not in result:
                result[key] = set()
            result[key].update(slide_df["slide_id"].unique())
        
        return result
    
    def _get_slide_url(self, dataset: str, task: str, slide_id: str) -> str:
        """Get the Biostudies URL for a slide."""
        if dataset == "sr386_":
            return f"{SR386_BASE_URL}{slide_id}.czi"
        elif dataset == "crc_outcomes":
            if task.startswith("braf_"):
                return f"{BRAF_BASE_URL}{slide_id}.czi"
            elif "valentino" in task.lower():
                return f"{VALENTINO_BASE_URL}{slide_id}.tif"
            else:
                # Default to .czi from BRAF if unknown
                return f"{BRAF_BASE_URL}{slide_id}.czi"
        return ""

    async def _download_file_async(
        self,
        session: aiohttp.ClientSession,
        limiter: AsyncLimiter,
        url: str,
        output_path: Path,
        slide_id: str,
    ) -> tuple[str, bool, str | None]:
        """Download a single file asynchronously."""
        temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            async with limiter:
                logger.info(f"  Starting download: {slide_id} ({url})")
                async with session.get(url) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}"
                        logger.error(f"  Failed {slide_id}: {error_msg} (URL: {url})")
                        return (slide_id, False, error_msg)
                    
                    total_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    last_log_time = time.time()
                    log_interval = 60  # Log every minute per file
                    
                    async with aiofiles.open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(DEFAULT_CHUNK_SIZE):
                            await f.write(chunk)
                            downloaded += len(chunk)
                            
                            current_time = time.time()
                            if current_time - last_log_time > log_interval:
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    logger.info(f"    {slide_id}: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({percent:.1f}%)")
                                else:
                                    logger.info(f"    {slide_id}: {downloaded/1024/1024:.1f}MB downloaded")
                                last_log_time = current_time
                    
                    temp_path.rename(output_path)
                    logger.info(f"  Downloaded {slide_id}")
                    return (slide_id, True, None)
                    
        except asyncio.CancelledError:
            if temp_path.exists():
                temp_path.unlink()
            raise
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return (slide_id, False, str(e))

    async def _download_slides_async(
        self,
        downloads: list[tuple[str, str, Path]],  # (slide_id, url, output_path)
    ) -> tuple[int, int, list[str]]:
        """Download multiple slides concurrently."""
        limiter = AsyncLimiter(self.rate_limit, 1.0)
        semaphore = asyncio.Semaphore(self.concurrent_downloads)
        
        async def bounded_download(session, slide_id, url, output_path):
            async with semaphore:
                return await self._download_file_async(
                    session, limiter, url, output_path, slide_id
                )
        
        # Biostudies often has SSL issues, using unverified context as backup if needed
        # but let's try standard first or common library pattern
        connector = aiohttp.TCPConnector(limit=self.concurrent_downloads)
        timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=600)  # No total timeout, but watch for hung sockets
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                bounded_download(session, slide_id, url, output_path)
                for slide_id, url, output_path in downloads
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        downloaded, failed = 0, 0
        failed_ids = []
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                logger.error(f"Download task failed with exception: {result}")
            else:
                slide_id, success, error = result
                if success:
                    downloaded += 1
                else:
                    failed += 1
                    failed_ids.append(slide_id)
        
        return downloaded, failed, failed_ids

    def download_slides(
        self,
        slide_ids: set[str],
        output_dir: Path,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        datasets: list[str] | None = None,
        **kwargs
    ) -> None:
        """Download specific Bioimage slides."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # We need to know which task each slide belongs to to determine URL
        # and extension.
        if not tasks_dir:
            logger.error("tasks_dir is required for BioimageProvider to determine slide URLs")
            return

        slide_to_task_info = {}
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            ds_name = tsv_path.parent.parent.name
            if ds_name not in self.datasets:
                continue
            if datasets and ds_name not in datasets:
                continue
            
            task_name = tsv_path.parent.name
            df = self._extract_slide_ids_from_tsv(tsv_path)
            for sid in df["slide_id"].unique():
                if sid in slide_ids:
                    # Prefer braf_ or valentino specifically if multiple tasks share slides
                    if sid not in slide_to_task_info:
                        slide_to_task_info[sid] = (ds_name, task_name)
                    else:
                        # If already there, check if we can get a better match
                        curr_ds, curr_task = slide_to_task_info[sid]
                        if "braf" in task_name.lower() or "valentino" in task_name.lower():
                            slide_to_task_info[sid] = (ds_name, task_name)

        downloads = []
        skipped = 0
        for slide_id in sorted(slide_ids):
            if slide_id not in slide_to_task_info:
                logger.warning(f"Could not find task info for slide {slide_id}, skipping")
                continue
            
            ds, task = slide_to_task_info[slide_id]
            url = self._get_slide_url(ds, task, slide_id)
            extension = ".tif" if url.endswith(".tif") else ".czi"
            
            # Organize by dataset subdirectory
            target_path = output_dir / ds / f"{slide_id}{extension}"
            
            if target_path.exists():
                skipped += 1
                continue
            
            downloads.append((slide_id, url, target_path))
            
        if skipped > 0:
            logger.info(f"Skipping {skipped} already downloaded slides")
            
        if downloads:
            logger.info(f"Downloading {len(downloads)} slides...")
            downloaded, failed, failed_ids = asyncio.run(self._download_slides_async(downloads))
            logger.info(f"Download complete. Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
        else:
            logger.info("All slides already downloaded")

        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir, datasets)

    def download_full(
        self,
        output_dir: Path,
        datasets: list[str] | None = None,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Download full Bioimage datasets."""
        if not tasks_dir:
            logger.error("tasks_dir is required for BioimageProvider")
            return
            
        slide_ids_by_task = self.get_slide_ids_for_tasks(tasks_dir, datasets=datasets)
        all_slides = set()
        for ids in slide_ids_by_task.values():
            all_slides.update(ids)
            
        if all_slides:
            self.download_slides(all_slides, output_dir, create_symlinks=create_symlinks, tasks_dir=tasks_dir, datasets=datasets)

    def get_storage_directories(self, output_dir: Path, datasets: list[str] | None = None) -> list[Path]:
        """Get storage directories for Bioimage datasets."""
        targets = datasets if datasets else self.datasets
        return [output_dir / d for d in targets]

    def _create_symlinks(self, tasks_dir: Path, slides_dir: Path, datasets: list[str] | None = None) -> None:
        """Create per-task symlink directories."""
        slide_ids_by_task = self.get_slide_ids_for_tasks(tasks_dir, datasets=datasets)
        
        for task_key, slide_ids in slide_ids_by_task.items():
            ds_name, task_name = task_key.split("/")
            task_dir = slides_dir / "by_task" / ds_name / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            symlink_count = 0
            ds_storage_dir = slides_dir / ds_name
            if not ds_storage_dir.exists():
                continue
                
            # Check for both .czi and .tif
            for slide_id in slide_ids:
                for ext in [".czi", ".tif"]:
                    source_file = ds_storage_dir / f"{slide_id}{ext}"
                    if source_file.exists():
                        symlink_path = task_dir / source_file.name
                        if not symlink_path.exists():
                            symlink_path.symlink_to(source_file.resolve())
                            symlink_count += 1
                        break
            
            if symlink_count > 0:
                logger.info(f"  {task_key}: {symlink_count} symlinks")
