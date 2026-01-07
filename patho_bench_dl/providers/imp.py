"""IMP dataset provider for direct HTTP download with concurrent downloads."""

import asyncio
import logging
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import aiofiles
import aiohttp
import pandas as pd
from aiolimiter import AsyncLimiter

from patho_bench_dl.providers.base import DatasetProvider

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

# Download settings
DEFAULT_CONCURRENT_DOWNLOADS = 4
DEFAULT_RATE_LIMIT = 5  # requests per second
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1MB chunks


class IMPProvider(DatasetProvider):
    """Provider for IMP-CRS2024 dataset via direct HTTP download with concurrency."""
    
    def __init__(
        self,
        concurrent_downloads: int = DEFAULT_CONCURRENT_DOWNLOADS,
        rate_limit: int = DEFAULT_RATE_LIMIT,
    ):
        """
        Initialize the IMP provider.
        
        Args:
            concurrent_downloads: Maximum number of concurrent downloads.
            rate_limit: Maximum requests per second.
        """
        self.concurrent_downloads = concurrent_downloads
        self.rate_limit = rate_limit
    
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
    
    def _get_slide_urls(self, slide_id: str) -> list[str]:
        """
        Get all possible URLs for a slide (one per folder).
        
        Slides are named like CRC_XXXX.svs and could be in any of the folders.
        """
        filename = f"{slide_id}{SLIDE_EXTENSION}"
        return [urljoin(BASE_URL, f"{folder}/{filename}") for folder in SLIDE_FOLDERS]
    
    async def _check_url_exists(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> bool:
        """Check if a URL exists (HEAD request)."""
        try:
            async with session.head(url) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _download_file_async(
        self,
        session: aiohttp.ClientSession,
        limiter: AsyncLimiter,
        url: str,
        output_path: Path,
        slide_id: str,
    ) -> tuple[str, bool, str | None]:
        """
        Download a single file asynchronously.
        
        Args:
            session: aiohttp session
            limiter: Rate limiter
            url: URL to download from
            output_path: Path to save the file
            slide_id: Slide identifier for logging
            
        Returns:
            Tuple of (slide_id, success, error_message)
        """
        temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
        
        try:
            async with limiter:
                async with session.get(url) as response:
                    if response.status == 404:
                        return (slide_id, False, "not_found")
                    
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}"
                        logger.error(f"  Failed {slide_id}: {error_msg}")
                        return (slide_id, False, error_msg)
                    
                    # Stream to temp file
                    async with aiofiles.open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(DEFAULT_CHUNK_SIZE):
                            await f.write(chunk)
                    
                    # Move temp to final
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
            error_msg = str(e)
            return (slide_id, False, error_msg)
    
    async def _try_download_from_folders(
        self,
        session: aiohttp.ClientSession,
        limiter: AsyncLimiter,
        slide_id: str,
        output_path: Path,
    ) -> tuple[str, bool, str | None]:
        """
        Try to download a slide from any of the possible folders.
        
        Returns:
            Tuple of (slide_id, success, error_message)
        """
        urls = self._get_slide_urls(slide_id)
        
        for url in urls:
            slide_id_result, success, error = await self._download_file_async(
                session, limiter, url, output_path, slide_id
            )
            if success:
                return (slide_id, True, None)
            # Only continue trying if the error was "not found"
            if error != "not_found":
                return (slide_id, False, error)
        
        return (slide_id, False, "Slide not found in any folder")
    
    async def _download_slides_async(
        self,
        downloads: list[tuple[str, Path]],  # (slide_id, output_path)
    ) -> tuple[int, int, list[str]]:
        """
        Download multiple slides concurrently.
        
        Args:
            downloads: List of (slide_id, output_path) tuples
            
        Returns:
            Tuple of (downloaded_count, failed_count, failed_ids)
        """
        limiter = AsyncLimiter(self.rate_limit, 1.0)
        semaphore = asyncio.Semaphore(self.concurrent_downloads)
        
        async def bounded_download(session, slide_id, output_path):
            async with semaphore:
                return await self._try_download_from_folders(
                    session, limiter, slide_id, output_path
                )
        
        # Create SSL context for aiohttp that skips verification
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(limit=self.concurrent_downloads, ssl=ssl_context)
        timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout per file
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                bounded_download(session, slide_id, output_path)
                for slide_id, output_path in downloads
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        downloaded = 0
        failed = 0
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
                    logger.warning(f"Could not download slide {slide_id}: {error}")
        
        return downloaded, failed, failed_ids
    
    def download_slides(
        self,
        slide_ids: set[str],
        output_dir: Path,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Download specific IMP slides concurrently."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build list of downloads needed
        downloads = []
        skipped = 0
        
        for slide_id in sorted(slide_ids):
            filename = f"{slide_id}{SLIDE_EXTENSION}"
            target_path = output_dir / filename
            
            if target_path.exists():
                skipped += 1
                continue
            
            downloads.append((slide_id, target_path))
        
        if skipped > 0:
            logger.info(f"Skipping {skipped} already downloaded slides")
        
        if downloads:
            logger.info(f"Downloading {len(downloads)} slides with {self.concurrent_downloads} concurrent connections...")
            downloaded, failed, failed_ids = asyncio.run(self._download_slides_async(downloads))
            logger.info(f"Download complete. Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
        else:
            logger.info("All slides already downloaded")
        
        # Create symlinks
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir)
    
    def download_full(
        self,
        output_dir: Path,
        datasets: list[str] | None = None,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
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
        effective_tasks_dir = tasks_dir or (output_dir.parent / "tasks")
        if effective_tasks_dir.exists():
            slide_ids_by_task = self.get_slide_ids_for_tasks(effective_tasks_dir)
            all_slides = set()
            for ids in slide_ids_by_task.values():
                all_slides.update(ids)
            
            if all_slides:
                logger.info(f"Found {len(all_slides)} slides from task files")
                self.download_slides(all_slides, output_dir)
                
                # Create symlinks if requested
                if create_symlinks and effective_tasks_dir.exists():
                    self._create_symlinks(effective_tasks_dir, output_dir)
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
