"""IDR (Image Data Resource) dataset provider using BioImage Archive downloads."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import pandas as pd
import requests
from aiolimiter import AsyncLimiter

from patho_bench_cli.providers.base import DatasetProvider

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# BioImage Archive base URL for IDR data
BIA_BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire"

# Download settings
DEFAULT_CONCURRENT_DOWNLOADS = 8
DEFAULT_RATE_LIMIT = 10  # requests per second
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1MB chunks

# Dataset configuration for IDR datasets
# Maps patho-bench dataset names to BioImage Archive paths
IDR_DATASETS = {
    "ucla_lung": {
        "idr_study": "idr0082",
        "project_name": "idr0082-pennycuick-lesions/experimentA",
        "project_id": 1251,
        "dataset_name": "Lung Carcinoma",
        "dataset_id": 10801,
        # BioImage Archive accession and path
        "bia_accession": "S-BIAD509",
        "bia_path": "S-BIAD/509/S-BIAD509/Files/20200517-ftp",
        # Image names are like "S11_HandE.ndpi", slide_ids are "S11_HandE"
        "slide_id_pattern": r"(S\d+_HandE)",
        "file_extension": ".ndpi",
    }
}


class IDRProvider(DatasetProvider):
    """Provider for datasets from the Image Data Resource (IDR) via BioImage Archive."""
    
    def __init__(
        self,
        concurrent_downloads: int = DEFAULT_CONCURRENT_DOWNLOADS,
        rate_limit: float = DEFAULT_RATE_LIMIT,
    ):
        """
        Initialize the IDR provider.
        
        Args:
            concurrent_downloads: Maximum number of concurrent downloads
            rate_limit: Maximum requests per second
        """
        self.concurrent_downloads = concurrent_downloads
        self.rate_limit = rate_limit
    
    @property
    def name(self) -> str:
        return "idr"
    
    @property
    def description(self) -> str:
        return "Image Data Resource (OpenMicroscopy) datasets via BioImage Archive"
    
    @property
    def datasets(self) -> list[str]:
        return list(IDR_DATASETS.keys())
    
    def _get_all_tsv_files(self, tasks_dir: Path) -> list[Path]:
        """Find all k=all.tsv files in the tasks directory."""
        return list(tasks_dir.glob("**/k=all.tsv"))
    
    def _extract_slide_ids_from_tsv(self, tsv_path: Path) -> pd.DataFrame:
        """Extract case_id and slide_id from a TSV file."""
        df = pd.read_csv(tsv_path, sep="\t")
        if "slide_id" in df.columns and "case_id" in df.columns:
            return df[["case_id", "slide_id"]].drop_duplicates()
        return pd.DataFrame(columns=["case_id", "slide_id"])
    
    def _slide_id_from_filename(self, filename: str, dataset_name: str) -> str | None:
        """
        Extract slide_id from a filename.
        
        Args:
            filename: The filename (e.g., "S135_HandE.ndpi")
            dataset_name: The patho-bench dataset name (e.g., "ucla_lung")
            
        Returns:
            The slide_id (e.g., "S135_HandE") or None if no match.
        """
        config = IDR_DATASETS.get(dataset_name)
        if not config:
            return None
        
        pattern = config.get("slide_id_pattern", r"(S\d+_HandE)")
        
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None
    
    def _get_download_url(self, slide_id: str, dataset_name: str) -> str | None:
        """
        Get the download URL for a slide.
        
        Args:
            slide_id: The slide_id (e.g., "S135_HandE")
            dataset_name: The patho-bench dataset name
            
        Returns:
            The download URL or None if not available.
        """
        config = IDR_DATASETS.get(dataset_name)
        if not config:
            return None
        
        bia_path = config["bia_path"]
        ext = config.get("file_extension", ".ndpi")
        filename = f"{slide_id}{ext}"
        
        return f"{BIA_BASE_URL}/{bia_path}/{filename}"
    
    def _list_available_files(self, dataset_name: str) -> list[dict]:
        """
        List all available files for a dataset from BioImage Archive.
        
        Args:
            dataset_name: The patho-bench dataset name
            
        Returns:
            List of dicts with slide_id, filename, url
        """
        config = IDR_DATASETS.get(dataset_name)
        if not config:
            return []
        
        bia_path = config["bia_path"]
        ext = config.get("file_extension", ".ndpi")
        base_url = f"{BIA_BASE_URL}/{bia_path}/"
        
        logger.info(f"Listing files from BioImage Archive: {base_url}")
        
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            
            # Parse directory listing (simple HTML parsing)
            files = []
            for match in re.finditer(r'href="([^"]+)"', response.text):
                filename = match.group(1)
                if filename.endswith(ext) and not filename.endswith('.ndpa'):
                    slide_id = self._slide_id_from_filename(filename, dataset_name)
                    if slide_id:
                        files.append({
                            "slide_id": slide_id,
                            "filename": filename,
                            "url": f"{base_url}{filename}",
                        })
            
            logger.info(f"Found {len(files)} slide files")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def _query_available_slides(
        self,
        dataset_name: str,
        cache_dir: Path | None = None
    ) -> dict[str, dict]:
        """
        Query available slides for a dataset.
        
        Args:
            dataset_name: The patho-bench dataset name (e.g., "ucla_lung")
            cache_dir: Optional directory to cache results
            
        Returns:
            Dict mapping slide_id to file info (url, filename, etc.)
        """
        config = IDR_DATASETS.get(dataset_name)
        if not config:
            raise ValueError(f"Unknown IDR dataset: {dataset_name}")
        
        # Check cache first
        if cache_dir:
            cache_file = cache_dir / f"idr_{dataset_name}_files.csv"
            if cache_file.exists():
                logger.info(f"Loading cached file list for {dataset_name}")
                df = pd.read_csv(cache_file)
                return {
                    row["slide_id"]: {
                        "filename": row["filename"],
                        "url": row["url"],
                    }
                    for _, row in df.iterrows()
                }
        
        # List files from BioImage Archive
        files = self._list_available_files(dataset_name)
        
        # Build mapping
        slides_info = {
            f["slide_id"]: {
                "filename": f["filename"],
                "url": f["url"],
            }
            for f in files
        }
        
        # Cache results
        if cache_dir and slides_info:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = [
                {
                    "slide_id": sid,
                    "filename": info["filename"],
                    "url": info["url"],
                }
                for sid, info in slides_info.items()
            ]
            pd.DataFrame(cache_data).to_csv(cache_file, index=False)
            logger.info(f"Cached file list to {cache_file}")
        
        return slides_info
    
    def list_tasks(self, tasks_dir: Path) -> list[dict[str, Any]]:
        """List all available IDR tasks."""
        tasks = []
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            if dataset_name not in IDR_DATASETS:
                continue
            
            task_name = tsv_path.parent.name
            df = self._extract_slide_ids_from_tsv(tsv_path)
            config = IDR_DATASETS[dataset_name]
            
            tasks.append({
                "dataset": dataset_name,
                "task": task_name,
                "n_slides": len(df),
                "n_cases": df["case_id"].nunique() if "case_id" in df.columns else 0,
                "idr_study": config["idr_study"],
                "bia_accession": config["bia_accession"],
            })
        return tasks
    
    def get_slide_ids_for_tasks(
        self,
        tasks_dir: Path,
        datasets: list[str] | None = None
    ) -> dict[str, set[str]]:
        """Get slide IDs needed for IDR Patho-Bench tasks."""
        result: dict[str, set[str]] = {}
        
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            
            # Only process IDR datasets
            if dataset_name not in IDR_DATASETS:
                continue
            
            # Filter to requested datasets
            if datasets and dataset_name not in datasets:
                continue
            
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            
            if dataset_name not in result:
                result[dataset_name] = set()
            result[dataset_name].update(slide_df["slide_id"].unique())
        
        return result
    
    async def _download_file_async(
        self,
        session: aiohttp.ClientSession,
        limiter: AsyncLimiter,
        url: str,
        output_path: Path,
        slide_id: str,
    ) -> tuple[str, bool, str | None]:
        """
        Download a single file asynchronously with rate limiting.
        
        Args:
            session: aiohttp client session
            limiter: Rate limiter
            url: URL to download from
            output_path: Path to save the file
            slide_id: Slide ID for logging
            
        Returns:
            Tuple of (slide_id, success, error_message)
        """
        async with limiter:
            try:
                logger.info(f"Downloading {slide_id}...")
                
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    # Get total size for progress
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    # Write to temp file first for atomicity
                    temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
                    
                    async with aiofiles.open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(DEFAULT_CHUNK_SIZE):
                            await f.write(chunk)
                            downloaded += len(chunk)
                    
                    # Rename temp to final
                    temp_path.rename(output_path)
                    
                    size_mb = downloaded / (1024 * 1024)
                    logger.info(f"  Completed {slide_id} ({size_mb:.1f} MB)")
                    return (slide_id, True, None)
                    
            except asyncio.CancelledError:
                # Clean up temp file on cancellation
                temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
                if temp_path.exists():
                    temp_path.unlink()
                raise
            except Exception as e:
                # Clean up temp file on error
                temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
                if temp_path.exists():
                    temp_path.unlink()
                error_msg = str(e)
                logger.error(f"  Failed {slide_id}: {error_msg}")
                return (slide_id, False, error_msg)
    
    async def _download_files_async(
        self,
        downloads: list[tuple[str, str, Path]],  # (slide_id, url, output_path)
    ) -> tuple[int, int, list[str]]:
        """
        Download multiple files concurrently.
        
        Args:
            downloads: List of (slide_id, url, output_path) tuples
            
        Returns:
            Tuple of (downloaded_count, failed_count, failed_ids)
        """
        limiter = AsyncLimiter(self.rate_limit, 1.0)  # rate_limit requests per second
        semaphore = asyncio.Semaphore(self.concurrent_downloads)
        
        async def bounded_download(session, slide_id, url, output_path):
            async with semaphore:
                return await self._download_file_async(
                    session, limiter, url, output_path, slide_id
                )
        
        connector = aiohttp.TCPConnector(limit=self.concurrent_downloads)
        timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout per file
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                bounded_download(session, slide_id, url, output_path)
                for slide_id, url, output_path in downloads
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        downloaded = 0
        failed = 0
        failed_ids = []
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                failed_ids.append(str(result))
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
        cache_dir: Path | None = None,
        datasets: list[str] | None = None,
        **kwargs
    ) -> None:
        """Download specific slides from BioImage Archive concurrently."""
        if cache_dir is None:
            cache_dir = output_dir.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which IDR datasets to query
        datasets_to_query = datasets if datasets else list(IDR_DATASETS.keys())
        
        for dataset_name in datasets_to_query:
            if dataset_name not in IDR_DATASETS:
                continue
            
            config = IDR_DATASETS[dataset_name]
            logger.info(f"Processing IDR dataset: {config['dataset_name']} ({config['bia_accession']})")
            
            # Query available files
            available_slides = self._query_available_slides(dataset_name, cache_dir)
            
            # Create dataset output directory
            dataset_dir = output_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Build download list
            downloads = []
            skipped = 0
            not_found = 0
            
            for slide_id in slide_ids:
                if slide_id not in available_slides:
                    not_found += 1
                    continue
                
                file_info = available_slides[slide_id]
                filename = file_info["filename"]
                url = file_info["url"]
                output_path = dataset_dir / filename
                
                if output_path.exists():
                    skipped += 1
                    continue
                
                downloads.append((slide_id, url, output_path))
            
            if downloads:
                logger.info(
                    f"Starting concurrent download of {len(downloads)} files "
                    f"({self.concurrent_downloads} concurrent, {self.rate_limit} req/s limit)"
                )
                
                # Run async downloads
                downloaded, failed, failed_ids = asyncio.run(
                    self._download_files_async(downloads)
                )
            else:
                downloaded = 0
                failed = 0
            
            logger.info(
                f"Download complete for {dataset_name}: "
                f"{downloaded} downloaded, {skipped} skipped, {failed} failed, {not_found} not found"
            )
        
        # Create symlinks if requested
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir, datasets)
    
    def download_full(
        self,
        output_dir: Path,
        datasets: list[str] | None = None,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        cache_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Download complete IDR dataset(s) from BioImage Archive concurrently."""
        if cache_dir is None:
            cache_dir = output_dir.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        datasets_to_download = datasets if datasets else list(IDR_DATASETS.keys())
        
        for dataset_name in datasets_to_download:
            if dataset_name not in IDR_DATASETS:
                logger.warning(f"Unknown IDR dataset: {dataset_name}")
                continue
            
            config = IDR_DATASETS[dataset_name]
            logger.info(f"Downloading full dataset: {config['dataset_name']} ({config['bia_accession']})")
            
            # Query all available files
            available_slides = self._query_available_slides(dataset_name, cache_dir)
            
            # Create dataset output directory
            dataset_dir = output_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Build download list
            downloads = []
            skipped = 0
            
            for slide_id, file_info in available_slides.items():
                filename = file_info["filename"]
                url = file_info["url"]
                output_path = dataset_dir / filename
                
                if output_path.exists():
                    skipped += 1
                    continue
                
                downloads.append((slide_id, url, output_path))
            
            if downloads:
                logger.info(
                    f"Starting concurrent download of {len(downloads)} files "
                    f"({self.concurrent_downloads} concurrent, {self.rate_limit} req/s limit)"
                )
                
                # Run async downloads
                downloaded, failed, failed_ids = asyncio.run(
                    self._download_files_async(downloads)
                )
            else:
                downloaded = 0
                failed = 0
            
            logger.info(
                f"Download complete for {dataset_name}: "
                f"{downloaded} downloaded, {skipped} skipped, {failed} failed"
            )
        
        # Create symlinks if requested
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir, datasets)
    
    def _create_symlinks(
        self,
        tasks_dir: Path,
        slides_dir: Path,
        datasets: list[str] | None = None
    ) -> None:
        """Create per-task symlink directories with only task-specific slides."""
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            task_name = tsv_path.parent.name
            
            if dataset_name not in IDR_DATASETS:
                continue
            if datasets and dataset_name not in datasets:
                continue
            
            dataset_dir = slides_dir / dataset_name
            if not dataset_dir.exists():
                continue
            
            # Get slide IDs needed for this specific task
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            task_slide_ids = set(slide_df["slide_id"].unique())
            
            task_dir = slides_dir / "by_task" / dataset_name / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            symlink_count = 0
            for img_file in dataset_dir.glob("*"):
                if img_file.is_file():
                    # Extract slide_id from filename using provider's method
                    file_slide_id = self._slide_id_from_filename(img_file.name, dataset_name)
                    if file_slide_id and file_slide_id in task_slide_ids:
                        symlink_path = task_dir / img_file.name
                        if not symlink_path.exists():
                            symlink_path.symlink_to(img_file.resolve())
                            symlink_count += 1
            
            if symlink_count > 0:
                logger.info(f"  {dataset_name}/{task_name}: {symlink_count} symlinks")
