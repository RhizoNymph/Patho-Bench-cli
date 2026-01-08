"""POST-NAT-BRCA dataset provider using TCIA PathDB API."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tcia_utils import pathdb

from patho_bench_cli.providers.base import DatasetProvider
from patho_bench_cli.utils import download_file_with_retry, DEFAULT_MAX_RETRIES

logging.basicConfig()
logger = logging.getLogger(__name__)

# TCIA collection name for POST-NAT-BRCA
POST_NAT_BRCA_COLLECTION = "Post-NAT-BRCA"

# Mapping from Patho-Bench dataset names to TCIA collection names
POST_NAT_BRCA_COLLECTION_MAP = {
    "post_nat_brca": POST_NAT_BRCA_COLLECTION,
}


class PostNatBrcaProvider(DatasetProvider):
    """Provider for POST-NAT-BRCA dataset from TCIA."""
    
    @property
    def name(self) -> str:
        return "post_nat_brca"
    
    @property
    def description(self) -> str:
        return "POST-NAT-BRCA: Assessment of Residual Breast Cancer Cellularity after Neoadjuvant Chemotherapy from TCIA"
    
    @property
    def datasets(self) -> list[str]:
        return list(POST_NAT_BRCA_COLLECTION_MAP.keys())
    
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
        """List all available POST-NAT-BRCA tasks."""
        tasks = []
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            if dataset_name not in POST_NAT_BRCA_COLLECTION_MAP:
                continue
            
            task_name = tsv_path.parent.name
            df = self._extract_slide_ids_from_tsv(tsv_path)
            tcia_collection = POST_NAT_BRCA_COLLECTION_MAP.get(dataset_name)
            
            tasks.append({
                "dataset": dataset_name,
                "task": task_name,
                "n_slides": len(df),
                "n_cases": df["case_id"].nunique() if "case_id" in df.columns else 0,
                "tcia_collection": tcia_collection,
            })
        return tasks
    
    def get_slide_ids_for_tasks(
        self,
        tasks_dir: Path,
        datasets: list[str] | None = None
    ) -> dict[str, set[str]]:
        """Get slide IDs needed for POST-NAT-BRCA Patho-Bench tasks."""
        result: dict[str, set[str]] = {}
        
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            
            # Only process POST-NAT-BRCA datasets
            if dataset_name not in POST_NAT_BRCA_COLLECTION_MAP:
                continue
            
            # Filter to requested datasets
            if datasets and dataset_name not in datasets:
                continue
            
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            
            if dataset_name not in result:
                result[dataset_name] = set()
            result[dataset_name].update(slide_df["slide_id"].unique())
        
        return result
    
    def _query_tcia_images(
        self,
        collection: str,
        cache_dir: Path
    ) -> pd.DataFrame:
        """Query TCIA for images in a collection, with caching."""
        cache_file = cache_dir / f"{collection.replace('-', '_')}_images.csv"
        
        if cache_file.exists():
            logger.info(f"Loading cached images for {collection}")
            return pd.read_csv(cache_file)
        
        logger.info(f"Querying TCIA for collection: {collection}")
        try:
            images = pathdb.getImages(collection, format="df")
            if images is not None and not images.empty:
                cache_dir.mkdir(parents=True, exist_ok=True)
                images.to_csv(cache_file, index=False)
                logger.info(f"Cached {len(images)} images to {cache_file}")
            return images if images is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to query {collection}: {e}")
            return pd.DataFrame()
    
    def _match_slides_to_tcia(
        self,
        needed_slide_ids: set[str],
        tcia_images: pd.DataFrame
    ) -> pd.DataFrame:
        """Match needed slides to TCIA images by filename."""
        if tcia_images.empty:
            return pd.DataFrame()
        
        # Match by extracting filename stem from imageUrl
        if "imageUrl" in tcia_images.columns:
            tcia_images = tcia_images.copy()
            tcia_images["filename_stem"] = tcia_images["imageUrl"].apply(
                lambda x: Path(x.split("/")[-1]).stem if pd.notna(x) else ""
            )
            return tcia_images[tcia_images["filename_stem"].isin(needed_slide_ids)]
        
        return pd.DataFrame()
    
    def _get_expected_files(self, images_df: pd.DataFrame) -> dict[str, str]:
        """
        Get mapping of expected filenames to their download URLs.
        
        Args:
            images_df: DataFrame with imageUrl column.
            
        Returns:
            Dict mapping filename to URL.
        """
        expected = {}
        if "imageUrl" in images_df.columns:
            for url in images_df["imageUrl"]:
                filename = url.split("/")[-1]
                expected[filename] = url
        return expected
    
    def _retry_failed_downloads(
        self,
        expected_files: dict[str, str],
        output_dir: Path,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> tuple[int, int]:
        """
        Retry downloading files that are missing from output_dir.
        
        Args:
            expected_files: Dict mapping filename to URL.
            output_dir: Directory where files should be saved.
            max_retries: Maximum retry attempts per file.
            
        Returns:
            Tuple of (successfully_recovered, still_failed).
        """
        missing = []
        for filename, url in expected_files.items():
            filepath = output_dir / filename
            if not filepath.exists():
                missing.append((filename, url))
        
        if not missing:
            return 0, 0
        
        logger.info(f"Retrying {len(missing)} failed downloads...")
        recovered = 0
        failed = 0
        
        for filename, url in missing:
            filepath = output_dir / filename
            logger.info(f"Retrying: {filename}")
            if download_file_with_retry(url, filepath, max_retries=max_retries):
                recovered += 1
            else:
                failed += 1
        
        logger.info(f"Retry complete. Recovered: {recovered}, Still failed: {failed}")
        return recovered, failed
    
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
        """Download specific POST-NAT-BRCA slides from TCIA."""
        if cache_dir is None:
            cache_dir = output_dir.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        collection = POST_NAT_BRCA_COLLECTION
        collection_dir = output_dir / collection
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        tcia_images = self._query_tcia_images(collection, cache_dir)
        if tcia_images.empty:
            logger.warning(f"No images found for {collection}")
            return
        
        matched = self._match_slides_to_tcia(slide_ids, tcia_images)
        if matched.empty:
            logger.warning(f"No matching slides found in {collection}")
            return
        
        # Get expected files before download for retry tracking
        expected_files = self._get_expected_files(matched)
        
        logger.info(f"Downloading {len(matched)} images to {collection_dir}")
        pathdb.downloadImages(matched, path=str(collection_dir))
        
        # Retry any failed downloads
        self._retry_failed_downloads(expected_files, collection_dir)
        
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
        """Download complete POST-NAT-BRCA collection from TCIA."""
        if cache_dir is None:
            cache_dir = output_dir.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        collection = POST_NAT_BRCA_COLLECTION
        collection_dir = output_dir / collection
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        tcia_images = self._query_tcia_images(collection, cache_dir)
        if tcia_images.empty:
            logger.warning(f"No images found for {collection}")
            return
        
        # Get expected files before download for retry tracking
        expected_files = self._get_expected_files(tcia_images)
        
        logger.info(f"Downloading ALL {len(tcia_images)} images to {collection_dir}")
        pathdb.downloadImages(tcia_images, path=str(collection_dir))
        
        # Retry any failed downloads
        self._retry_failed_downloads(expected_files, collection_dir)
        
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
            
            if dataset_name not in POST_NAT_BRCA_COLLECTION_MAP:
                continue
            if datasets and dataset_name not in datasets:
                continue
            
            collection = POST_NAT_BRCA_COLLECTION_MAP.get(dataset_name)
            if not collection:
                continue
            
            collection_dir = slides_dir / collection
            if not collection_dir.exists():
                continue
            
            # Get slide IDs needed for this specific task
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            task_slide_ids = set(slide_df["slide_id"].unique())
            
            task_dir = slides_dir / "by_task" / dataset_name / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            symlink_count = 0
            for img_file in collection_dir.glob("*"):
                if img_file.is_file():
                    # Check if this file's stem matches a task slide_id
                    if img_file.stem in task_slide_ids:
                        symlink_path = task_dir / img_file.name
                        if not symlink_path.exists():
                            symlink_path.symlink_to(img_file.resolve())
                            symlink_count += 1
            
            if symlink_count > 0:
                logger.info(f"  {dataset_name}/{task_name}: {symlink_count} symlinks")

