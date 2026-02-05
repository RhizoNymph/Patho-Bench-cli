"""NADT-PROSTATE dataset provider using TCIA PathDB API."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tcia_utils import pathdb

from patho_bench_cli.providers.base import DatasetProvider
from patho_bench_cli.utils import download_file_with_retry, DEFAULT_MAX_RETRIES

logging.basicConfig()
logger = logging.getLogger(__name__)

NADT_COLLECTION = "NADT-PROSTATE"
NADT_DATASETS = ["nadt"]


class NADTProstateProvider(DatasetProvider):
    """Provider for NADT-PROSTATE dataset from TCIA."""

    @property
    def name(self) -> str:
        return "nadt_prostate"

    @property
    def description(self) -> str:
        return "NADT-PROSTATE: Neoadjuvant Androgen Deprivation Therapy dataset from TCIA"

    @property
    def datasets(self) -> list[str]:
        return NADT_DATASETS

    def get_storage_directories(self, output_dir: Path, datasets: list[str] | None = None) -> list[Path]:
        """Get the NADT-PROSTATE subdirectory."""
        return [output_dir / NADT_COLLECTION]

    def _get_all_tsv_files(self, tasks_dir: Path) -> list[Path]:
        """Find all k=all.tsv files in the nadt tasks directory."""
        nadt_tasks_path = tasks_dir / "nadt"
        if not nadt_tasks_path.exists():
            return []
        return list(nadt_tasks_path.glob("**/k=all.tsv"))

    def _extract_slide_ids_from_tsv(self, tsv_path: Path) -> pd.DataFrame:
        """Extract case_id and slide_id from a TSV file."""
        df = pd.read_csv(tsv_path, sep="\t")
        if "slide_id" in df.columns and "case_id" in df.columns:
            return df[["case_id", "slide_id"]].drop_duplicates()
        return pd.DataFrame(columns=["case_id", "slide_id"])

    def list_tasks(self, tasks_dir: Path, datasets: list[str] | None = None) -> list[dict[str, Any]]:
        """List all available NADT-PROSTATE tasks."""
        tasks = []

        if datasets and "nadt" not in datasets:
            return tasks

        for tsv_path in self._get_all_tsv_files(tasks_dir):
            task_name = tsv_path.parent.name
            df = self._extract_slide_ids_from_tsv(tsv_path)

            tasks.append({
                "dataset": "nadt",
                "task": task_name,
                "n_slides": len(df),
                "n_cases": df["case_id"].nunique() if "case_id" in df.columns else 0,
                "tcia_collection": NADT_COLLECTION,
            })
        return tasks

    def get_slide_ids_for_tasks(
        self,
        tasks_dir: Path,
        datasets: list[str] | None = None
    ) -> dict[str, set[str]]:
        """Get slide IDs needed for NADT-PROSTATE Patho-Bench tasks."""
        result: dict[str, set[str]] = {}

        if datasets and "nadt" not in datasets:
            return result

        for tsv_path in self._get_all_tsv_files(tasks_dir):
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)

            if "nadt" not in result:
                result["nadt"] = set()
            result["nadt"].update(slide_df["slide_id"].unique())

        return result

    def _query_tcia_images(self, cache_dir: Path) -> pd.DataFrame:
        """Query TCIA for images in the collection, with caching."""
        cache_file = cache_dir / f"{NADT_COLLECTION.replace('-', '_')}_images.csv"

        if cache_file.exists():
            logger.info(f"Loading cached images for {NADT_COLLECTION}")
            return pd.read_csv(cache_file)

        logger.info(f"Querying TCIA for collection: {NADT_COLLECTION}")
        try:
            images = pathdb.getImages(NADT_COLLECTION, format="df")
            if images is not None and not images.empty:
                cache_dir.mkdir(parents=True, exist_ok=True)
                images.to_csv(cache_file, index=False)
                logger.info(f"Cached {len(images)} images to {cache_file}")
            return images if images is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to query {NADT_COLLECTION}: {e}")
            return pd.DataFrame()

    def _match_slides_to_tcia(
        self,
        needed_slide_ids: set[str],
        tcia_images: pd.DataFrame
    ) -> pd.DataFrame:
        """Match needed slides to TCIA images by filename stem."""
        if tcia_images.empty:
            return pd.DataFrame()

        # Extract filename stems from TCIA URLs for matching
        if "imageUrl" not in tcia_images.columns:
            return pd.DataFrame()

        def get_stem(url: str) -> str:
            filename = url.split("/")[-1]
            # Remove extension
            return Path(filename).stem

        tcia_images = tcia_images.copy()
        tcia_images["filename_stem"] = tcia_images["imageUrl"].apply(get_stem)

        # Match by slide_id (which corresponds to filename stem)
        matched = tcia_images[tcia_images["filename_stem"].isin(needed_slide_ids)]
        return matched

    def _get_expected_files(self, images_df: pd.DataFrame) -> dict[str, str]:
        """Get mapping of expected filenames to their download URLs."""
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
        """Retry downloading files that are missing from output_dir."""
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
        """Download specific NADT-PROSTATE slides from TCIA."""
        if cache_dir is None:
            cache_dir = output_dir.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        tcia_images = self._query_tcia_images(cache_dir)
        if tcia_images.empty:
            logger.warning(f"No images found for {NADT_COLLECTION}")
            return

        matched = self._match_slides_to_tcia(slide_ids, tcia_images)
        if matched.empty:
            logger.warning("No matching slides found in TCIA")
            return

        collection_dir = output_dir / NADT_COLLECTION
        collection_dir.mkdir(parents=True, exist_ok=True)

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
        """Download complete NADT-PROSTATE collection from TCIA."""
        if cache_dir is None:
            cache_dir = output_dir.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        tcia_images = self._query_tcia_images(cache_dir)
        if tcia_images.empty:
            logger.warning(f"No images found for {NADT_COLLECTION}")
            return

        collection_dir = output_dir / NADT_COLLECTION
        collection_dir.mkdir(parents=True, exist_ok=True)

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
        """Create per-task symlink directories for NADT-PROSTATE slides."""
        tsv_files = self._get_all_tsv_files(tasks_dir)
        if not tsv_files:
            logger.warning(f"No task TSV files found in {tasks_dir / 'nadt'}. Skipping symlink creation.")
            return

        collection_dir = slides_dir / NADT_COLLECTION
        if not collection_dir.exists():
            logger.warning(f"NADT-PROSTATE slides directory not found at {collection_dir}. Skipping symlink creation.")
            return

        # Get all available files in collection directory
        available_files = {f.stem: f for f in collection_dir.glob("*") if f.is_file()}
        if not available_files:
            logger.warning(f"No files found in {collection_dir}. Skipping symlink creation.")
            return

        logger.info(f"Found {len(available_files)} files in {collection_dir}")
        # Debug: show a sample of available file stems
        sample_stems = list(available_files.keys())[:3]
        logger.info(f"Sample file stems: {sample_stems}")

        for tsv_path in tsv_files:
            dataset_name = tsv_path.parent.parent.name
            task_name = tsv_path.parent.name

            # Only process nadt dataset
            if dataset_name != "nadt":
                continue
            if datasets and dataset_name not in datasets:
                continue

            # Get slide IDs needed for this specific task
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)
            task_slide_ids = set(slide_df["slide_id"].unique())

            # Debug: show sample slide IDs from task
            sample_ids = list(task_slide_ids)[:3]
            logger.info(f"Sample slide_ids from task: {sample_ids}")
            # Check for any matches
            matches = task_slide_ids & set(available_files.keys())
            logger.info(f"Direct matches: {len(matches)} out of {len(task_slide_ids)}")

            task_dir = slides_dir / "by_task" / dataset_name / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            symlink_count = 0
            existing_count = 0
            for slide_id in task_slide_ids:
                if slide_id in available_files:
                    img_file = available_files[slide_id]
                    symlink_path = task_dir / img_file.name
                    if symlink_path.is_symlink() or symlink_path.exists():
                        existing_count += 1
                    else:
                        symlink_path.symlink_to(img_file.resolve())
                        symlink_count += 1

            if symlink_count > 0 or existing_count > 0:
                logger.info(f"  {dataset_name}/{task_name}: {symlink_count} new symlinks, {existing_count} already existed")
            else:
                logger.warning(f"  {dataset_name}/{task_name}: No matching slides found (needed {len(task_slide_ids)}, available {len(available_files)})")
