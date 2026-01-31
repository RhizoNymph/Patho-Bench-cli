"""EBRAINS DigitalBrainTumorAtlas dataset provider."""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from patho_bench_cli.providers.base import DatasetProvider
from patho_bench_cli.utils import DEFAULT_MAX_RETRIES

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# EBRAINS dataset configuration
DATASET_ID = "8fc108ab-e2b4-406f-8999-60269dc1f994"
DATASET_URL = "https://search.kg.ebrains.eu/instances/Dataset/8fc108ab-e2b4-406f-8999-60269dc1f994"
BASE_URL = "https://data-proxy.ebrains.eu/api/v1"
SLIDE_EXTENSION = ".ndpi"
FILE_PREFIX = "v1.0/"  # Files in the bucket are under v1.0/
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for streaming
MAX_WORKERS = 8


class EBRAINSProvider(DatasetProvider):
    """Provider for EBRAINS DigitalBrainTumorAtlas dataset."""

    @property
    def name(self) -> str:
        return "ebrains"

    @property
    def description(self) -> str:
        return "EBRAINS DigitalBrainTumorAtlas (brain tumor whole slide images)"

    @property
    def datasets(self) -> list[str]:
        return ["ebrains"]

    def _get_token(self) -> str:
        """Get EBRAINS authentication token from environment."""
        token = os.getenv("EBRAINS_AUTH_TOKEN")
        if not token:
            raise ValueError(
                "EBRAINS_AUTH_TOKEN environment variable not set.\n"
                f"You must first request access to the dataset at:\n  {DATASET_URL}\n"
                "Then set EBRAINS_AUTH_TOKEN in your .env file or environment."
            )
        return token

    def _warn_access_required(self) -> None:
        """Print warning about access requirements."""
        logger.warning(
            "EBRAINS DigitalBrainTumorAtlas requires requesting access first.\n"
            f"  Request access at: {DATASET_URL}\n"
            "  Once approved, set EBRAINS_AUTH_TOKEN in your .env file."
        )

    def _get_headers(self, token: str) -> dict[str, str]:
        """Get HTTP headers for EBRAINS API requests."""
        return {"Authorization": f"Bearer {token}"}

    def _fetch_file_list(self, token: str, cache_dir: Path | None = None) -> list[dict[str, Any]]:
        """Fetch file list from EBRAINS API, with optional caching."""
        cache_file = None
        if cache_dir:
            cache_file = cache_dir / "ebrains_file_list.json"
            if cache_file.exists():
                logger.info(f"Loading cached file list from {cache_file}")
                with open(cache_file) as f:
                    return json.load(f)

        logger.info("Fetching file list from EBRAINS API...")

        try:
            from ebrains_drive import BucketApiClient
        except ImportError:
            raise ImportError(
                "ebrains_drive package is required for EBRAINS downloads.\n"
                "Install it with: pip install ebrains_drive"
            )

        client = BucketApiClient(token=token)
        bucket = client.buckets.get_dataset(DATASET_ID, request_access=True)

        all_files = []
        for f in bucket.ls(prefix=FILE_PREFIX):
            file_info = {
                "name": f.name,
                "bytes": getattr(f, "bytes", 0) or getattr(f, "size", 0) or 0,
            }
            # Try to get hash from various attributes
            for attr in ("hash", "etag", "md5", "content_md5", "checksum"):
                val = getattr(f, attr, None)
                if val:
                    file_info["hash"] = val.strip('"')
                    break
            all_files.append(file_info)

        logger.info(f"Fetched {len(all_files)} files from EBRAINS")

        # Cache the result
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(all_files, f, indent=2)
            logger.info(f"Cached file list to {cache_file}")

        return all_files

    def _get_download_url(self, token: str, object_name: str) -> str | None:
        """Get a signed download URL for a file."""
        resp = requests.get(
            f"{BASE_URL}/datasets/{DATASET_ID}/{object_name}",
            headers=self._get_headers(token),
            allow_redirects=False,
            timeout=30
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            return resp.headers.get("Location")
        resp.raise_for_status()
        return None

    def _compute_md5(self, filepath: Path) -> str:
        """Compute MD5 hash of a file."""
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _download_file(
        self,
        token: str,
        file_info: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Download a single file if it doesn't exist, with hash verification."""
        object_name = file_info["name"]
        file_size = file_info.get("bytes", 0)
        expected_hash = file_info.get("hash")

        # Extract just the filename (flat structure)
        filename = Path(object_name).name
        output_path = output_dir / filename

        # Check if already exists
        if output_path.exists():
            existing_size = output_path.stat().st_size
            size_ok = (existing_size == file_size) or (file_size == 0 and existing_size > 0)

            if size_ok:
                # Verify hash if we have one
                if expected_hash:
                    actual_hash = self._compute_md5(output_path)
                    if actual_hash == expected_hash:
                        return {"status": "verified", "name": object_name, "path": output_path}
                    # Hash mismatch - will re-download below
                else:
                    # No hash to verify, trust size
                    return {"status": "skipped", "name": object_name, "path": output_path,
                            "reason": "exists (no hash to verify)"}

        # Create directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get download URL
        download_url = self._get_download_url(token, object_name)
        if not download_url:
            return {"status": "error", "name": object_name, "reason": "no download URL"}

        # Download with streaming and compute hash simultaneously
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            md5 = hashlib.md5()
            with requests.get(download_url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                response_etag = resp.headers.get("etag", "").strip('"')

                with open(temp_path, "wb") as f:
                    downloaded = 0
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            md5.update(chunk)
                            downloaded += len(chunk)

            computed_hash = md5.hexdigest()

            # Verify hash if available
            hash_to_check = expected_hash or response_etag
            if hash_to_check and computed_hash != hash_to_check:
                temp_path.unlink()
                return {"status": "error", "name": object_name,
                        "reason": f"hash mismatch: expected {hash_to_check}, got {computed_hash}"}

            # Atomic rename
            temp_path.rename(output_path)
            return {"status": "downloaded", "name": object_name, "path": output_path,
                    "size": total, "hash": computed_hash}

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return {"status": "error", "name": object_name, "reason": str(e)}

    def _get_all_tsv_files(self, tasks_dir: Path) -> list[Path]:
        """Find all k=all.tsv files for ebrains in the tasks directory."""
        return list(tasks_dir.glob("ebrains/*/k=all.tsv"))

    def _extract_slide_ids_from_tsv(self, tsv_path: Path) -> pd.DataFrame:
        """Extract case_id and slide_id from a TSV file."""
        df = pd.read_csv(tsv_path, sep="\t", dtype={"slide_id": str, "case_id": str})
        if "slide_id" in df.columns and "case_id" in df.columns:
            return df[["case_id", "slide_id"]].drop_duplicates()
        return pd.DataFrame(columns=["case_id", "slide_id"])

    def list_tasks(self, tasks_dir: Path, datasets: list[str] | None = None) -> list[dict[str, Any]]:
        """List all available EBRAINS tasks."""
        tasks = []
        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            if dataset_name != "ebrains":
                continue

            # Filter to requested datasets
            if datasets and "ebrains" not in datasets:
                continue

            task_name = tsv_path.parent.name
            df = self._extract_slide_ids_from_tsv(tsv_path)

            tasks.append({
                "dataset": dataset_name,
                "task": task_name,
                "n_slides": df["slide_id"].nunique(),
                "n_cases": df["case_id"].nunique() if "case_id" in df.columns else 0,
            })
        return tasks

    def get_slide_ids_for_tasks(
        self,
        tasks_dir: Path,
        datasets: list[str] | None = None
    ) -> dict[str, set[str]]:
        """Get slide IDs needed for EBRAINS Patho-Bench tasks."""
        result: dict[str, set[str]] = {}

        for tsv_path in self._get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name

            if dataset_name != "ebrains":
                continue

            task_name = tsv_path.parent.name
            slide_df = self._extract_slide_ids_from_tsv(tsv_path)

            key = f"ebrains/{task_name}"
            if key not in result:
                result[key] = set()
            result[key].update(str(sid) for sid in slide_df["slide_id"].unique())

        return result

    def _match_slides_to_files(
        self,
        slide_ids: set[str],
        all_files: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Match slide_ids to actual files in the bucket."""
        # slide_ids are like "a1965749-357f-11eb-801a-001a7dda7111"
        # files are like "v1.0/subdir/a1965749-357f-11eb-801a-001a7dda7111.ndpi"
        matched = []
        for f in all_files:
            filename = Path(f["name"]).name  # Get just the filename
            stem = Path(filename).stem  # Remove extension
            if stem in slide_ids and filename.endswith(SLIDE_EXTENSION):
                matched.append(f)
        return matched

    def download_slides(
        self,
        slide_ids: set[str],
        output_dir: Path,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        cache_dir: Path | None = None,
        max_workers: int = MAX_WORKERS,
        **kwargs
    ) -> None:
        """Download specific EBRAINS slides."""
        self._warn_access_required()

        token = self._get_token()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use cache_dir for file list if not specified
        effective_cache_dir = cache_dir or output_dir / ".cache"

        # Fetch file list
        all_files = self._fetch_file_list(token, effective_cache_dir)

        # Match slide_ids to files
        files_to_download = self._match_slides_to_files(slide_ids, all_files)
        logger.info(f"Found {len(files_to_download)} files matching {len(slide_ids)} requested slides")

        if not files_to_download:
            logger.warning("No matching files found to download")
            if create_symlinks and tasks_dir:
                self._create_symlinks(tasks_dir, output_dir)
            return

        # Download in parallel
        downloaded = 0
        skipped = 0
        verified = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_file, token, f, output_dir): f
                for f in files_to_download
            }

            for future in as_completed(futures):
                result = future.result()
                status = result["status"]

                if status == "downloaded":
                    downloaded += 1
                    logger.info(f"Downloaded: {Path(result['name']).name}")
                elif status == "verified":
                    verified += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "error":
                    errors += 1
                    logger.error(f"Error downloading {result['name']}: {result.get('reason', 'unknown')}")

        logger.info(f"Download complete. Downloaded: {downloaded}, Verified: {verified}, "
                   f"Skipped: {skipped}, Errors: {errors}")

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
        cache_dir: Path | None = None,
        max_workers: int = MAX_WORKERS,
        **kwargs
    ) -> None:
        """Download complete EBRAINS dataset (all .ndpi files)."""
        self._warn_access_required()

        token = self._get_token()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use cache_dir for file list if not specified
        effective_cache_dir = cache_dir or output_dir / ".cache"

        # Fetch file list
        all_files = self._fetch_file_list(token, effective_cache_dir)

        # Filter to only .ndpi files
        ndpi_files = [f for f in all_files if f["name"].endswith(SLIDE_EXTENSION)]
        logger.info(f"Found {len(ndpi_files)} .ndpi files to download")

        if not ndpi_files:
            logger.warning("No .ndpi files found in dataset")
            return

        # Download in parallel
        downloaded = 0
        skipped = 0
        verified = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_file, token, f, output_dir): f
                for f in ndpi_files
            }

            for future in as_completed(futures):
                result = future.result()
                status = result["status"]

                if status == "downloaded":
                    downloaded += 1
                    logger.info(f"Downloaded: {Path(result['name']).name}")
                elif status == "verified":
                    verified += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "error":
                    errors += 1
                    logger.error(f"Error downloading {result['name']}: {result.get('reason', 'unknown')}")

        logger.info(f"Download complete. Downloaded: {downloaded}, Verified: {verified}, "
                   f"Skipped: {skipped}, Errors: {errors}")

        # Create symlinks if requested
        if create_symlinks and tasks_dir:
            self._create_symlinks(tasks_dir, output_dir)

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
