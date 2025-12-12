"""
Script to download only the PANDA slides needed for Patho-Bench benchmarking.

This script:
1. Parses all TSV split files to extract unique slide_ids for PANDA dataset
2. Downloads the required slides from Kaggle competition: prostate-cancer-grade-assessment
3. Extracts only the slides needed for the benchmark

Usage:
    # List available PANDA tasks
    python get_panda_slides.py --list
    
    # Download slides (creates manifest first)
    python get_panda_slides.py --download
    
    # Just create manifest without downloading
    python get_panda_slides.py
    
    # Download with symlinks organized by task
    python get_panda_slides.py --download --create-symlinks

Requirements:
    - Kaggle API configured (~/.kaggle/kaggle.json)
    - Install kaggle: pip install kaggle
"""

import argparse
import os
import subprocess
import pandas as pd
from pathlib import Path
import logging
import zipfile
import shutil
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

COMPETITION_NAME = "prostate-cancer-grade-assessment"
# PANDA slides are TIFF files with the slide_id as basename
SLIDE_EXTENSION = ".tiff"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download PANDA slides needed for Patho-Bench benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                              # Create manifest only
    %(prog)s --download                   # Download the slides
    %(prog)s --list                       # List available tasks
    %(prog)s --download --create-symlinks # Download and create task symlinks
        """
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available PANDA tasks and exit"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download the images (default: just create manifest)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent / "slides" / "panda",
        help="Output directory for slides (default: ./slides/panda)"
    )
    parser.add_argument(
        "--create-symlinks",
        action="store_true",
        help="Create per-task symlink directories pointing to downloaded slides"
    )
    parser.add_argument(
        "--kaggle-cache-dir",
        type=Path,
        default=Path(__file__).parent / "cache" / "kaggle",
        help="Directory to cache Kaggle downloads (default: ./cache/kaggle)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )
    return parser.parse_args()


def get_all_tsv_files(tasks_dir: Path) -> list[Path]:
    """Find all k=all.tsv files in the tasks directory."""
    return list(tasks_dir.glob("**/k=all.tsv"))


def extract_slide_ids_from_tsv(tsv_path: Path) -> pd.DataFrame:
    """Extract case_id and slide_id from a TSV file."""
    df = pd.read_csv(tsv_path, sep="\t")
    if "slide_id" in df.columns and "case_id" in df.columns:
        return df[["case_id", "slide_id"]].drop_duplicates()
    return pd.DataFrame(columns=["case_id", "slide_id"])


def get_panda_slide_ids(tasks_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Get all unique slide_ids needed for each PANDA task.
    
    Returns:
        dict mapping task name to DataFrame with case_id and slide_id columns
    """
    tasks = {}
    
    for tsv_path in get_all_tsv_files(tasks_dir):
        # Extract dataset name from path (e.g., tasks/panda/isup_grade/k=all.tsv)
        dataset_name = tsv_path.parent.parent.name
        
        # Only process PANDA datasets
        if dataset_name != "panda":
            continue
        
        task_name = tsv_path.parent.name
        slide_df = extract_slide_ids_from_tsv(tsv_path)
        
        if task_name not in tasks:
            tasks[task_name] = slide_df
        else:
            tasks[task_name] = pd.concat(
                [tasks[task_name], slide_df], ignore_index=True
            ).drop_duplicates()
    
    return tasks


def get_all_unique_slide_ids(tasks: dict[str, pd.DataFrame]) -> set[str]:
    """Get all unique slide_ids across all tasks."""
    all_slide_ids = set()
    for df in tasks.values():
        all_slide_ids.update(df["slide_id"].unique())
    return all_slide_ids


import sys

def get_kaggle_executable() -> str:
    """Find the kaggle executable."""
    # Check if kaggle is in PATH
    kaggle_path = shutil.which("kaggle")
    if kaggle_path:
        return kaggle_path
    
    # Check if kaggle is in the same directory as python executable (venv)
    python_dir = Path(sys.executable).parent
    kaggle_path = python_dir / "kaggle"
    if kaggle_path.exists():
        return str(kaggle_path)
        
    return "kaggle"  # Fallback to just command name


def download_competition_zip(output_dir: Path) -> Path:
    """
    Download the full competition zip file.
    Returns path to the downloaded zip file.
    """
    kaggle_cmd = get_kaggle_executable()
    zip_name = f"{COMPETITION_NAME}.zip"
    zip_path = output_dir / zip_name
    
    if zip_path.exists():
        logger.info(f"Zip file already exists at {zip_path}")
        return zip_path
        
    logger.info(f"Downloading competition zip to {output_dir}...")
    try:
        subprocess.run(
            [kaggle_cmd, "competitions", "download", 
             "-c", COMPETITION_NAME, 
             "-p", str(output_dir)],
            check=True
        )
        return zip_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download competition zip: {e}")
        raise


def extract_slides_from_zip(zip_path: Path, slide_ids: set[str], output_dir: Path):
    """
    Extract specific slides from the competition zip.
    """
    logger.info(f"Opening zip file: {zip_path}")
    
    # Track stats
    extracted_count = 0
    skipped_count = 0
    missing_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get list of files in zip to check existence efficiently
        # The structure is usually just the filenames or train_images/filename
        # Let's check the first few names to guess structure if needed, 
        # but standard Kaggle is often just the files or a folder.
        # For PANDA, it is likely 'train_images/{slide_id}.tiff' based on previous code.
        
        all_files = set(zf.namelist())
        
        logger.info(f"Extracting {len(slide_ids)} slides...")
        
        for i, slide_id in enumerate(slide_ids):
            filename = f"{slide_id}{SLIDE_EXTENSION}"
            # Try possible paths
            possible_paths = [
                f"train_images/{filename}",
                filename
            ]
            
            found_path = None
            for p in possible_paths:
                if p in all_files:
                    found_path = p
                    break
            
            target_path = output_dir / filename
            if target_path.exists():
                skipped_count += 1
                continue
                
            if found_path:
                # Extract to temp location then move, or open and write
                # zipfile.extract extracts with full path, so we might get output_dir/train_images/file.tiff
                # We want output_dir/file.tiff
                
                source = zf.open(found_path)
                with open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                extracted_count += 1
            else:
                missing_count += 1
                logger.warning(f"Slide not found in zip: {slide_id}")
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(slide_ids)} slides")

    logger.info(f"Extraction complete.")
    logger.info(f"  Extracted: {extracted_count}")
    logger.info(f"  Skipped (already existed): {skipped_count}")
    logger.info(f"  Missing in zip: {missing_count}")


def generate_manifest(
    slide_ids: set[str],
    output_dir: Path,
    tasks: dict[str, pd.DataFrame]
) -> Path:
    """Generate a manifest of slides to download."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest with slide info
    manifest_data = []
    for slide_id in sorted(slide_ids):
        # Find which tasks use this slide
        used_by = []
        for task_name, df in tasks.items():
            if slide_id in df["slide_id"].values:
                used_by.append(task_name)
        
        manifest_data.append({
            "slide_id": slide_id,
            "filename": f"{slide_id}{SLIDE_EXTENSION}",
            "tasks": ",".join(used_by)
        })
    
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = output_dir / "panda_download_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    return manifest_path


def create_task_symlinks(
    tasks: dict[str, pd.DataFrame],
    slides_dir: Path,
    symlinks_base_dir: Path
):
    """Create per-task symlink directories."""
    for task_name, df in tasks.items():
        task_dir = symlinks_base_dir / "panda" / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        symlink_count = 0
        for slide_id in df["slide_id"].unique():
            source_file = slides_dir / f"{slide_id}{SLIDE_EXTENSION}"
            if source_file.exists():
                symlink_path = task_dir / source_file.name
                if not symlink_path.exists():
                    symlink_path.symlink_to(source_file.resolve())
                    symlink_count += 1
        
        if symlink_count > 0:
            logger.info(f"  panda/{task_name}: {symlink_count} symlinks")


def list_available_tasks(tasks_dir: Path):
    """List all available PANDA tasks in the tasks directory."""
    tasks = []
    for tsv_path in get_all_tsv_files(tasks_dir):
        dataset_name = tsv_path.parent.parent.name
        if dataset_name == "panda":
            task_name = tsv_path.parent.name
            df = extract_slide_ids_from_tsv(tsv_path)
            tasks.append({
                "task": task_name,
                "n_slides": len(df),
                "n_cases": df["case_id"].nunique() if "case_id" in df.columns else 0
            })
    return tasks


def main():
    args = parse_args()
    
    tasks_dir = Path(__file__).parent / "tasks"
    output_dir = args.output_dir
    
    # Handle --list flag
    if args.list:
        tasks = list_available_tasks(tasks_dir)
        if not tasks:
            print("No PANDA tasks found in tasks directory")
            return
        
        print("Available PANDA tasks:")
        print("-" * 50)
        for t in tasks:
            print(f"  {t['task']}: {t['n_slides']} slides, {t['n_cases']} cases")
        print("-" * 50)
        print(f"Data source: Kaggle competition '{COMPETITION_NAME}'")
        return
    
    # Step 1: Extract all needed slide_ids from TSV files
    logger.info("=" * 60)
    logger.info("Step 1: Extracting slide_ids from Patho-Bench split files...")
    logger.info("=" * 60)
    
    tasks = get_panda_slide_ids(tasks_dir)
    
    if not tasks:
        logger.error("No PANDA tasks found in tasks directory")
        return
    
    for task_name, df in sorted(tasks.items()):
        n_cases = df["case_id"].nunique()
        n_slides = df["slide_id"].nunique()
        logger.info(f"  {task_name}: {n_cases} cases, {n_slides} slides")
    
    all_slide_ids = get_all_unique_slide_ids(tasks)
    logger.info(f"\nTotal unique slides needed: {len(all_slide_ids)}")
    
    # Step 2: Generate manifest
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Generating download manifest...")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = generate_manifest(all_slide_ids, output_dir.parent, tasks)
    logger.info(f"Manifest saved to: {manifest_path}")
    
    # Step 3: Download from Kaggle (if requested)
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Download and Extract")
    logger.info("=" * 60)
    
    if args.download:
        try:
            # Download zip
            zip_path = download_competition_zip(output_dir)
            
            # Extract files
            extract_slides_from_zip(zip_path, all_slide_ids, output_dir)
            
            # Cleanup
            logger.info(f"Removing zip file: {zip_path}")
            zip_path.unlink()
            
            logger.info("Done!")
            
        except Exception as e:
            logger.error(f"An error occurred during download/extraction: {e}")
            # Don't delete zip on error in case user wants to resume/debug, 
            # unless we want to be strict. Let's keep it for now.
        
    else:
        logger.info("Download skipped (use --download flag to download)")
        logger.info(f"Manifest saved to: {manifest_path}")
        logger.info(f"Total slides in manifest: {len(all_slide_ids)}")
    
    # Step 4: Create per-task symlinks (optional)
    if args.create_symlinks:
        logger.info("\n" + "=" * 60)
        logger.info("Step 4: Creating per-task symlinks")
        logger.info("=" * 60)
        
        if not output_dir.exists() or not list(output_dir.glob(f"*{SLIDE_EXTENSION}")):
            logger.warning("No slides found in output directory. Run with --download first.")
        else:
            symlinks_dir = output_dir.parent / "by_task"
            create_task_symlinks(tasks, output_dir, symlinks_dir)
            logger.info(f"Symlinks created in: {symlinks_dir}")


if __name__ == "__main__":
    main()
