"""
Script to download only the CPTAC slides needed for Patho-Bench benchmarking.

This script:
1. Parses all TSV split files to extract unique slide_ids
2. Queries TCIA's PathDB to get image metadata
3. Creates a manifest of slides to download
4. Downloads only the slides that appear in the benchmark splits

Usage:
    # Download all CPTAC datasets
    python get_slides.py
    
    # Download specific dataset(s)
    python get_slides.py --datasets cptac_ccrcc cptac_brca
    
    # List available datasets
    python get_slides.py --list
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from tcia_utils import pathdb
import logging
import json

logging.basicConfig()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CPTAC slides needed for Patho-Bench benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                              # Process all CPTAC datasets
    %(prog)s --datasets cptac_ccrcc       # Process only cptac_ccrcc
    %(prog)s -d cptac_ccrcc cptac_brca    # Process multiple datasets
    %(prog)s --list                       # List available datasets
        """
    )
    parser.add_argument(
        "-d", "--datasets",
        nargs="+",
        help="Specific dataset(s) to process. If not specified, all CPTAC datasets are processed."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download the images (default: just create manifest)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent / "slides",
        help="Output directory for slides (default: ./slides)"
    )
    parser.add_argument(
        "--create-symlinks",
        action="store_true",
        help="Create per-dataset symlink directories pointing to downloaded slides"
    )
    return parser.parse_args()

# Mapping from Patho-Bench dataset names to TCIA collection names
CPTAC_COLLECTION_MAP = {
    "cptac_ccrcc": "CPTAC-CCRCC",
    "cptac_ccrcc_dhmc": "CPTAC-CCRCC",  # Same collection, different task subset
    "cptac_brca": "CPTAC-BRCA",
    "cptac_coad": "CPTAC-COAD",
    "cptac_gbm": "CPTAC-GBM",
    "cptac_hnsc": "CPTAC-HNSCC",
    "cptac_lscc": "CPTAC-LSCC",
    "cptac_luad": "CPTAC-LUAD",
    "cptac_lung": None,  # Combined dataset - need both LUAD and LSCC
    "cptac_ov": "CPTAC-OV",
    "cptac_pda": "CPTAC-PDA",
    "cptac_ucec": "CPTAC-UCEC",
    "cptac_all": None,  # Meta-dataset
}


def get_all_tsv_files(tasks_dir: Path) -> list[Path]:
    """Find all k=all.tsv files in the tasks directory."""
    return list(tasks_dir.glob("**/k=all.tsv"))


def extract_slide_ids_from_tsv(tsv_path: Path) -> pd.DataFrame:
    """Extract case_id and slide_id from a TSV file."""
    df = pd.read_csv(tsv_path, sep="\t")
    if "slide_id" in df.columns and "case_id" in df.columns:
        return df[["case_id", "slide_id"]].drop_duplicates()
    return pd.DataFrame(columns=["case_id", "slide_id"])


def get_cptac_slide_ids(tasks_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Get all unique slide_ids needed for each CPTAC dataset.
    
    Returns:
        dict mapping dataset name to DataFrame with case_id and slide_id columns
    """
    datasets = {}
    
    for tsv_path in get_all_tsv_files(tasks_dir):
        # Extract dataset name from path
        dataset_name = tsv_path.parent.parent.name
        
        # Only process CPTAC datasets
        if not dataset_name.startswith("cptac"):
            continue
            
        slide_df = extract_slide_ids_from_tsv(tsv_path)
        
        if dataset_name not in datasets:
            datasets[dataset_name] = slide_df
        else:
            datasets[dataset_name] = pd.concat(
                [datasets[dataset_name], slide_df], ignore_index=True
            ).drop_duplicates()
    
    return datasets


def get_all_unique_case_ids(datasets: dict[str, pd.DataFrame]) -> set[str]:
    """Get all unique case_ids across all datasets."""
    all_case_ids = set()
    for df in datasets.values():
        all_case_ids.update(df["case_id"].unique())
    return all_case_ids


def get_all_unique_slide_ids(datasets: dict[str, pd.DataFrame]) -> set[str]:
    """Get all unique slide_ids across all datasets."""
    all_slide_ids = set()
    for df in datasets.values():
        all_slide_ids.update(df["slide_id"].unique())
    return all_slide_ids


def create_slide_id_mapping(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a comprehensive mapping of all needed slides.
    
    The slide_id format in Patho-Bench appears to be: {case_id}-{suffix}
    e.g., C3N-00646-21 where C3N-00646 is the case_id and 21 is the slide number
    """
    all_rows = []
    for dataset_name, df in datasets.items():
        df_copy = df.copy()
        df_copy["dataset"] = dataset_name
        all_rows.append(df_copy)
    
    combined = pd.concat(all_rows, ignore_index=True)
    
    # Group by slide_id to see which datasets use each slide
    slide_datasets = combined.groupby("slide_id").agg({
        "case_id": "first",
        "dataset": lambda x: list(x.unique())
    }).reset_index()
    
    return slide_datasets


def query_tcia_images_for_collection(collection_name: str, cache_dir: Path) -> pd.DataFrame:
    """Query all images for a specific TCIA collection, with caching."""
    cache_file = cache_dir / f"{collection_name.replace('-', '_')}_images.csv"
    
    if cache_file.exists():
        logger.info(f"Loading cached images for {collection_name}")
        return pd.read_csv(cache_file)
    
    logger.info(f"Querying images for collection: {collection_name}")
    try:
        images = pathdb.getImages(collection_name, format="df")
        if images is not None and not images.empty:
            images.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(images)} images to {cache_file}")
        return images
    except Exception as e:
        logger.error(f"Failed to query {collection_name}: {e}")
        return pd.DataFrame()


def match_slides_to_tcia(
    needed_slides: pd.DataFrame,
    tcia_images: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match Patho-Bench slide_ids to TCIA images.
    
    The matching logic:
    - Patho-Bench slide_id: C3N-00646-21 
    - TCIA subjectId: C3N-00646
    - TCIA imageId: might contain the slide number
    """
    if tcia_images.empty:
        return pd.DataFrame()
    
    # Get unique case_ids from needed slides
    needed_case_ids = set(needed_slides["case_id"].unique())
    
    # Filter TCIA images to only those cases we need
    matched = tcia_images[tcia_images["subjectId"].isin(needed_case_ids)].copy()
    
    return matched


def generate_summary(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path
):
    """Generate a summary of what needs to be downloaded."""
    summary = []
    
    for dataset_name, df in sorted(datasets.items()):
        n_cases = df["case_id"].nunique()
        n_slides = len(df)
        tcia_collection = CPTAC_COLLECTION_MAP.get(dataset_name, "Unknown")
        
        summary.append({
            "dataset": dataset_name,
            "tcia_collection": tcia_collection,
            "n_cases": n_cases,
            "n_slides": n_slides,
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "dataset_summary.csv", index=False)
    
    # Also create per-dataset slide lists
    for dataset_name, df in datasets.items():
        df.to_csv(output_dir / f"{dataset_name}_slides.csv", index=False)
    
    return summary_df


def list_available_datasets(tasks_dir: Path):
    """List all available datasets in the tasks directory."""
    datasets = set()
    for tsv_path in get_all_tsv_files(tasks_dir):
        dataset_name = tsv_path.parent.parent.name
        if dataset_name.startswith("cptac"):
            datasets.add(dataset_name)
    return sorted(datasets)


def main():
    args = parse_args()
    
    tasks_dir = Path(__file__).parent / "tasks"
    output_dir = args.output_dir
    cache_dir = Path(__file__).parent / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    
    # Handle --list flag
    if args.list:
        available = list_available_datasets(tasks_dir)
        print("Available CPTAC datasets:")
        for ds in available:
            tcia = CPTAC_COLLECTION_MAP.get(ds, "(no TCIA mapping)")
            print(f"  {ds} -> {tcia}")
        return
    
    # Step 1: Extract all needed slide_ids from TSV files
    logger.info("=" * 60)
    logger.info("Step 1: Extracting slide_ids from Patho-Bench split files...")
    logger.info("=" * 60)
    
    all_datasets = get_cptac_slide_ids(tasks_dir)
    
    # Filter to requested datasets if specified
    if args.datasets:
        # Validate requested datasets
        invalid = set(args.datasets) - set(all_datasets.keys())
        if invalid:
            logger.error(f"Unknown dataset(s): {invalid}")
            logger.info(f"Available datasets: {sorted(all_datasets.keys())}")
            return
        
        datasets = {k: v for k, v in all_datasets.items() if k in args.datasets}
        logger.info(f"Filtering to requested datasets: {args.datasets}")
    else:
        datasets = all_datasets
        logger.info("Processing all CPTAC datasets")
    
    for dataset_name, df in sorted(datasets.items()):
        n_cases = df["case_id"].nunique()
        n_slides = df["slide_id"].nunique()
        logger.info(f"  {dataset_name}: {n_cases} cases, {n_slides} slides")
    
    all_case_ids = get_all_unique_case_ids(datasets)
    all_slide_ids = get_all_unique_slide_ids(datasets)
    logger.info(f"\nTotal unique cases needed: {len(all_case_ids)}")
    logger.info(f"Total unique slides needed: {len(all_slide_ids)}")
    
    # Generate and save summary
    summary_df = generate_summary(datasets, output_dir)
    logger.info(f"\nSummary:\n{summary_df.to_string(index=False)}")
    
    # Create slide mapping
    slide_mapping = create_slide_id_mapping(datasets)
    slide_mapping.to_csv(output_dir / "all_slides_needed.csv", index=False)
    logger.info(f"\nSaved slide mapping to {output_dir / 'all_slides_needed.csv'}")
    
    # Step 2: Query TCIA for each collection
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Querying TCIA PathDB for available images...")
    logger.info("=" * 60)
    
    # Get unique collections we need to query
    collections_needed = set()
    for dataset_name in datasets.keys():
        collection = CPTAC_COLLECTION_MAP.get(dataset_name)
        if collection:
            collections_needed.add(collection)
    
    logger.info(f"Collections to query: {collections_needed}")
    
    all_tcia_images = []
    for collection in sorted(collections_needed):
        images = query_tcia_images_for_collection(collection, cache_dir)
        if not images.empty:
            images["tcia_collection"] = collection
            all_tcia_images.append(images)
            logger.info(f"  {collection}: {len(images)} total images")
    
    if not all_tcia_images:
        logger.error("No images found from TCIA. Exiting.")
        return
    
    all_tcia_df = pd.concat(all_tcia_images, ignore_index=True)
    logger.info(f"\nTotal images in TCIA: {len(all_tcia_df)}")
    
    # Step 3: Match TCIA images to needed slides
    logger.info("\n" + "=" * 60)  
    logger.info("Step 3: Matching TCIA images to needed slides...")
    logger.info("=" * 60)
    
    matched_images = match_slides_to_tcia(slide_mapping, all_tcia_df)
    logger.info(f"Matched images: {len(matched_images)}")
    
    # Save the matched images manifest
    manifest_path = output_dir / "download_manifest.csv"
    matched_images.to_csv(manifest_path, index=False)
    logger.info(f"Saved download manifest to {manifest_path}")
    
    # Show some stats about coverage
    matched_cases = set()
    if not matched_images.empty:
        matched_cases = set(matched_images["subjectId"].unique())
    
    missing_cases = all_case_ids - matched_cases
    
    logger.info(f"\nCoverage:")
    logger.info(f"  Cases found in TCIA: {len(matched_cases)}")
    logger.info(f"  Cases missing from TCIA: {len(missing_cases)}")
    
    # Show missing cases per dataset
    if missing_cases:
        logger.info(f"\n  Missing cases by dataset:")
        for dataset_name, df in sorted(datasets.items()):
            dataset_cases = set(df["case_id"].unique())
            dataset_missing = dataset_cases & missing_cases
            if dataset_missing:
                logger.warning(f"    {dataset_name}: {len(dataset_missing)} missing cases")
                # Show up to 5 examples
                examples = list(dataset_missing)[:5]
                logger.warning(f"      Examples: {examples}")
    
    # Step 4: Download (organized by collection)
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Download")
    logger.info("=" * 60)
    
    if args.download:
        # Download into per-collection subdirectories
        for collection in sorted(collections_needed):
            collection_images = matched_images[matched_images["tcia_collection"] == collection]
            if collection_images.empty:
                continue
            
            collection_dir = output_dir / collection
            collection_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {len(collection_images)} images to {collection_dir}...")
            pathdb.downloadImages(collection_images, path=str(collection_dir))
        
        logger.info("Download complete!")
    else:
        logger.info("Download skipped (use --download flag to download)")
        logger.info(f"Manifest saved to: {manifest_path}")
        logger.info(f"Total images in manifest: {len(matched_images)}")
    
    # Step 5: Create per-task symlinks (optional)
    if args.create_symlinks:
        logger.info("\n" + "=" * 60)
        logger.info("Step 5: Creating per-task symlinks")
        logger.info("=" * 60)
        
        # Read the TSV files to get slide_ids per task
        for tsv_path in get_all_tsv_files(tasks_dir):
            dataset_name = tsv_path.parent.parent.name
            task_name = tsv_path.parent.name
            
            # Skip if this dataset wasn't requested
            if dataset_name not in datasets:
                continue
            
            collection = CPTAC_COLLECTION_MAP.get(dataset_name)
            if not collection:
                continue
            
            collection_dir = output_dir / collection
            if not collection_dir.exists():
                logger.warning(f"Collection dir {collection_dir} doesn't exist, skipping {dataset_name}/{task_name}")
                continue
            
            # Create task directory: by_task/dataset_name/task_name/
            task_dir = output_dir / "by_task" / dataset_name / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Read the TSV to get slide_ids for this specific task
            df = pd.read_csv(tsv_path, sep="\t")
            if "slide_id" not in df.columns:
                continue
            
            task_slide_ids = set(df["slide_id"].unique())
            
            # Map slide_ids to actual filenames in the collection
            # The slide_id format is like "C3L-00004-21" and files might have different names
            symlink_count = 0
            for img_file in collection_dir.glob("*"):
                if img_file.is_file():
                    # Create symlink in task directory
                    symlink_path = task_dir / img_file.name
                    if not symlink_path.exists():
                        symlink_path.symlink_to(img_file.resolve())
                        symlink_count += 1
            
            if symlink_count > 0:
                logger.info(f"  {dataset_name}/{task_name}: {symlink_count} symlinks")


if __name__ == "__main__":
    main()
