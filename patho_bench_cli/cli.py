"""
Patho-Bench-dl: Unified CLI for downloading Patho-Bench datasets.

Usage:
    patho-bench-cli list [PROVIDER]        List available providers/datasets
    patho-bench-cli download PROVIDER      Download slides from a provider
    patho-bench-cli tasks                  Download Patho-Bench task definitions
    patho-bench-cli verify TARGET_DIR     Verify WSI files in a directory
"""

import argparse
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import datasets
import openslide

from patho_bench_cli.providers import get_provider, list_providers


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr at the file descriptor level."""
    # Only try to suppress if we can get a fileno (e.g., skip in some test environments)
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, ValueError):
        yield
        return

    with open(os.devnull, 'w') as devnull:
        old_stderr_fd = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, stderr_fd)
            os.close(old_stderr_fd)


def get_mpp(slide):
    """Extract MPP from OpenSlide object properties."""
    mpp_keys = [
        openslide.PROPERTY_NAME_MPP_X,
        'openslide.mirax.MPP',
        'aperio.MPP',
        'hamamatsu.XResolution',
        'openslide.comment',
    ]
    
    for key in mpp_keys:
        if key in slide.properties:
            try:
                # Some properties might contain multiple values or non-float strings
                val = slide.properties[key]
                if key == 'hamamatsu.XResolution':
                    # Hamamatsu resolution is often in nm/pixel, needs conversion to um/pixel
                    # TRIDENT seems to just cast it, let's follow that but be careful
                    mpp_x = float(val) / 1000.0 if float(val) > 100 else float(val)
                else:
                    mpp_x = float(val)
                return round(mpp_x, 4)
            except (ValueError, TypeError):
                continue

    x_res = slide.properties.get('tiff.XResolution')
    unit = slide.properties.get('tiff.ResolutionUnit')
    if x_res and unit:
        try:
            if unit.lower() == 'centimeter':
                return round(10000 / float(x_res), 4)
            elif unit.lower() == 'inch':
                return round(25400 / float(x_res), 4)
        except (ValueError, TypeError):
            pass
    return None


def get_mag(slide):
    """Extract magnification from OpenSlide object properties."""
    mag = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if mag is not None:
        try:
            return int(float(mag))
        except (ValueError, TypeError):
            pass
    return None


def cmd_list(args):
    """Handle the 'list' subcommand."""
    if args.provider:
        # List tasks for a specific provider
        try:
            provider = get_provider(args.provider)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        tasks_dir = Path(args.tasks_dir)
        if not tasks_dir.exists():
            print(f"Tasks directory not found: {tasks_dir}", file=sys.stderr)
            print("Run 'patho-bench-dl tasks' to download task definitions first.")
            return 1
        
        tasks = provider.list_tasks(tasks_dir)
        if not tasks:
            print(f"No tasks found for provider '{provider.name}'")
            return 0
        
        print(f"\n{provider.name.upper()} - {provider.description}")
        print("-" * 60)
        
        # Group by dataset
        datasets_seen = set()
        for task in tasks:
            ds = task.get("dataset", "")
            if ds not in datasets_seen:
                datasets_seen.add(ds)
                print(f"\n  {ds}:")
            print(f"    {task['task']}: {task['n_slides']} slides, {task['n_cases']} cases")
        
        print()
    else:
        # List all providers
        providers = list_providers()
        print("\nAvailable providers:")
        print("-" * 40)
        for name, provider in providers.items():
            print(f"  {name}: {provider.description}")
            print(f"    Datasets: {', '.join(provider.datasets)}")
        print()
        print("Use 'patho-bench-dl list <provider>' for details.")
    
    return 0


def cmd_download(args):
    """Handle the 'download' subcommand."""
    try:
        provider = get_provider(args.provider)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output_dir)
    
    # Auto-download tasks if needed for patho-bench mode
    if not args.full and not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        print("Downloading Patho-Bench task definitions from HuggingFace...")
        datasets.load_dataset(
            'MahmoodLab/Patho-Bench',
            cache_dir=str(tasks_dir),
            dataset_to_download="*",
            trust_remote_code=True
        )
        print(f"Tasks downloaded to: {tasks_dir}\n")
    
    print(f"Provider: {provider.name}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'Full dataset' if args.full else 'Patho-Bench slides only'}")
    
    all_slide_ids = set()
    if not args.full:
        # Get slide IDs from task files
        slide_ids_by_dataset = provider.get_slide_ids_for_tasks(
            tasks_dir=tasks_dir,
            datasets=args.datasets,
        )
        
        if not slide_ids_by_dataset:
            print("No slides found for the specified datasets.")
            return 1
        
        # Combine all slide IDs
        for ids in slide_ids_by_dataset.values():
            all_slide_ids.update(ids)
        
        print(f"\nTotal unique slides needed: {len(all_slide_ids)}")
        
        # Generate manifest
        manifest_path = provider.generate_manifest(
            slide_ids=all_slide_ids,
            output_dir=output_dir,
            tasks_info=slide_ids_by_dataset,
        )
        print(f"Manifest saved: {manifest_path}")

    if args.download:
        print("\nDownloading slides...")
        
        # Shared download helper
        def do_download(ids_to_download=None):
            # Filter out arguments already passed explicitly to avoid TypeErrors
            kwargs = {k: v for k, v in vars(args).items() if k not in [
                'slide_ids', 'output_dir', 'create_symlinks', 'tasks_dir', 'datasets'
            ]}
            if args.full and ids_to_download is None:
                provider.download_full(
                    output_dir=output_dir,
                    datasets=args.datasets,
                    create_symlinks=args.create_symlinks,
                    tasks_dir=tasks_dir,
                    **kwargs
                )
            else:
                # If ids_to_download is None here, it means we're in Patho-Bench mode initial download
                target_ids = ids_to_download if ids_to_download is not None else all_slide_ids
                provider.download_slides(
                    slide_ids=target_ids,
                    output_dir=output_dir,
                    create_symlinks=args.create_symlinks,
                    tasks_dir=tasks_dir,
                    datasets=args.datasets,
                    **kwargs
                )

        # Initial download
        if args.full:
            do_download()
        else:
            do_download(all_slide_ids)
        
        if args.verify:
            # Common WSI extensions
            extensions = {'.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn', '.bif', '.vms', '.vmu'}
            
            attempt = 0
            max_attempts = getattr(args, 'max_retries', 3)
            while attempt < max_attempts:
                print(f"\nVerifying downloaded slides (Attempt {attempt + 1}/{max_attempts})...")
                
                # Find downloaded files (excluding symlinks) in target directories
                storage_dirs = provider.get_storage_directories(output_dir, args.datasets)
                wsi_paths = []
                for s_dir in storage_dirs:
                    if not s_dir.exists():
                        continue
                    for p in s_dir.rglob("*"):
                        if p.suffix.lower() in extensions and not p.is_symlink():
                            # In Patho-Bench mode, only verify slides we explicitly wanted
                            # (Avoids verifying entire shared directory)
                            if args.full or not all_slide_ids or p.stem in all_slide_ids:
                                wsi_paths.append(p)
                
                if not wsi_paths:
                    print("No WSI files found to verify.")
                    break
                    
                passed, failed = verify_slides_in_parallel(
                    wsi_paths, 
                    args.jobs, 
                    delete=True,  # Always delete failed so we can redownload
                    verbose=args.verbose
                )
                
                if not failed:
                    print("All slides verified successfully!")
                    break
                    
                print(f"{len(failed)} slides failed verification and were deleted.")
                attempt += 1
                
                if attempt < max_attempts:
                    # Identify slide IDs for failed files
                    failed_slide_ids = set()
                    failed_filenames = {p.name for p, _, _ in failed}
                    
                    # Heuristic 1: If we have all_slide_ids, use them
                    if all_slide_ids:
                        for sid in all_slide_ids:
                            for ext in extensions:
                                if f"{sid}{ext}" in failed_filenames:
                                    failed_slide_ids.add(sid)
                    
                    # Heuristic 2: If heuristic 1 missed some, or we don't have all_slide_ids (full mode)
                    # use the filename stem as slide_id
                    if len(failed_slide_ids) < len(failed):
                        found_names = {p.name for p, _, _ in failed}
                        for fname in found_names:
                            # Skip if already found via Heuristic 1
                            already_found = False
                            for sid in failed_slide_ids:
                                if fname.startswith(sid):
                                    already_found = True
                                    break
                            if not already_found:
                                # Just use the stem
                                failed_slide_ids.add(Path(fname).stem)

                    if failed_slide_ids:
                        print(f"Retrying download for {len(failed_slide_ids)} slides...")
                        do_download(failed_slide_ids)
                    else:
                        print("Nothing to retry (could not map files).")
                        break
                else:
                    print("Reached maximum retry attempts.")

        print("\nDownload process complete!")
    else:
        print("\nDry run complete. Use --download to actually download slides.")
    
    return 0


def cmd_tasks(args):
    """Handle the 'tasks' subcommand - download Patho-Bench task definitions."""
    print("Downloading Patho-Bench task definitions from HuggingFace...")
    
    dataset_filter = args.dataset if args.dataset else "*"
    
    datasets.load_dataset(
        'MahmoodLab/Patho-Bench',
        cache_dir=args.tasks_dir,
        dataset_to_download=dataset_filter,
        trust_remote_code=True
    )
    
    print(f"Tasks downloaded to: {args.tasks_dir}")
    return 0


def verify_slides_in_parallel(wsi_paths, jobs, delete=False, verbose=False):
    """
    Verify a list of WSI files in parallel.
    
    Returns:
        tuple: (passed_paths, failed_info)
        failed_info is a list of (path, error, deleted_boolean)
    """
    def verify_single(path):
        is_valid = False
        error = None
        deleted = False
        try:
            # Try to open the slide
            slide = openslide.OpenSlide(str(path))
            
            # 1. Accessing dimensions often triggers format validation
            _ = slide.dimensions
            
            # 2. Check for MPP (required for TRIDENT and most analysis)
            mpp = get_mpp(slide)
            if mpp is None:
                raise ValueError("Missing MPP (microns per pixel) metadata")
                
            # 3. Check for Magnification
            mag = get_mag(slide)
            if mag is None:
                # Optional: Some workflows might proceed without mag, 
                # but TRIDENT's OpenSlideWSI raises for it.
                raise ValueError("Missing magnification metadata")

            slide.close()
            is_valid = True
        except Exception as e:
            error = str(e)
            if delete:
                try:
                    path.unlink()
                    deleted = True
                except Exception as de:
                    error = f"{error} (Deletion failed: {de})"

        return path, is_valid, error, deleted

    try:
        from tqdm import tqdm
        # Use stdout for tqdm so it doesn't get suppressed if we silence stderr
        pbar = tqdm(total=len(wsi_paths), desc="Verifying", file=sys.stdout)
    except ImportError:
        pbar = None

    passed = []
    failed = []

    def run_verification():
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(verify_single, p) for p in wsi_paths]
            for future in futures:
                path, is_valid, error, deleted = future.result()
                if is_valid:
                    passed.append(path)
                else:
                    failed.append((path, error, deleted))
                if pbar:
                    pbar.update(1)

    if verbose:
        run_verification()
    else:
        with suppress_stderr():
            run_verification()

    if pbar:
        pbar.close()
        
    return passed, failed


def cmd_verify(args):
    """Handle the 'verify' subcommand."""
    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        print(f"Error: Target directory not found: {target_dir}", file=sys.stderr)
        return 1

    # Common WSI extensions
    extensions = {'.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn', '.bif', '.vms', '.vmu'}
    
    print(f"Searching for WSI files in {target_dir}...")
    wsi_paths = []
    # Find all files with matching extensions (case-insensitive-ish via list check)
    for p in target_dir.rglob("*"):
        if p.suffix.lower() in extensions:
            if not p.is_symlink():
                wsi_paths.append(p)
    
    if not wsi_paths:
        print("No WSI files found.")
        return 0

    print(f"Found {len(wsi_paths)} files. Verifying using {args.jobs} jobs...")
    
    passed, failed = verify_slides_in_parallel(
        wsi_paths, 
        args.jobs, 
        delete=args.delete, 
        verbose=args.verbose
    )

    print(f"\nVerification Results:")
    print(f"  Total:  {len(wsi_paths)}")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed Slides:")
        for path, error, deleted in failed:
            status = " [DELETED]" if deleted else ""
            print(f"  {path}: {error}{status}")

    return 1 if failed else 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="patho-bench-cli",
        description="Unified downloader for Patho-Bench datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global options
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="./tasks",
        help="Path to Patho-Bench tasks directory (default: ./tasks)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- list subcommand ---
    list_parser = subparsers.add_parser(
        "list",
        help="List available providers and datasets"
    )
    list_parser.add_argument(
        "provider",
        nargs="?",
        help="Provider name to show details for"
    )
    
    # --- download subcommand ---
    download_parser = subparsers.add_parser(
        "download",
        help="Download slides from a provider"
    )
    download_parser.add_argument(
        "provider",
        help="Provider name (e.g., 'cptac', 'panda')"
    )
    download_parser.add_argument(
        "-d", "--datasets",
        nargs="+",
        help="Specific dataset(s) to download (default: all for provider)"
    )
    download_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./slides",
        help="Output directory for slides (default: ./slides)"
    )
    download_parser.add_argument(
        "--full",
        action="store_true",
        help="Download complete dataset(s), not just Patho-Bench slides"
    )
    download_parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download (default is dry-run with manifest only)"
    )
    download_parser.add_argument(
        "--create-symlinks",
        action="store_true",
        help="Create per-task symlink directories"
    )
    download_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify slides after download and retry failures"
    )
    download_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed slides (default: 3)"
    )
    download_parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help=f"Number of parallel jobs (default: {os.cpu_count() or 1})"
    )
    download_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show library warnings"
    )
    
    # --- tasks subcommand ---
    tasks_parser = subparsers.add_parser(
        "tasks",
        help="Download Patho-Bench task definitions from HuggingFace"
    )
    tasks_parser.add_argument(
        "--dataset",
        type=str,
        default="*",
        help="Specific dataset to download (default: all)"
    )
    
    # --- verify subcommand ---
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify WSI files in a directory"
    )
    verify_parser.add_argument(
        "target_dir",
        help="Directory to search for WSI files"
    )
    verify_parser.add_argument(
        "-d", "--delete",
        action="store_true",
        help="Delete slides that fail to open"
    )
    verify_parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help=f"Number of parallel jobs (default: {os.cpu_count() or 1})"
    )
    verify_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show library warnings"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Dispatch to command handler
    if args.command == "list":
        return cmd_list(args)
    elif args.command == "download":
        return cmd_download(args)
    elif args.command == "tasks":
        return cmd_tasks(args)
    elif args.command == "verify":
        return cmd_verify(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
