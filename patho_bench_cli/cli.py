"""
Patho-Bench-dl: Unified CLI for downloading Patho-Bench datasets.

Usage:
    patho-bench-cli list [PROVIDER]        List available providers/datasets
    patho-bench-cli download PROVIDER      Download slides from a provider
    patho-bench-cli tasks                  Download Patho-Bench task definitions
    patho-bench-cli embed PROVIDER         Generate embeddings for dataset tasks
    patho-bench-cli verify TARGET_DIR      Verify WSI files in a directory
    patho-bench-cli bench                  Run Patho-Bench benchmarking
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import datasets
import openslide
import pandas as pd
from dotenv import load_dotenv

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
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.slides_dir)

    if args.provider == 'all':
        if args.datasets:
            print("Error: Dataset filtering is not allowed when provider is 'all'", file=sys.stderr)
            return 1
        providers = sorted(list_providers().values(), key=lambda p: p.name)
    else:
        try:
            providers = [get_provider(args.provider)]
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
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
    
    for provider in providers:
        print(f"\n{'='*60}")
        print(f"Provider: {provider.name}")
        print(f"Output: {output_dir}")
        print(f"Mode: {'Full dataset' if args.full else 'Patho-Bench slides only'}")
        print(f"{'='*60}")
    
        all_slide_ids = set()
        if not args.full:
            # Get slide IDs from task files
            slide_ids_by_dataset = provider.get_slide_ids_for_tasks(
                tasks_dir=tasks_dir,
                datasets=args.datasets,
            )
            
            if not slide_ids_by_dataset:
                print(f"No slides found for the specified datasets in {provider.name}.")
                continue
            
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
                    'slide_ids', 'output_dir', 'create_symlinks', 'tasks_dir', 'datasets', 'provider'
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

            print(f"\nDownload process complete for {provider.name}!")
        else:
            print(f"\nDry run complete for {provider.name}. Use --download to actually download slides.")
    
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

            # 4. Try to get a thumbnail (checks global structure and base-level readability)
            try:
                _ = slide.get_thumbnail((512, 512))
            except Exception as e:
                raise ValueError(f"Failed to generate thumbnail (Data Corruption?): {e}")

            # 5. Trial read_region (checks if pixel data can be decoded)
            # We check multiple random points to catch sparse corruption
            # (especially common in NDPI or network-interrupted downloads)
            import random
            
            # Use a larger number of points to be more thorough.
            # 500 points at ~0.001s-0.01s each is ~0.5-5 seconds per slide.
            n_trial_reads = 1000
            
            # Distribution across levels: mostly level 0, but some higher levels
            levels_to_check = [0]
            if slide.level_count > 1:
                levels_to_check.extend([1, min(2, slide.level_count - 1)])
            
            # Always check a patch at the very end of the slide (typical for truncation)
            w, h = slide.dimensions
            tail_points = [
                (max(0, w - 256), max(0, h - 256)), # Bottom-right
                (max(0, w // 2), max(0, h - 256)),   # Bottom-center
            ]
            for x, y in tail_points:
                _ = slide.read_region((x, y), 0, (224, 224))

            # Random sampling
            for _ in range(n_trial_reads):
                lvl = random.choice(levels_to_check)
                lw, lh = slide.level_dimensions[lvl]
                
                # Pick random location in current level
                x = random.randint(0, max(0, lw - 256))
                y = random.randint(0, max(0, lh - 256))
                
                # Note: read_region location is always in level 0 coordinates
                ds = slide.level_downsamples[lvl]
                loc_level0 = (int(x * ds), int(y * ds))
                
                _ = slide.read_region(loc_level0, lvl, (224, 224))

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

    if failed:
        print("\nFailed Slides:")
        for path, error, deleted in failed:
            status = " [DELETED]" if deleted else ""
            print(f"  {path}: {error}{status}")

    print(f"\nVerification Results:")
    print(f"  Total:  {len(wsi_paths)}")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")

    return 1 if failed else 0


def update_embed_status_csv(
    status_csv_path: Path,
    dataset: str,
    task: str,
    status: str
) -> None:
    """
    Update the embed status CSV with the result for a dataset/task.

    If the dataset/task already exists in the CSV, it will be overwritten.
    If not, a new row will be added.

    Args:
        status_csv_path: Path to the status CSV file
        dataset: Dataset name
        task: Task name
        status: Status string (e.g., "success", "failed", "skipped")
    """
    # Read existing CSV if it exists
    if status_csv_path.exists():
        df = pd.read_csv(status_csv_path, dtype=str)
    else:
        df = pd.DataFrame(columns=["Dataset", "Task", "Status"])

    # Check if this dataset/task combination exists
    mask = (df["Dataset"] == dataset) & (df["Task"] == task)

    if mask.any():
        # Update existing row
        df.loc[mask, "Status"] = status
    else:
        # Add new row
        new_row = pd.DataFrame([{"Dataset": dataset, "Task": task, "Status": status}])
        df = pd.concat([df, new_row], ignore_index=True)

    # Sort by Dataset, then Task for consistent ordering
    df = df.sort_values(["Dataset", "Task"]).reset_index(drop=True)

    # Ensure parent directory exists
    status_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write back
    df.to_csv(status_csv_path, index=False)


def create_embedding_symlinks(
    embeddings_dir: Path,
    by_task_dir: Path,
    dataset: str,
    task: str,
    slide_ids: set[str]
) -> int:
    """
    Create symlinks from embeddings/by_task/DATASET/TASK to embeddings/DATASET
    for the necessary slide IDs.

    Args:
        embeddings_dir: Base embeddings directory (e.g., /embeddings/PATCH_ENCODER/DATASET)
        by_task_dir: Task-specific symlink directory (e.g., /embeddings/PATCH_ENCODER/by_task/DATASET/TASK)
        dataset: Dataset name
        task: Task name
        slide_ids: Set of slide IDs to create symlinks for

    Returns:
        Number of symlinks created
    """
    by_task_dir.mkdir(parents=True, exist_ok=True)

    # Common embedding file extensions from TRIDENT
    extensions = ['.h5', '.pt']

    symlinks_created = 0
    # Create a mapping of slide_id to actual file to handle nested TRIDENT structure
    # (e.g., dataset/20x_224px/features_uni_v1/slide_id.h5)
    found_files = {}
    
    # We use rglob to find all .h5 and .pt files
    for ext in extensions:
        for f in embeddings_dir.rglob(f"*{ext}"):
            # Check if this file name starts with one of our slide IDs
            # TRIDENT might save as slide_id.h5 or slide_id.svs.h5
            stem = f.name.split('.')[0]
            if stem in slide_ids:
                if stem not in found_files:
                    found_files[stem] = f
                else:
                    # If multiple files exist, prefer the one with the shortest path 
                    # (it might be closer to root) or just take the first one we found
                    pass

    for slide_id in slide_ids:
        if slide_id in found_files:
            src_file = found_files[slide_id]
            # Use original extension from source file
            ext = src_file.suffix
            dst_file = by_task_dir / f"{slide_id}{ext}"
            
            if not dst_file.exists():
                # We want the symlink to be relative if possible, but absolute is safer for now
                # given how the directories are structured
                dst_file.symlink_to(src_file.resolve())
                symlinks_created += 1

    return symlinks_created


def cmd_embed(args):
    """Handle the 'embed' subcommand - generate embeddings using TRIDENT."""
    tasks_dir = Path(args.tasks_dir)
    slides_dir = Path(args.slides_dir)
    embeddings_dir = Path(args.embeddings_dir)
    status_csv_path = Path(args.status_csv)

    if args.provider == 'all':
        if args.datasets:
            print("Error: Dataset filtering is not allowed when provider is 'all'", file=sys.stderr)
            return 1
        if args.tasks:
            print("Error: Task filtering is not allowed when provider is 'all'", file=sys.stderr)
            return 1
        providers = sorted(list_providers().values(), key=lambda p: p.name)
    else:
        try:
            providers = [get_provider(args.provider)]
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if not tasks_dir.exists():
        print(f"Error: Tasks directory not found: {tasks_dir}", file=sys.stderr)
        print("Run 'patho-bench-cli tasks' to download task definitions first.", file=sys.stderr)
        return 1

    if not slides_dir.exists():
        print(f"Error: Slides directory not found: {slides_dir}", file=sys.stderr)
        print("Make sure you've downloaded slides with 'patho-bench-cli download --create-symlinks'", file=sys.stderr)
        return 1

    # Find TRIDENT script
    trident_script = Path(__file__).parent.parent / "TRIDENT" / "run_batch_of_slides.py"
    if not trident_script.exists():
        print(f"Error: TRIDENT script not found at {trident_script}", file=sys.stderr)
        return 1

    total_success = 0
    total_failed = 0

    for provider in providers:
        print(f"\n{'='*60}")
        print(f"Processing provider: {provider.name}")
        print(f"{'='*60}")
        
        # Get all tasks for the provider, optionally filtered by datasets
        all_tasks = provider.list_tasks(tasks_dir, datasets=args.datasets)

        # Filter by specific tasks if specified
        if args.tasks:
            all_tasks = [t for t in all_tasks if t['task'] in args.tasks]

        if not all_tasks:
            print(f"No tasks found matching the specified criteria for {provider.name}.", file=sys.stderr)
            continue

        print(f"Patch encoder: {args.patch_encoder}")
        print(f"Magnification: {args.mag}x")
        print(f"Patch size: {args.patch_size}px")
        print(f"Tasks to process: {len(all_tasks)}")

        # Group tasks by dataset
        tasks_by_dataset = {}
        for task in all_tasks:
            dataset = task['dataset']
            if dataset not in tasks_by_dataset:
                tasks_by_dataset[dataset] = []
            tasks_by_dataset[dataset].append(task)

        # Process each dataset
        for dataset, tasks in tasks_by_dataset.items():
            print(f"\nProcessing dataset: {dataset}")

            for task_info in tasks:
                task = task_info['task']
                print(f"\nTask: {task} ({task_info['n_slides']} slides)")

                # Define paths
                task_slides_dir = slides_dir / "by_task" / dataset / task
                dataset_embeddings_dir = embeddings_dir / args.patch_encoder / dataset
                task_embeddings_symlinks_dir = embeddings_dir / args.patch_encoder / "by_task" / dataset / task

                if not task_slides_dir.exists():
                    print(f"  Warning: Slides directory not found: {task_slides_dir}", file=sys.stderr)
                    print(f"  Skipping task {task}", file=sys.stderr)
                    update_embed_status_csv(status_csv_path, dataset, task, "skipped: slides dir not found")
                    total_failed += 1
                    continue

                # Create embeddings directory
                dataset_embeddings_dir.mkdir(parents=True, exist_ok=True)

                # Build TRIDENT command using current Python interpreter
                cmd = [
                    sys.executable,  # Use the current Python interpreter
                    str(trident_script),
                    "--task", "all",
                    "--wsi_dir", str(task_slides_dir),
                    "--job_dir", str(dataset_embeddings_dir),
                    "--patch_encoder", args.patch_encoder,
                    "--mag", str(args.mag),
                    "--patch_size", str(args.patch_size),
                ]

                # Add optional arguments
                if args.gpu is not None:
                    cmd.extend(["--gpu", str(args.gpu)])
                if args.batch_size:
                    cmd.extend(["--batch_size", str(args.batch_size)])
                if args.skip_errors:
                    cmd.append("--skip_errors")
                if args.wsi_ext:
                    cmd.extend(["--wsi_ext"] + args.wsi_ext)

                # Check for provider-specific custom WSI list (e.g., visiomel per-slide magnification)
                custom_wsi_list = provider.get_custom_wsi_list(
                    task_slides_dir=task_slides_dir,
                    slides_dir=slides_dir,
                    dataset=dataset,
                    task=task,
                )
                if custom_wsi_list:
                    cmd.extend(["--custom_list_of_wsis", str(custom_wsi_list)])
                    if args.verbose:
                        print(f"  Using custom WSI list: {custom_wsi_list}")

                # If MPP is provided (and no custom WSI list), create a temporary CSV for TRIDENT
                # This is needed for standard image formats (JPG, PNG) that lack MPP metadata
                mpp_csv_path = None
                if args.mpp is not None and custom_wsi_list is None:
                    # List all slide files in the task slides directory
                    slide_extensions = {'.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn',
                                        '.bif', '.vms', '.vmu', '.jpg', '.jpeg', '.png'}
                    if args.wsi_ext:
                        slide_extensions = set(args.wsi_ext)

                    slide_files = []
                    for f in task_slides_dir.iterdir():
                        if f.is_file() or f.is_symlink():
                            if f.suffix.lower() in slide_extensions:
                                slide_files.append(f.name)

                    if slide_files:
                        # Create temp CSV with wsi and mpp columns
                        mpp_df = pd.DataFrame({
                            'wsi': slide_files,
                            'mpp': [args.mpp] * len(slide_files)
                        })
                        mpp_csv_path = task_slides_dir / "_mpp_list.csv"
                        mpp_df.to_csv(mpp_csv_path, index=False)
                        cmd.extend(["--custom_list_of_wsis", str(mpp_csv_path)])
                        if args.verbose:
                            print(f"  Created MPP CSV with {len(slide_files)} slides at mpp={args.mpp}")

                print(f"  Running TRIDENT...")
                if args.verbose:
                    print(f"  Command: {' '.join(cmd)}")

                try:
                    subprocess.run(
                        cmd,
                        cwd=trident_script.parent,
                        check=True,
                        capture_output=not args.verbose
                    )
                    print(f"  ✓ TRIDENT completed successfully")
                    update_embed_status_csv(status_csv_path, dataset, task, "success")
                    total_success += 1
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ TRIDENT failed with exit code {e.returncode}", file=sys.stderr)
                    if args.verbose and e.stdout:
                        print(f"  stdout: {e.stdout.decode()}", file=sys.stderr)
                    if args.verbose and e.stderr:
                        print(f"  stderr: {e.stderr.decode()}", file=sys.stderr)
                    update_embed_status_csv(status_csv_path, dataset, task, f"failed: exit code {e.returncode}")
                    total_failed += 1
                    continue
                finally:
                    # Clean up temp MPP CSV
                    if mpp_csv_path and mpp_csv_path.exists():
                        mpp_csv_path.unlink()

                # Create symlinks
                if args.create_symlinks:
                    print(f"  Creating symlinks...")
                    # Read slide IDs from the task file
                    task_file = tasks_dir / dataset / task / "k=all.tsv"
                    if task_file.exists():
                        task_df = pd.read_csv(task_file, sep='\t', dtype={'slide_id': str})
                        slide_ids = set(task_df['slide_id'].unique())

                        n_symlinks = create_embedding_symlinks(
                            embeddings_dir=dataset_embeddings_dir,
                            by_task_dir=task_embeddings_symlinks_dir,
                            dataset=dataset,
                            task=task,
                            slide_ids=slide_ids
                        )
                        print(f"  ✓ Created {n_symlinks} symlinks")
                    else:
                        print(f"  Warning: Task file not found: {task_file}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"Embedding generation complete!")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"  Status CSV: {status_csv_path}")
    print(f"{'='*60}")

    return 1 if total_failed > 0 else 0


def cmd_bench(args):
    """Handle the 'bench' subcommand - run Patho-Bench benchmarking."""
    # Find the Patho-Bench run script
    patho_bench_script = Path(__file__).parent.parent / "Patho-Bench" / "advanced_usage" / "run.py"
    if not patho_bench_script.exists():
        print(f"Error: Patho-Bench run script not found at {patho_bench_script}", file=sys.stderr)
        return 1

    # Determine venv path - use provided or default to root .venv
    if args.venv:
        venv_path = Path(args.venv)
    else:
        venv_path = Path(__file__).parent.parent / ".venv" / "bin" / "activate"

    if not venv_path.exists():
        print(f"Error: Virtual environment not found at {venv_path}", file=sys.stderr)
        print("Please provide --venv path or ensure .venv exists in the project root.", file=sys.stderr)
        return 1

    # Expand experiment types
    ALL_EXPERIMENT_TYPES = ["linprobe", "coxnet", "retrieval", "finetune"]

    if "all" in args.experiment_type:
        experiment_types = ALL_EXPERIMENT_TYPES
    else:
        experiment_types = args.experiment_type

    # Track results for each experiment
    results = {}
    use_subdirs = len(experiment_types) > 1

    print(f"Running Patho-Bench benchmarking...")
    print(f"  Experiment type(s): {', '.join(experiment_types)}")
    print(f"  Model(s): {', '.join(args.model_name)}")
    print(f"  Save to: {args.saveto}")
    if use_subdirs:
        print(f"  (Results will be saved to subdirectories per experiment type)")

    for exp_type in experiment_types:
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_type}")
        print(f"{'='*60}")

        # Determine saveto path - use subdir when running multiple experiments
        if use_subdirs:
            exp_saveto = str(Path(args.saveto) / exp_type)
        else:
            exp_saveto = args.saveto

        # Build command for this experiment type
        cmd = [
            sys.executable,
            str(patho_bench_script),
            "--experiment_type", exp_type,
            "--model_name", *args.model_name,
            "--saveto", exp_saveto,
            "--venv", str(venv_path),
        ]

        # Add hyperparams_yaml
        if args.hyperparams_yaml:
            cmd.extend(["--hyperparams_yaml", args.hyperparams_yaml])
        else:
            # Use default config based on experiment type
            default_config = patho_bench_script.parent / "configs" / exp_type / f"{exp_type}.yaml"
            if default_config.exists():
                cmd.extend(["--hyperparams_yaml", str(default_config)])
            else:
                print(f"Error: No hyperparams_yaml provided and default config not found at {default_config}", file=sys.stderr)
                results[exp_type] = False
                continue

        # Add tasks_yaml
        if args.tasks_yaml:
            cmd.extend(["--tasks_yaml", args.tasks_yaml])
        else:
            default_tasks = patho_bench_script.parent / "configs" / "tasks.yaml"
            if default_tasks.exists():
                cmd.extend(["--tasks_yaml", str(default_tasks)])

        # Add optional arguments
        if args.pooled_dirs_yaml:
            cmd.extend(["--pooled_dirs_yaml", args.pooled_dirs_yaml])
        if args.patch_dirs_yaml:
            cmd.extend(["--patch_dirs_yaml", args.patch_dirs_yaml])
        if args.splits_root:
            cmd.extend(["--splits_root", args.splits_root])
        if args.model_kwargs_yaml:
            cmd.extend(["--model_kwargs_yaml", args.model_kwargs_yaml])
        if args.combine_slides_per_patient is not None:
            cmd.extend(["--combine_slides_per_patient", str(args.combine_slides_per_patient)])
        if args.tmux_id:
            # Append experiment type to tmux_id to avoid conflicts
            tmux_id = f"{args.tmux_id}_{exp_type}" if use_subdirs else args.tmux_id
            cmd.extend(["--tmux_id", tmux_id])
        if args.preserve:
            cmd.append("--preserve")
        if args.delay_interval != 2:
            cmd.extend(["--delay_interval", str(args.delay_interval)])
        if args.global_delay != 0:
            cmd.extend(["--global_delay", str(args.global_delay)])
        if args.gpu != -1:
            cmd.extend(["--gpu", str(args.gpu)])

        if args.verbose:
            print(f"  Command: {' '.join(cmd)}")

        try:
            subprocess.run(
                cmd,
                cwd=patho_bench_script.parent,
                check=True,
                capture_output=not args.verbose
            )
            print(f"  {exp_type}: launched successfully!")
            results[exp_type] = True
        except subprocess.CalledProcessError as e:
            print(f"  {exp_type}: failed with exit code {e.returncode}", file=sys.stderr)
            if e.stdout:
                print(f"  stdout: {e.stdout.decode()}", file=sys.stderr)
            if e.stderr:
                print(f"  stderr: {e.stderr.decode()}", file=sys.stderr)
            results[exp_type] = False

    # Print summary
    print(f"\n{'='*60}")
    print("Benchmarking Summary")
    print(f"{'='*60}")
    for exp_type, success in results.items():
        status = "success" if success else "FAILED"
        print(f"  {exp_type}: {status}")

    failed_count = sum(1 for success in results.values() if not success)
    return 1 if failed_count > 0 else 0


def main():
    """Main CLI entry point."""
    # Load environment variables from .env file
    load_dotenv()

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
        "--slides-dir",
        type=str,
        default=os.environ.get("SLIDES_DIR", "./slides"),
        help="Output directory for slides (default: ./slides, or SLIDES_DIR env var)"
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

    # --- embed subcommand ---
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate embeddings for dataset tasks using TRIDENT"
    )
    embed_parser.add_argument(
        "provider",
        help="Provider name (e.g., 'cptac', 'panda')"
    )
    embed_parser.add_argument(
        "-d", "--datasets",
        nargs="+",
        help="Specific dataset(s) to embed (default: all for provider)"
    )
    embed_parser.add_argument(
        "-t", "--tasks",
        nargs="+",
        help="Specific tasks to embed (default: all tasks for selected datasets)"
    )
    embed_parser.add_argument(
        "--slides-dir",
        type=str,
        default=os.environ.get("SLIDES_DIR", "./slides"),
        help="Base slides directory containing by_task/ subdirectory (default: ./slides, or SLIDES_DIR env var)"
    )
    embed_parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=os.environ.get("EMBEDDINGS_DIR", "./embeddings"),
        help="Base directory for embeddings output (default: ./embeddings, or EMBEDDINGS_DIR env var)"
    )
    embed_parser.add_argument(
        "--patch-encoder",
        type=str,
        default=os.environ.get("PATCH_ENCODER", "conch_v15"),
        help="Patch encoder to use (default: conch_v15, or PATCH_ENCODER env var)"
    )
    embed_parser.add_argument(
        "--mag",
        type=int,
        choices=[5, 10, 20, 40, 80],
        default=20,
        help="Magnification for feature extraction (default: 20)"
    )
    embed_parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size for feature extraction (default: 224)"
    )
    embed_parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0)"
    )
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for feature extraction (default: TRIDENT's default)"
    )
    embed_parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip errored slides and continue processing"
    )
    embed_parser.add_argument(
        "--wsi-ext",
        nargs="+",
        help="WSI file extensions to include (e.g., --wsi-ext .czi .svs)"
    )
    embed_parser.add_argument(
        "--mpp",
        type=float,
        help="Microns per pixel (required for standard image formats like JPG/PNG that lack MPP metadata)"
    )
    embed_parser.add_argument(
        "--create-symlinks",
        action="store_true",
        help="Create per-task symlink directories for embeddings"
    )
    embed_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show TRIDENT output"
    )
    embed_parser.add_argument(
        "--status-csv",
        type=str,
        default="./embed_status.csv",
        help="Path to CSV file tracking embed status per dataset/task (default: ./embed_status.csv)"
    )

    # --- bench subcommand ---
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run Patho-Bench benchmarking using the Patho-Bench run script"
    )
    bench_parser.add_argument(
        "--experiment-type",
        type=str,
        nargs='+',
        choices=["linprobe", "coxnet", "retrieval", "finetune", "all"],
        required=True,
        help="Type(s) of experiment to run. Use 'all' to run all experiment types sequentially."
    )
    bench_parser.add_argument(
        "--model-name",
        type=str,
        nargs='+',
        required=True,
        help="Name(s) of the model(s) to use. Multiple models will run in parallel."
    )
    bench_parser.add_argument(
        "--saveto",
        type=str,
        required=True,
        help="Save results to this directory"
    )
    bench_parser.add_argument(
        "--hyperparams-yaml",
        type=str,
        help="Path to config YAML specifying hyperparameters (default: configs/{experiment_type}/{experiment_type}.yaml)"
    )
    bench_parser.add_argument(
        "--tasks-yaml",
        type=str,
        help="Path to the YAML file containing the task codes (default: configs/tasks.yaml)"
    )
    bench_parser.add_argument(
        "--pooled-dirs-yaml",
        type=str,
        help="Path to YAML file mapping data sources to pooled embeddings directories"
    )
    bench_parser.add_argument(
        "--patch-dirs-yaml",
        type=str,
        help="Path to YAML file mapping data sources to patch embeddings directories"
    )
    bench_parser.add_argument(
        "--splits-root",
        type=str,
        help="Root directory for downloading splits from HuggingFace"
    )
    bench_parser.add_argument(
        "--model-kwargs-yaml",
        type=str,
        help="Path to YAML file containing optional kwargs for initializing the model"
    )
    bench_parser.add_argument(
        "--combine-slides-per-patient",
        type=lambda x: x.lower() == 'true',
        help="Whether to combine patches from multiple slides when pooling at case_id level"
    )
    bench_parser.add_argument(
        "--venv",
        type=str,
        help="Path to the virtual environment activate script (default: .venv/bin/activate)"
    )
    bench_parser.add_argument(
        "--tmux-id",
        type=str,
        help="Optional tmux session name"
    )
    bench_parser.add_argument(
        "--preserve",
        action="store_true",
        help="Preserve the finished tmux panes"
    )
    bench_parser.add_argument(
        "--delay-interval",
        type=float,
        default=2,
        help="Factor by which to delay each pane (default: 2)"
    )
    bench_parser.add_argument(
        "--global-delay",
        type=int,
        default=0,
        help="Delay the whole study in minutes (default: 0)"
    )
    bench_parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU to use for pooling. If -1, the best available GPU is used (default: -1)"
    )
    bench_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show command and full output"
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
    elif args.command == "embed":
        return cmd_embed(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "bench":
        return cmd_bench(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
