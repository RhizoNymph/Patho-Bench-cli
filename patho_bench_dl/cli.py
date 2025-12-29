"""
Patho-Bench-dl: Unified CLI for downloading Patho-Bench datasets.

Usage:
    patho-bench-dl list [PROVIDER]        List available providers/datasets
    patho-bench-dl download PROVIDER      Download slides from a provider
    patho-bench-dl tasks                  Download Patho-Bench task definitions
"""

import argparse
import sys
from pathlib import Path

import datasets

from patho_bench_dl.providers import get_provider, list_providers


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
    
    if args.full:
        # Download full dataset
        print("\nDownloading full dataset(s)...")
        provider.download_full(
            output_dir=output_dir,
            datasets=args.datasets,
        )
    else:
        # Get slide IDs from task files
        slide_ids_by_dataset = provider.get_slide_ids_for_tasks(
            tasks_dir=tasks_dir,
            datasets=args.datasets,
        )
        
        if not slide_ids_by_dataset:
            print("No slides found for the specified datasets.")
            return 1
        
        # Combine all slide IDs
        all_slide_ids: set[str] = set()
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
            provider.download_slides(
                slide_ids=all_slide_ids,
                output_dir=output_dir,
                create_symlinks=args.create_symlinks,
                tasks_dir=tasks_dir,
                datasets=args.datasets,
            )
            print("Download complete!")
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="patho-bench-dl",
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
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
