"""Abstract base class for dataset providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DatasetProvider(ABC):
    """
    Abstract base class for dataset providers.
    
    Each provider implements downloading logic for a specific data source
    (e.g., TCIA for CPTAC, Kaggle for PANDA).
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g., 'cptac', 'panda')."""
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this provider."""
        ...
    
    @property
    @abstractmethod
    def datasets(self) -> list[str]:
        """List of dataset names available from this provider."""
        ...
    
    @abstractmethod
    def list_tasks(self, tasks_dir: Path) -> list[dict[str, Any]]:
        """
        List all tasks available for this provider.
        
        Args:
            tasks_dir: Path to the tasks directory containing TSV files.
            
        Returns:
            List of dicts with task info (name, n_slides, n_cases, etc.)
        """
        ...
    
    @abstractmethod
    def get_slide_ids_for_tasks(
        self, 
        tasks_dir: Path, 
        datasets: list[str] | None = None
    ) -> dict[str, set[str]]:
        """
        Get slide IDs needed for Patho-Bench tasks.
        
        Args:
            tasks_dir: Path to the tasks directory.
            datasets: Optional list of specific datasets to filter to.
            
        Returns:
            Dict mapping dataset name to set of slide_ids.
        """
        ...
    
    @abstractmethod
    def download_slides(
        self,
        slide_ids: set[str],
        output_dir: Path,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        **kwargs
    ) -> None:
        """
        Download specific slides.
        
        Args:
            slide_ids: Set of slide IDs to download.
            output_dir: Directory to save slides to.
            create_symlinks: If True, create per-task symlink directories.
            tasks_dir: Path to tasks directory (needed for symlinks).
            **kwargs: Provider-specific options.
        """
        ...
    
    @abstractmethod
    def download_full(
        self,
        output_dir: Path,
        datasets: list[str] | None = None,
        *,
        create_symlinks: bool = False,
        tasks_dir: Path | None = None,
        **kwargs
    ) -> None:
        """
        Download complete dataset(s), not just Patho-Bench slides.
        
        Args:
            output_dir: Directory to save slides to.
            datasets: Optional list of specific datasets to download.
            create_symlinks: If True, create per-task symlink directories.
            tasks_dir: Path to tasks directory (needed for symlinks).
            **kwargs: Provider-specific options.
        """
        ...
    
    def generate_manifest(
        self,
        slide_ids: set[str],
        output_dir: Path,
        tasks_info: dict[str, set[str]] | None = None
    ) -> Path:
        """
        Generate a manifest CSV of slides to download.
        
        Args:
            slide_ids: Set of slide IDs.
            output_dir: Directory to save manifest to.
            tasks_info: Optional dict mapping task names to slide IDs.
            
        Returns:
            Path to the generated manifest file.
        """
        import pandas as pd
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_data = []
        for slide_id in sorted(slide_ids):
            row = {"slide_id": slide_id}
            if tasks_info:
                used_by = [
                    task for task, ids in tasks_info.items() 
                    if slide_id in ids
                ]
                row["tasks"] = ",".join(used_by)
            manifest_data.append(row)
        
        manifest_df = pd.DataFrame(manifest_data)
        manifest_path = output_dir / f"{self.name}_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        
        return manifest_path

    def get_storage_directories(self, output_dir: Path, datasets: list[str] | None = None) -> list[Path]:
        """
        Get the directories where slides for the given datasets are stored.
        Standard implementation returns the root output_dir.
        
        Args:
            output_dir: Root slides directory.
            datasets: Optional list of dataset names.
            
        Returns:
            List of Paths to subdirectories.
        """
        return [output_dir]
