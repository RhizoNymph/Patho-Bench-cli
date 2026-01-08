"""Dataset providers for Patho-Bench-dl."""

from patho_bench_cli.providers.base import DatasetProvider
from patho_bench_cli.providers.registry import get_provider, list_providers

__all__ = ["DatasetProvider", "get_provider", "list_providers"]
