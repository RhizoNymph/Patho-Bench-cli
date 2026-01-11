"""Provider registry for dataset sources."""

from patho_bench_cli.providers.base import DatasetProvider

# Registry of available providers
_PROVIDERS: dict[str, DatasetProvider] = {}


def register_provider(provider: DatasetProvider) -> None:
    """Register a provider in the global registry."""
    _PROVIDERS[provider.name] = provider


def get_provider(name: str) -> DatasetProvider:
    """
    Get a provider by name.
    
    Args:
        name: Provider name (e.g., 'cptac', 'panda').
        
    Returns:
        The provider instance.
        
    Raises:
        KeyError: If provider not found.
    """
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise KeyError(f"Unknown provider: {name}. Available: {available}")
    return _PROVIDERS[name]


def list_providers() -> dict[str, DatasetProvider]:
    """Get all registered providers."""
    return dict(_PROVIDERS)


# Auto-register providers on import
def _auto_register():
    """Import and register all built-in providers."""
    from patho_bench_cli.providers.cptac import CPTACProvider
    from patho_bench_cli.providers.panda import PANDAProvider
    from patho_bench_cli.providers.imp import IMPProvider
    from patho_bench_cli.providers.ovarian_bevacizumab import OvarianBevacizumabProvider
    from patho_bench_cli.providers.post_nat_brca import PostNatBrcaProvider
    from patho_bench_cli.providers.idr import IDRProvider
    from patho_bench_cli.providers.bcnb import BCNBProvider
    from patho_bench_cli.providers.bioimage import BioimageProvider
    from patho_bench_cli.providers.hancock import HancockProvider
    from patho_bench_cli.providers.boehmk import BoehmkProvider
    
    register_provider(CPTACProvider())
    register_provider(PANDAProvider())
    register_provider(IMPProvider())
    register_provider(OvarianBevacizumabProvider())
    register_provider(PostNatBrcaProvider())
    register_provider(IDRProvider())
    register_provider(BCNBProvider())
    register_provider(BioimageProvider())
    register_provider(HancockProvider())
    register_provider(BoehmkProvider())


_auto_register()
