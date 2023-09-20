import importlib


class ModuleInterface:
    @staticmethod
    def register() -> None:
        """Register Modules."""


def import_module(name: str) -> ModuleInterface:
    """Imports a module given a name."""
    return importlib.import_module(name)