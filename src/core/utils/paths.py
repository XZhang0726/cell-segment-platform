"""
Project path utilities.

Provides common path operations used throughout the project.
"""
from pathlib import Path
from typing import Union, List


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root directory.
    """
    # Search parent directories for one containing setup.py.
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists():
            return parent
    # Fall back to the fourth parent directory of this module.
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure the directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_project_root() / "models"


def get_configs_dir() -> Path:
    """Get the configuration directory."""
    return get_project_root() / "configs"


def get_results_dir() -> Path:
    """Get the results directory."""
    return get_data_dir() / "results"


def list_files(
    directory: Union[str, Path],
    extensions: List[str] = None,
    recursive: bool = False
) -> List[Path]:
    """
    List files in a directory.

    Args:
        directory: Directory path.
        extensions: List of filename extensions, e.g. ['.png', '.jpg'].
        recursive: Search subdirectories recursively.

    Returns:
        List of file paths.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    files = []
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append(file_path)

    return sorted(files)


def get_relative_path(path: Union[str, Path], base: Union[str, Path] = None) -> Path:
    """
    Get a relative path.

    Args:
        path: Target path.
        base: Base path; defaults to the project root.

    Returns:
        Relative path.
    """
    path = Path(path).resolve()
    if base is None:
        base = get_project_root()
    else:
        base = Path(base).resolve()

    try:
        return path.relative_to(base)
    except ValueError:
        # Return an absolute path if the target is not under base.
        return path
