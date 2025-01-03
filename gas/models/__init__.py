import importlib
import os
from inspect import getmembers, isclass
from pathlib import Path
from typing import Any, List, Tuple, Type, TypeGuard

from deepeval.models import DeepEvalBaseLLM

# Initialize the __all__ and model_name_class_map variables
__all__: List[str] = []
model_name_class_map: dict[str, Type[DeepEvalBaseLLM]] = {}

# Define the folder path containing your files (adjust if needed)
folder_path = Path(__file__).parent  # Use __file__ to get the current script's folder


# Function to filter classes
def is_valid_class(member: Any, module_name: str) -> TypeGuard[type]:
    """Check if a member is a class defined in the given module."""
    return isclass(member) and member.__module__ == module_name


# Iterate over all Python files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".py") and file_name != "__init__.py":
        # Remove the ".py" extension to get the module name
        module_name = file_name[:-3]
        # Import the module dynamically
        module = importlib.import_module(f"{__name__}.{module_name}")

        # Get all classes defined in the module
        module_classes: List[Tuple[str, Type[object]]] = getmembers(
            module,
            lambda member: is_valid_class(member, module.__name__),  # noqa: B023
        )

        for class_name, class_obj in module_classes:
            # Add to __all__ and model_name_class_map
            __all__.append(class_name)
            model_name_class_map[class_name] = class_obj
