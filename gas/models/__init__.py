from .gemma import Gemma_9B
from .llama import Llama3_8B
from .mistralai import Ministral_8B

__all__ = [
    "Gemma_9B",
    "Llama3_8B",
    "Ministral_8B",
]

model_name_class_map = {
    "Gemma_9B": Gemma_9B,
    "Llama3_8B": Llama3_8B,
    "Ministral_8B": Ministral_8B,
}
