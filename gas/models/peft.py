import torch
from deepeval.models import DeepEvalBaseLLM
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

from gas.commons import (
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    PENALTY_ALPHA,
    SEED,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    TRUNCATION,
)
from gas.logger import Logger

logger = Logger().get_logger()


class PEFTModel(DeepEvalBaseLLM):
    """
    PEFT model for evaluation.
    """

    def __init__(self, **kwargs):
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load the model with CUDA support
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=kwargs.get("path", None),
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            cache_dir=kwargs.get("cache_dir", None),
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.peft_config["default"].base_model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token

    def load_model(self) -> AutoPeftModelForCausalLM:
        """
        Load the model.
        """
        return self.model

    def generate(self, prompt: str) -> str:
        """
        Generate text based on the prompt.
        Args:
            prompt: The input prompt.
        Returns:
            The generated text.
        """
        # Tokenize input and set attention mask
        model = self.load_model()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=TRUNCATION,
        )

        # Ensure input tensors are moved to the same device as the model
        device = model.device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        torch.manual_seed(SEED)

        # Generate text with detailed parameters
        generation_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "min_new_tokens": MIN_NEW_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,  # Control randomness
            "top_k": TOP_K,  # Top-k sampling
            "top_p": TOP_P,  # Nucleus sampling
            "penalty_alpha": PENALTY_ALPHA,
            "do_sample": DO_SAMPLE,  # Enable sampling
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Generate output
        outputs = model.generate(**generation_params)
        new_tokens = outputs[0][len(input_ids[0]) :]
        logger.debug(len(new_tokens))
        logger.debug(new_tokens)
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return generated_text

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate text based on the prompt.
        Args:
            prompt: The input prompt.
        Returns:
            The generated text.
        """
        return self.generate(prompt)

    def get_model_name(self):
        """
        Get the model name.
        """
        return "google/gemma-2-9b-it"
