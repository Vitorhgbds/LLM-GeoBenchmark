import torch
from deepeval.models import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from gas.commons import (
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    PENALTY_ALPHA,
    SEED,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from gas.logger import Logger

logger = Logger().get_logger()


class K2(DeepEvalBaseLLM):
    """
    K2 model for evaluation.
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
        self.model = AutoModelForCausalLM.from_pretrained(
            "daven3/k2",
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            cache_dir=kwargs.get("cache_dir", None),
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

    def load_model(self) -> AutoModelForCausalLM:
        """
        Load the model.
        """
        return self.model

    def generate(self, messages: str) -> str:
        """
        Generate text based on the prompt.
        Args:
            prompt: The input prompt.
        Returns:
            The generated text.
        """
        model = self.load_model()

        inputs = self.tokenizer(
            messages,
            return_tensors="pt",
        )

        # Ensure input tensors are moved to the same device as the model
        device = model.device
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)

        torch.manual_seed(SEED)
        set_seed(SEED)

        # Generate text with detailed parameters
        generation_params = {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            "min_new_tokens": MIN_NEW_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,  # Control randomness
            "do_sample": DO_SAMPLE,  # Enable sampling
            "top_k": TOP_K,  # Top-k sampling
            "top_p": TOP_P,  # Nucleus sampling
            "penalty_alpha": PENALTY_ALPHA,
            # "repetition_penalty": REPETITION_PENALTY,
            # "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,  # Avoid repeating n-grams
            # "num_beams": 4,
            # "pad_token_id": self.tokenizer.pad_token_id,
            # "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Generate output
        outputs = model.generate(**generation_params)
        new_tokens = outputs[0][len(input_ids[0]) :]
        if Logger().is_verbose:
            logger.debug(f"[bold dark_magenta]Attention mask: [/bold dark_magenta]{attention_mask}")
            logger.debug(f"[bold dark_magenta]Total new tokens generated: [/bold dark_magenta]{len(new_tokens)}")
            logger.debug(f"[bold dark_magenta]New tokens generated: [/bold dark_magenta]{new_tokens}")
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
        return "daven3/k2"
