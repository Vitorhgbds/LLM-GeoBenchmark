import torch
from deepeval.models import DeepEvalBaseLLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from gas.commons import (
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    NO_REPEAT_NGRAM_SIZE,
    SEED,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    TRUNCATION,
)
from gas.logger import Logger

logger = Logger().get_logger()
class K2(DeepEvalBaseLLM):
    """
    K2 model for evaluation.
    """

    def __init__(self):
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
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token

    def load_model(self) -> AutoModelForCausalLM:
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
            "temperature": TEMPERATURE,  # Control randomness
            "top_k": TOP_K,  # Top-k sampling
            "top_p": TOP_P,  # Nucleus sampling
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,  # Avoid repeating n-grams
            "do_sample": DO_SAMPLE,  # Enable sampling
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "min_new_tokens": MIN_NEW_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
        }

        # Generate output
        outputs = model.generate(**generation_params)
        new_tokens = outputs[0][len(input_ids[0]) :]
        # Decode generated tokens
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
