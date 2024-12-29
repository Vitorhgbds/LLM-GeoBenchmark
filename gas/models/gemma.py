import torch
from deepeval.models import DeepEvalBaseLLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    TextGenerationPipeline,
    pipeline,
)

from gas.commons import (
    DO_SAMPLE,
    MAX_LENGHT,
    NO_REPEAT_NGRAM_SIZE,
    REPETITION_PENALTY,
    RETURN_FULL_TEXT,
    SEED,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    TRUNCATION,
)


class Gemma_9B(DeepEvalBaseLLM):
    """
    Gemma 9B Instruct model for evaluation.
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
            "google/gemma-2-9b-it",
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

        # Load the tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
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

        torch.manual_seed(SEED)
        pipe: TextGenerationPipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
        )
        outputs = pipe(
            prompt,
            max_length=MAX_LENGHT,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,  # Prevent repetitive phrases
            repetition_penalty=REPETITION_PENALTY,  # Penalize repetition
            temperature=TEMPERATURE,  # Control randomness
            top_k=TOP_K,  # Top-k sampling
            top_p=TOP_P,  # Nucleus sampling
            do_sample=DO_SAMPLE,  # Enable sampling
            truncation=TRUNCATION,  # Ensure truncation to max_length
            return_full_text=RETURN_FULL_TEXT,  # Avoid repeating the prompt
        )
        generated_text = outputs[0]["generated_text"].strip()
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
