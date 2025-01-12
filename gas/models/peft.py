from typing import Any

from deepeval.models import DeepEvalBaseLLM
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from gas.logger import Logger

logging = Logger()
logger = logging.get_logger()


class BasePeftModel(DeepEvalBaseLLM):
    """
    Base model for evaluation.
    """

    def __init__(self, model_params: dict[str, Any], generation_params: dict[str, Any], **kwargs):
        self.model_params = model_params
        self.generation_params = generation_params
        self.seed = kwargs.get("seed", None)
        self.model_path_or_name: str = model_params.get("pretrained_model_name_or_path", "")
        self.model_name = self.model_path_or_name.split("/")[-1].replace(".", "_")
        
        self.model = None
        self.tokenizer = None

        self.should_apply_chat_template = model_params.get("should_apply_chat_template", True)

    def load_model(self, *args, **kwargs) -> AutoPeftModelForCausalLM:
        """
        Load the model.
        """
        if self.model:
            return self.model

        logger.debug("Trying to initialize peft model...")
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
 
        # Load the model with CUDA support
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_path_or_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            cache_dir=self.model_params.get("cache_dir", None),
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.peft_config["default"].base_model_name_or_path
            )
        self.pad_token_id = (
            self.tokenizer.pad_token_id 
            if self.tokenizer.pad_token_id 
            else self.tokenizer.pad_token_type_id
            )
        
        logger.debug(f"pad token id: {self.pad_token_id}")
        logger.debug("Done.")
        return self.model

    def generate(self, prompt: list[str] | str) -> str:
        """
        Generate text based on the prompt.
        Args:
            prompt: The input prompt.
        Returns:
            The generated text.
        """
        self.load_model()

        if self.seed:
            set_seed(self.seed)

        if self.should_apply_chat_template:
            logger.debug("trying to apply chat template...")
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            logger.debug(prompt)

        logger.debug("Trying to build inputs...")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        
        device = self.model.device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        outputs = self.model.generate(
            **{
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **self.generation_params,
                "pad_token_id": self.pad_token_id,
            }
        )
        new_tokens = outputs[0][len(input_ids[0]) :]
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

    def get_model_name(self) -> str:
        """
        Get the model name.
        Returns:
            (str): Model name
        """
        return self.model_name