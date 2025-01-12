from typing import Any
from deepeval.models import DeepEvalBaseLLM
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, TextGenerationPipeline, pipeline, set_seed
from gas.logger import Logger

logging = Logger()
logger = logging.get_logger()


class BaseModel(DeepEvalBaseLLM):
    """
    Base model for evaluation.
    """

    def __init__(self, model_params: dict[str,Any], generation_params: dict[str, Any], **kwargs):
        self.should_apply_chat_template = model_params.get("should_apply_chat_template", True)
        self.peft = model_params.get("peft", False)
        self.model = None
        self.model_path_or_name: str = model_params.get("pretrained_model_name_or_path", "")
        self.model_name = self.model_path_or_name.split("/")[-1].replace(".","_")
        self.seed = kwargs.get("seed", None)
        # Load the tokenizer
        tokenizer_name_or_path = model_params.get("tokenizer_name_or_path") if model_params.get("tokenizer_name_or_path", None) else self.model_path_or_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.generation_params = generation_params
        self.generation_config = GenerationConfig(
            **{
            **generation_params,
            "pad_token_id": self.tokenizer.pad_token_type_id,
            })
        
        self.pipeline = None
        
        
    def load_model(self, *args, **kwargs) -> AutoModelForCausalLM:
        """
        Load the model.
        """
        if self.model:
            return self.model
        
        logger.debug("Trying to initialize model...")
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if self.peft:
            auto = AutoPeftModelForCausalLM
        else:
            auto = AutoModelForCausalLM
        # Load the model with CUDA support
        self.model = auto.from_pretrained(
            pretrained_model_name_or_path=self.model_path_or_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            cache_dir=kwargs.get("cache_dir", None),
        )
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            framework="pt",
            tokenizer=self.tokenizer
            )
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
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        
        self.pipeline.generation_config = self.generation_config
        sequences: list = self.pipeline(prompt, return_full_text = False)
        
        return sequences[0]["generated_text"]

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
