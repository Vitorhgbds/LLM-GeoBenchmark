import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from deepeval.models import DeepEvalBaseLLM
from commons import *

# Define your CustomLlama32_1B class
class CustomLlama32_1B(DeepEvalBaseLLM):
    def __init__(self):
        # Load Llama 3.2 model locally with a smaller parameter count for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(PATH_CONSOLIDATED_LLAMA_3_2_1B)
        self.tokenizer = AutoTokenizer.from_pretrained(PATH_TOKENIZER_LLAMA_3_2_1B)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        # Set up a generation pipeline with constraints for JSON validity
        generator = pipeline(
            "text-generation",
            model=self.load_model(),
            tokenizer=self.tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Use lm-format-enforcer to enforce valid JSON schema for LLM output
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(generator.tokenizer, parser)

        # Generate and parse the output to match the JSON schema
        output_dict = generator(prompt, prefix_allowed_tokens_fn=prefix_function)
        output_text = output_dict[0]["generated_text"][len(prompt):]
        json_result = json.loads(output_text)  # Convert text to JSON

        # Return JSON formatted result as specified by schema
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Llama-3.2 1B"
