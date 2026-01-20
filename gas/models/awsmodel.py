import requests
import time
import json
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from lmformatenforcer import JsonSchemaParser

class AWSModel(DeepEvalBaseLLM):
    """
    Custom LLM wrapper for DeepEval.
    Integrado com lmformatenforcer para garantir outputs JSON válidos via API.
    """

    def __init__(self, 
                 endpoint_url: str, 
                 model=None, 
                 generation_params: dict[str, any] = None, 
                 should_apply_chat_template: bool = False,
                 *args, **kwargs):
        """ 
        Constructor for AWSModel.
        
        Args:
            endpoint_url (str): URL of AWS endpoint API.
            model (_type_, optional):  . Defaults to None.
            generation_params (dict[str, any], optional): Generation params for the model. Defaults to None.
            should_apply_chat_template (bool, optional): Whether to apply chat template. Defaults to False.
        """
        self.endpoint_url = endpoint_url
        self.model = model
        self.generation_params = generation_params or {}
        self.should_apply_chat_template = should_apply_chat_template

    def load_model(self):
        return None

    def generate(self, prompt: list[str] | str, schema: BaseModel = None) -> str | BaseModel:
        """ Generate text based on the prompt.
        Args:
            prompt: The input prompt.
            schema: Optional Pydantic BaseModel schema for output formatting.
        Returns:
            The generated text.
        """
        headers = {"Content-Type": "application/json"}
        
        if schema:
            json_schema_str = json.dumps(schema.model_json_schema(), indent=2)
            prompt = f"{prompt}\n\nRespond ONLY with a JSON object that follows this schema:\n{json_schema_str}"

        body = {
            "prompt": prompt,
            "temperature": self.generation_params.get("temperature", 0.1),
            "top_p": self.generation_params.get("top_p", 0.9),
            "model": self.model,
            "max_tokens": self.generation_params.get("max_tokens", 4096),
            **self.generation_params,
        }

        while True:
            try:
                response = requests.post(
                    self.endpoint_url,
                    json=body,
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()
                
                data_raw = response.json()
                text_output = data_raw['body']['response']['output']['message']['content'][0]['text']
                
                return text_output

            except Exception as e:
                print(f"Erro na geração: {e}. Tentando novamente em 10s...")
                time.sleep(10)
                continue
    
    async def a_generate(self, prompt: list[str] | str, schema: BaseModel = None) -> str | BaseModel:
        """ Asynchronously generate text based on the prompt.
        Args:
            prompt: The input prompt.
            schema: Optional Pydantic BaseModel schema for output formatting.
        """
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        """ gets the model name

        Returns:
            str: the model name
        """
        return "aws_endpoint_custom_formatted"