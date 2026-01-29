import requests
import time
import json
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from lmformatenforcer import JsonSchemaParser
from gas.logger import Logger

logging = Logger()
logger = logging.get_logger()

class AWSModel(DeepEvalBaseLLM):
    """
    Custom LLM wrapper for DeepEval.
    Integrado com lmformatenforcer para garantir outputs JSON válidos via API.
    """

    def __init__(self, 
                 endpoint_url: str, 
                 model: str, 
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
        self.model_name = self.model.split("/")[-1].replace(".", "_").replace(":", "_")

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

        done = False
        retries = 0
        while not done:
            try:
                response = requests.post(
                    self.endpoint_url,
                    json=body,
                    headers=headers,
                    timeout=60
                )
                
                json_response = response.json()
                status_code = json_response.get("statusCode", None)
                if status_code != 200:
                    logger.warning(f"Received non-200 response: {status_code} - {json_response.get('body', '')}")
                    logger.info("Retrying in 10s...")
                    time.sleep(10)
                    continue
                
                text_output = json_response['body']['response']['output']['message']['content'][0]['text']
                done = True
                
            except Exception as e:
                logger.warning(f"Error while generating: {e}.\nResponse json: {json_response}\nTrying again")
                logger.exception(e.with_traceback(e.__traceback__))
                retries += 1
                if retries >= 10:
                    done = True
                    text_output = ""
                    logger.error("Max retries reached. Exiting generation loop with text_output set to empty string.")
            
        return text_output
    
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
        return self.model_name