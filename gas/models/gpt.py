from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI


class GPT4oMini(DeepEvalBaseLLM):
    def __init__(self):
        """
        Initialize the GPT4o-mini model using OpenAI API through OpenAI().
        """
        self.openai_client = OpenAI()

    def load_model(self):
        """
        No explicit model loading is required when using OpenAI().
        This is a placeholder to match the interface.
        """
        return None

    def generate(self, prompt: str) -> str:
        """
        Generate a response using GPT4o-mini via OpenAI().
        :param prompt: The input prompt for the model.
        :return: Generated text from the model.
        """
        response = self.openai_client.Completion.create(
            engine="gpt-4o-mini",  # Replace with the correct engine name if different
            prompt=prompt,
            max_tokens=2500,
            temperature=0.7,
            top_p=1.0,
            n=1,
            stop=None,
        )
        return response["choices"][0]["text"].strip()

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronous version of generate.
        :param prompt: The input prompt for the model.
        :return: Generated text from the model.
        """
        response = await self.openai_client.Completion.acreate(
            engine="gpt-4o-mini",  # Replace with the correct engine name if different
            prompt=prompt,
            max_tokens=2500,
            temperature=0.7,
            top_p=1.0,
            n=1,
            stop=None,
        )
        return response["choices"][0]["text"].strip()

    def get_model_name(self):
        """
        Return the name of the model.
        :return: Model name as a string.
        """
        return "GPT4o-mini"
