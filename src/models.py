import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric

# Define the custom LLM class
class CustomLlama32_1B(DeepEvalBaseLLM):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
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

        parser = JsonSchemaParser(schema.schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(generator.tokenizer, parser)

        output_dict = generator(prompt, prefix_allowed_tokens_fn=prefix_function)
        output_text = output_dict[0]["generated_text"][len(prompt):]
        json_result = json.loads(output_text)
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Llama-3.2 1B"

# Instantiate and evaluate
def run_evaluation():
    custom_llm = CustomLlama32_1B()
    metric = AnswerRelevancyMetric(model=custom_llm)
    
    # Define references and candidates for evaluation
    references = ["What are the applications of AI in geology?"]
    candidates = ["AI is used in geology for predictive modeling, data analysis, and exploration."]
    
    result = metric.measure(references=references, candidates=candidates)
    print("Evaluation Result:", result)

if __name__ == "__main__":
    run_evaluation()
