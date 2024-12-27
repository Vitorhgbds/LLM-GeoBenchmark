import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.models import DeepEvalBaseLLM

from gas.commons import SEED

class Llama3_8B(DeepEvalBaseLLM):
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
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        # Format the prompt for exam-style Q&A
        formatted_prompt = f"Q: {prompt}\nA:"
        # Tokenize input and set attention mask
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()

        # Set the random seed for reproducibility
        torch.manual_seed(SEED)

        # Generate output with controlled parameters
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=500,  # Limit output length
            no_repeat_ngram_size=2,  # Prevent repetitive phrases
            repetition_penalty=1.2,  # Penalize repetition
            temperature=0.1,  # Almost Fully deterministic
            top_p=1,  # Always select the most likely next token
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode and clean up the output
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to extract only the answer after "A:"
        answer_start = raw_output.find("A:") + 2
        final_output = raw_output[answer_start:].strip()
        return final_output

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3.2 8B"
    