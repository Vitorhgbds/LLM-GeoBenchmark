SEED = 42
GPT_JUDGE = "gpt-4o-mini"

MAX_LENGHT = 500  # Adjust for shorter output
NO_REPEAT_NGRAM_SIZE = 2  # Prevent repetitive phrases
REPETITION_PENALTY = 1.2  # Penalize repetition
TEMPERATURE = 0.7  # Control randomness
TOP_K = 50  # Top-k sampling
TOP_P = 0.9  # Nucleus sampling
DO_SAMPLE = True  # Enable sampling
TRUNCATION = True  # Ensure truncation to max_length
RETURN_FULL_TEXT = False  # Avoid repeating the prompt
