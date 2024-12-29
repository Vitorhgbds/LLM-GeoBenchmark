SEED = 42
GPT_JUDGE = "gpt-4o-mini"

MAX_LENGHT = 500  # Adjust for shorter output
NO_REPEAT_NGRAM_SIZE = 2  # Prevent repetitive phrases
REPETITION_PENALTY = 1.2  # Penalize repetition
TEMPERATURE = None  # Control randomness
TOP_K = None  # Top-k sampling
TOP_P = None  # Nucleus sampling
DO_SAMPLE = False  # Enable sampling
TRUNCATION = True  # Ensure truncation to max_length
RETURN_FULL_TEXT = False  # Avoid repeating the prompt
