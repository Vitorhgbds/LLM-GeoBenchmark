SEED = 42
GPT_JUDGE = "gpt-4o-mini"

MAX_NEW_TOKENS = 150  # Adjust for shorter output
MIN_NEW_TOKENS = 2  # Adjust for shorter output
NO_REPEAT_NGRAM_SIZE = 2  # Prevent repetitive phrases
REPETITION_PENALTY = 1.1  # Penalize repetition in word level
PENALTY_ALPHA = 1.1  # penalize repetition with context awareness
TEMPERATURE = 0.2  # Control randomness
TOP_K = 40  # Top-k sampling
TOP_P = 0.75  # Nucleus sampling
DO_SAMPLE = True  # Enable sampling
TRUNCATION = True  # Ensure truncation to max_length
