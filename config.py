# config.py

# Tokens per chunk
CHUNK_SIZE = 500
# Tokens shared between neighboring chunks
CHUNK_OVERLAP = 50
# tiktoken encoding used to count tokens
TOKENIZER = "cl100k_base"

# JSON output formatting
JSON_INDENT = 4
JSON_ENSURE_ASCII = False
JSON_SORT_KEYS = False