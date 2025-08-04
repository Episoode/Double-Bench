# OpenAI API Configuration
OPENAI_API_KEY = 'your_openai_api_key_here'
OPENAI_BASE_URL = 'https://api.openai.com/v1'

# Gemini API Configuration
GEMINI_API_KEY = 'your_gemini_api_key_here'
GEMINI_BASE_URL = 'https://openrouter.ai/api/v1'

# Qwen API Configuration
QWEN_API_KEY = 'your_qwen_api_key_here'
QWEN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

# Model Configuration
QA_GENERATE_MODEL = 'gpt-4o'
QA_JUDGE_MODEL = 'google/gemini-2.5-pro-preview'
CAPTION_MODEL = 'gpt-4o'
GT_MODEL = 'Qwen2.5-VL-32B-Instruct'

# Image Processing Configuration
MIN_IMAGE_SIZE = 250
MIN_IMAGE_FILTER_SIZE = 200
DEFAULT_SCALE = 2.0

# Concurrency Control Configuration
MAX_CONCURRENT_REQUESTS = 16

# Document Processing Path Configuration
SOURCE_DIR = r'path/to/source/directory'
TARGET_DIR = r'path/to/target/directory'

# Single-hop QA Generation and Judging Paths
SINGLEHOP_QA_JSON_PATH = r'path/to/singlehop_qa.json'
SINGLEHOP_JUDGED_JSON_PATH = r'path/to/singlehop_qa_judged.json'
FINAL_SINGLEHOP_BENCH = r'path/to/singlehop_bench.json'

# Multi-hop QA Generation and Judging Paths
MULTIHOP_QA_JSON_PATH = r'path/to/multihop_qa.json'
MULTIHOP_JUDGED_JSON_PATH = r'path/to/multihop_qa_judged.json'
MULTIHOP_TEMP_PATH = r'path/to/multihop_temp.json'
FINAL_MULTIHOP_BENCH = r'path/to/multihop_bench.json'