from enum import Enum

class SupportedFrameworks(Enum):
    LIGHT_EVAL = "lighteval"
    ONE_CLICK_EVAL = "one_click_eval"

class SupportedServers(Enum):
    VLLM = "vllm"
    HFT_TEXT = "hft_text"
    HFT_IMAGE = "hft_image"
    HFT_VIDEO = "hft_video"

class JudgeBackend(Enum):
    OPENAI = "openai"
    VLLM = "vllm"
    LITELLM = "litellm"
    TRANSFORMERS = "transformers"
    TGI = "tgi"
    INFERENCE_PROVIDERS = "inference-providers"
    
API_KEY = "EMPTY"
BASE_URL = "http://localhost:8001/v1"
DUMMY_HF_ORG = "dummy_org"

TMP_MODEL_PATH = "/tmp/eval/model"
BASE_MODEL_TYPE = "BASE_MODEL_TYPE"
