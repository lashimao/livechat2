from dotenv import load_dotenv
import os

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.ephone.ai/v1")
WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "https://amadeus-ai-api-2.zeabur.app/v1")

