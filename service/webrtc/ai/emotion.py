"""
情感分析模块
根据文本内容预测情感状态
"""

import os
import json
import logging
import aiohttp
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()


# 从环境变量获取默认 API 密钥和基础 URL
# DEFAULT_OPENAI_API_KEY = os.getenv("LLM_API_KEY", "") # No longer needed for local version
# DEFAULT_OPENAI_API_BASE_URL = os.getenv("LLM_BASE_URL", "") # No longer needed for local version

def predict_emotion(text_to_analyze: str) -> str: # Made synchronous, client param removed
    """
    Predicts emotion based on the input text using simple rules.
    This is a simplified local version for function calling integration.
    
    Args:
        text_to_analyze (str): The text to analyze for emotion.
        
    Returns:
        str: Predicted emotion string.
    """
    logging.info(f"Predicting emotion locally for: {text_to_analyze[:50]}...")
    text_lower = text_to_analyze.lower()
    
    # Simple rule-based emotion detection
    if "!" in text_to_analyze or "surprise" in text_lower or "wow" in text_lower:
        emotion = "surprised" # Assuming "surprised" is a valid enum for your avatar
    elif "?" in text_to_analyze or "hmm" in text_lower or "wonder" in text_lower:
        emotion = "thinking" # Assuming "thinking" is a valid enum
    elif "sad" in text_lower or "cry" in text_lower or "sorry" in text_lower and "not" not in text_lower :
        emotion = "sadness"
    elif "happy" in text_lower or "joy" in text_lower or "great" in text_lower or "wonderful" in text_lower:
        emotion = "joy"
    elif "angry" in text_lower or "hate" in text_lower:
        emotion = "anger"
    elif "shy" in text_lower or "blush" in text_lower:
        emotion = "shy" # Example, replace with actual valid emotion strings for your avatar
    else:
        emotion = "neutral" # Default emotion

    logging.info(f"Locally predicted emotion: {emotion}")
    return emotion