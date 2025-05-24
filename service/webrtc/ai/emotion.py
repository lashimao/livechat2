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
# Use specific env vars for emotion prediction to decouple from main LLM config
DEFAULT_EMOTION_OPENAI_API_KEY = os.getenv("EMOTION_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")) # Fallback to general OPENAI_API_KEY
DEFAULT_EMOTION_OPENAI_BASE_URL = os.getenv("EMOTION_OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")) # Fallback to general OpenAI base

async def predict_emotion(message: str) -> str: # Removed client parameter
    """
    根据给定的消息文本预测情感
    
    参数:
        message (str): 用于情感分析的消息文本
        
    返回:
        str: 预测的情感类型，如'neutral'、'anger'、'joy'等
    """
    try:
        api_key = DEFAULT_EMOTION_OPENAI_API_KEY
        base_url = DEFAULT_EMOTION_OPENAI_BASE_URL

        if not api_key:
            logging.warning("EMOTION_OPENAI_API_KEY not set. Emotion prediction will be skipped.")
            return 'neutral'
        
        # 准备请求数据
        data = {
            "model": "gpt-4o-mini", # Using the model name as per task description
            "messages": [
                {
                    "role": "system",
                    "content": "你现在是一个虚拟形象的动作驱动器，你需要根据输入的虚拟形象的语言，驱动虚拟形象的动作和表情，请尽量输出得随机并丰富一些"
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "motion_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "string",
                                "enum": ["neutral", "anger", "joy", "sadness", "shy", "shy2", "smile1", "smile2", "unhappy"]
                            }
                        },
                        "required": ["result"],
                        "additionalProperties": False
                    }
                }
            }
        }
        
        # Always create a new client for this specific call, or use aiohttp if preferred.
        # For simplicity and to align with original structure's fallback, let's use aiohttp directly.
        # This avoids managing a separate OpenAI client instance in server.py for this one function.

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        # Construct the endpoint URL, ensuring no double slashes if base_url already ends with /v1
        endpoint_url = f"{base_url.rstrip('/')}/chat/completions"
        if not base_url.endswith("/v1") and not base_url.endswith("/v1/"): # common issue
             if "/chat/completions" not in base_url: # if it's just base, add /v1
                 endpoint_url = f"{base_url.rstrip('/')}/v1/chat/completions"


        logging.info(f"Sending emotion prediction request to: {endpoint_url} with model {data['model']}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint_url,
                headers=headers,
                json=data
            ) as response:
                # 检查响应状态
                if response.status == 200:
                    response_data = await response.json()
                    content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                    
                    try:
                        parsed_content = json.loads(content)
                        emotion = parsed_content.get('result', 'neutral')
                        logging.info(f"情感分析结果: {emotion}")
                        return emotion
                    except json.JSONDecodeError:
                        logging.error(f"无法解析JSON响应: {content}")
                        return 'neutral'
                else:
                    response_text = await response.text()
                    logging.error(f"API请求失败: {response.status} {response_text}")
                    return 'neutral'
            
    except Exception as e:
        logging.error(f"预测情感时出错: {e}")
        return 'neutral' 