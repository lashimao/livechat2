"""
文本转语音模块
提供将文本转换为语音的功能
"""

import logging
import os
import requests
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量获取默认的 API 密钥和语音模型
DEFAULT_SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
DEFAULT_SILICONFLOW_VOICE = os.getenv("SILICONFLOW_VOICE", "speech:siliconflow-kurisu:clzv7bjjm041fufyct2z0setm:mphrsbbmvrjfophbsted")

# 翻译API的URL
TRANSLATE_URL = "https://amadeus-translate-api.zeabur.app/translate"

def translate_text(text, target_language, source_language='zh'):
    """
    调用翻译API将文本从源语言翻译到目标语言
    
    参数:
        text (str): 要翻译的文本
        target_language (str): 目标语言代码
        source_language (str): 源语言代码，默认为'zh'
        
    返回:
        str: 翻译后的文本，如果翻译失败则返回原文本
    """
    if not text or not text.strip():
        return text
    
    try:
        # 设置请求数据
        data = {
            'text': text,
            'source_lang': source_language,
            'target_lang': target_language,
        }
        
        # 发送POST请求
        response = requests.post(TRANSLATE_URL, json=data)
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            # 假设翻译API返回的JSON格式包含一个'translated_text'字段
            # 根据实际API调整字段名
            translated_text = result.get('data', text)
            logging.info(f"文本翻译成功: {text} -> {translated_text}")
            return translated_text
        else:
            logging.error(f"翻译API返回错误: {response.status_code} {response.text}")
            return text
    except Exception as e:
        logging.error(f"调用翻译API时出错: {e}")
        return text

def text_to_speech_stream(text, voice=None, sample_rate=32000, target_language=None, source_language='zh', api_key=None):
    """
    将文本转换为语音流
    
    参数:
        text (str): 要转换为语音的文本
        voice (str): 使用的声音模型，如不指定则使用环境变量中的设置
        sample_rate (int): 采样率，默认为32000Hz
        target_language (str): 目标语言代码，如果指定则先将文本翻译为目标语言
        source_language (str): 源语言代码，默认为'zh'
        api_key (str): SiliconFlow API 密钥，如不指定则使用环境变量中的设置
        
    返回:
        generator: 生成(sample_rate, audio_array)元组的生成器
    """
    if not text or not text.strip():
        logging.warning("文本为空，不进行转换")
        return
    
    # 如果指定了目标语言且源语言与目标语言不同，才进行翻译
    if target_language and source_language != target_language:
        text = translate_text(text, target_language, source_language)
    
    # 如果未指定voice参数，则使用环境变量中的设置
    if voice is None:
        voice = DEFAULT_SILICONFLOW_VOICE
    
    # 如果未指定api_key参数，则使用环境变量中的设置
    if api_key is None:
        api_key = DEFAULT_SILICONFLOW_API_KEY
    
    # 检查API密钥是否有效
    if not api_key:
        logging.error("缺少SiliconFlow API密钥，无法进行文本转语音")
        return
        
    # 设置请求头
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    # 设置请求数据
    data = {
        'model': 'FunAudioLLM/CosyVoice2-0.5B',
        'input': text,
        'voice': voice,
        'sample_rate': sample_rate,
        'response_format': 'pcm',
    }
    
    try:
        # 发送请求，获取流式响应
        response = requests.post(
            'https://api.siliconflow.cn/v1/audio/speech',
            json=data,
            headers=headers,
            stream=True
        )
        
        # 处理流式响应
        if response.status_code == 200:
            # 创建一个缓冲区来存储接收到的数据
            buffer = bytearray()
            
            # 处理流式响应的每个块
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    # 将数据添加到缓冲区
                    buffer.extend(chunk)
                    
                    # 确保缓冲区大小是偶数（对于16位PCM）
                    if len(buffer) >= 4096 and len(buffer) % 2 == 0:
                        # 从缓冲区中提取数据
                        data_to_process = buffer[:4096]
                        buffer = buffer[4096:]
                        
                        # 将字节数据转换为音频数组
                        try:
                            # 将16位整数数据转换为-1到1之间的浮点数
                            audio_array = np.frombuffer(data_to_process, dtype=np.int16).astype(np.float32) / 32768.0
                            yield (sample_rate, audio_array)
                        except Exception as e:
                            logging.error(f"处理音频数据时出错: {e}")
            
            # 处理缓冲区中剩余的数据
            if len(buffer) > 0 and len(buffer) % 2 == 0:
                try:
                    # 将剩余的字节数据转换为音频数组
                    audio_array = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    yield (sample_rate, audio_array)
                except Exception as e:
                    logging.error(f"处理剩余音频数据时出错: {e}")
        else:
            logging.error(f"SILICONFLOW API返回错误: {response.status_code} {response.text}")
    except Exception as e:
        logging.error(f"调用文本转语音API时出错: {e}")    