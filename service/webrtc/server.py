# 导入必要的库和模块
import fastapi  # 用于创建Web API服务
from fastapi.responses import FileResponse  # 用于返回文件响应
from fastrtc import ReplyOnPause, Stream, AdditionalOutputs, audio_to_bytes  # 用于处理WebRTC流
import logging  # 用于记录日志
import time  # 用于计时和时间相关操作
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware  # 用于处理跨域请求
import numpy as np  # 用于数值计算和数组操作
import io  # 用于处理输入输出流
import base64 # Added for TTS audio processing
import requests  # 用于发送HTTP请求
import asyncio  # 用于异步编程
from mem0 import AsyncMemoryClient
import os  # 用于操作系统相关功能
from io import BytesIO  # 用于在内存中处理二进制数据
from dotenv import load_dotenv  # 用于加载环境变量
import aiohttp  # 用于异步HTTP请求
import json  # 用于JSON处理
from datetime import datetime, timedelta
from typing import Dict, Optional
from openai import OpenAI
# 导入自定义的工具函数
from utils import run_async, generate_sys_prompt, process_llm_stream, generate_unique_user_id
from ai import ai_stream, AI_MODEL, predict_emotion  # 从ai模块导入
from ai.plan import ActionPlanner  # 导入ActionPlanner类
from .ai.google_live_api import GoogleLiveClient # Added GoogleLiveClient import
from stt import transcribe # Keep for potential fallback or other uses, though primary STT is changing
from tts import text_to_speech_stream
from routes import router, init_router, get_user_config, InputData  # 导入路由模块及用户配置
from contextlib import asynccontextmanager

# 加载默认环境变量（作为备用）
load_dotenv()

from humaware_vad import HumAwareVADModel
vad_model = HumAwareVADModel()

# 获取默认环境变量
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
DEFAULT_WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "")
DEFAULT_SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
DEFAULT_GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", "")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.ephone.ai/v1")
DEFAULT_WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "https://amadeus-ai-api-2.zeabur.app/v1")
DEFAULT_AI_MODEL = os.getenv("AI_MODEL")
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
DEFAULT_MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
# 添加WebRTC流的时间限制和并发限制环境变量
DEFAULT_TIME_LIMIT = int(os.getenv("TIME_LIMIT", "600"))
DEFAULT_CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "10"))

# 设置默认的语言选项和参数
DEFAULT_VOICE_OUTPUT_LANGUAGE = 'ja'
DEFAULT_TEXT_OUTPUT_LANGUAGE = 'zh'
DEFAULT_SYSTEM_PROMPT = """命运石之门(steins gate)的牧濑红莉栖(kurisu),一个天才少女,性格傲娇,不喜欢被叫克里斯蒂娜""" # This is the raw system prompt
DEFAULT_USER_NAME = "用户"
DEFAULT_MAX_CONTEXT_LENGTH = 20 # Max number of turns (user + model messages) for API history (e.g., 10 user + 10 model = 20 total messages)
DEFAULT_LIVE_API_MODEL = "gemini-1.5-flash-latest" # Default model for Google Live API
DEFAULT_LIVE_API_VOICE = "echo-alloy" # Placeholder default voice for Google Live API TTS
# 会话超时设置
SESSION_TIMEOUT = timedelta(seconds=DEFAULT_TIME_LIMIT)
# 清理间隔
CLEANUP_INTERVAL = 60

# 用户会话状态字典，存储每个用户的消息、设置等
user_sessions = {}
# 用户会话最后活动时间
user_sessions_last_active = {}

# 初始化OpenAI客户端字典，为每个用户创建一个客户端
openai_clients = {}

# 异步清理过期会话
async def cleanup_expired_sessions():
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            current_time = time.time()
            expired_sessions = []
            
            # 查找过期会话
            for webrtc_id, last_active in user_sessions_last_active.items():
                if current_time - last_active > SESSION_TIMEOUT.total_seconds():
                    expired_sessions.append(webrtc_id)
            
            # 清理过期会话
            for webrtc_id in expired_sessions:
                logging.info(f"清理过期会话: {webrtc_id}")
                user_sessions.pop(webrtc_id, None)
                user_sessions_last_active.pop(webrtc_id, None)
                openai_clients.pop(webrtc_id, None)
                
            logging.info(f"清理完成，当前活跃会话数: {len(user_sessions)}")
        except Exception as e:
            logging.error(f"清理过期会话时出错: {e}")

# 获取用户特定的会话状态
def get_user_session(webrtc_id: str):
    # 更新用户最后活动时间
    user_sessions_last_active[webrtc_id] = time.time()
    
    if webrtc_id not in user_sessions:
        # 创建新用户的初始会话状态
        config = get_user_config(webrtc_id)
        voice_output_language = config.voice_output_language if config and config.voice_output_language else DEFAULT_VOICE_OUTPUT_LANGUAGE
        text_output_language = config.text_output_language if config and config.text_output_language else DEFAULT_TEXT_OUTPUT_LANGUAGE
        system_prompt = config.system_prompt if config and config.system_prompt else DEFAULT_SYSTEM_PROMPT
        user_name = config.user_name if config and config.user_name else DEFAULT_USER_NAME
        
        # 生成系统提示词
        sys_prompt = generate_sys_prompt(
            voice_output_language=voice_output_language,
            text_output_language=text_output_language,
            is_same_language=(voice_output_language == text_output_language),
            current_user_name=user_name,
            system_prompt=system_prompt,
            model=get_user_ai_model(webrtc_id)
        )
        
        # Create initial session state
        # 'messages' will store conversation history in the format:
        # [{"role": "user", "parts": [{"text": "..."}, ...]}, {"role": "model", "parts": [{"text": "..."}, ...]}]
        user_sessions[webrtc_id] = {
            "messages": [],  # Conversation history (user/model turns) starts empty
            "raw_system_prompt": system_prompt, # Store the original (potentially unformatted) system prompt text from config/defaults
            "system_prompt_text_for_api": sys_prompt, # Store the formatted system prompt for GoogleLiveClient constructor
            "voice_output_language": voice_output_language,
            "text_output_language": text_output_language,
            "user_name": user_name,
            "is_same_language": (voice_output_language == text_output_language),
            "next_action": None
        }
        # The system prompt is no longer the first item in "messages".
        # It's passed separately to GoogleLiveClient via system_instruction_text in its constructor.
    
    return user_sessions[webrtc_id]

# Helper function to trim conversation history
def trim_conversation_history(messages: list[dict], max_length: int) -> list[dict]:
    """
    Trims messages to the most recent max_length items.
    Each item in 'messages' is a turn object e.g. {"role": "user", "parts": [...]}.
    """
    if len(messages) > max_length:
        logging.info(f"Trimming conversation history from {len(messages)} to {max_length} turns.")
        return messages[-max_length:]
    return messages

# 获取用户的OpenAI客户端
def get_user_openai_client(webrtc_id: str):
    # 更新用户最后活动时间
    user_sessions_last_active[webrtc_id] = time.time()
    
    if webrtc_id not in openai_clients:
        config = get_user_config(webrtc_id)
        api_key = config.llm_api_key if config and config.llm_api_key else DEFAULT_LLM_API_KEY
        base_url = config.llm_base_url if config and config.llm_base_url else DEFAULT_LLM_BASE_URL   
        openai_clients[webrtc_id] = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    return openai_clients[webrtc_id]

# 获取用户的AI模型
def get_user_ai_model(webrtc_id: str):
    config = get_user_config(webrtc_id)
    return config.ai_model if config and config.ai_model else DEFAULT_AI_MODEL

# Get user-specific Google Live API model name
def get_user_live_api_model(webrtc_id: str) -> str:
    config = get_user_config(webrtc_id)
    return config.live_api_model_name if config and config.live_api_model_name else DEFAULT_LIVE_API_MODEL

# Get user-specific Google Live API TTS voice name
def get_user_live_api_voice(webrtc_id: str) -> str:
    config = get_user_config(webrtc_id)
    return config.live_api_voice_name if config and config.live_api_voice_name else DEFAULT_LIVE_API_VOICE

# 获取用户的语音转文本API配置
def get_user_whisper_config(webrtc_id: str):
    config = get_user_config(webrtc_id)
    return {
        "api_key": config.whisper_api_key if config and config.whisper_api_key else DEFAULT_WHISPER_API_KEY,
        "base_url": config.whisper_base_url if config and config.whisper_base_url else DEFAULT_WHISPER_BASE_URL,
        "model": config.whisper_model if config and config.whisper_model else DEFAULT_WHISPER_MODEL
    }

# 获取用户的文本转语音配置
def get_user_siliconflow_config(webrtc_id: str):
    config = get_user_config(webrtc_id)
    return {
        "api_key": config.siliconflow_api_key if config and config.siliconflow_api_key else DEFAULT_SILICONFLOW_API_KEY,
        "voice": config.siliconflow_voice if config and config.siliconflow_voice else None
    }

# 获取用户的MEM0记忆服务配置
def get_user_mem0_config(webrtc_id: str):
    config = get_user_config(webrtc_id)
    return {
        "api_key": config.mem0_api_key if config and config.mem0_api_key else DEFAULT_MEM0_API_KEY
    }

# 获取用户的Gemini API配置
def get_user_gemini_config(webrtc_id: str):
    config = get_user_config(webrtc_id)
    return {
        "api_key": config.google_gemini_api_key if config and config.google_gemini_api_key else DEFAULT_GOOGLE_GEMINI_API_KEY
    }

logging.basicConfig(level=logging.INFO)
rtc_configuration = {
    "iceServers": [
        {
            "urls": "turn:43.160.205.75:80",
            "username": "okabe",
            "credential": "elpsycongroo"
        },
    ]
}

def start_up(webrtc_id):
    logging.info(f"用户 {webrtc_id} 开始函数已执行")
    
    # 获取用户会话状态
    session = get_user_session(webrtc_id)
    logging.info(f"session: {session}")
    # 生成最新的系统提示词  
    current_sys_prompt = generate_sys_prompt(
        voice_output_language=session["voice_output_language"],
        text_output_language=session["text_output_language"],
        is_same_language=session["is_same_language"],
        current_user_name=session["user_name"],
        system_prompt=session["system_prompt"],
        model=get_user_ai_model(webrtc_id)
    )
    
    # 创建一个临时消息列表，包含系统提示和一个特定的用户消息
    temp_messages = [
        {"role": "system", "content": current_sys_prompt},
        {"role": "user", "content": "self_motivated"}
    ]

    logging.info(f"current_sys_prompt: {current_sys_prompt}")
    
    # 获取用户相关配置
    client = get_user_openai_client(webrtc_id)
    model = get_user_ai_model(webrtc_id)
    siliconflow_config = get_user_siliconflow_config(webrtc_id)
    
    # 生成用户唯一ID
    user_id = generate_unique_user_id(session["user_name"])
    
    # 使用封装的流处理函数
    welcome_text = ""
    stream_generator = process_llm_stream(
        client=client,
        messages=temp_messages,
        model=model,
        siliconflow_config=siliconflow_config,
        voice_output_language=session["voice_output_language"],
        is_same_language=session["is_same_language"],
        run_predict_emotion=run_predict_emotion,
        ai_stream=ai_stream,
        text_to_speech_stream=text_to_speech_stream,
        max_tokens=100,
        max_context_length=20,
    )
    
    # 处理生成器的输出
    for item in stream_generator:
        if isinstance(item, str):
            welcome_text = item
        else:
            yield item
    try:
        # 创建ActionPlanner实例
        action_planner = ActionPlanner(conversation_history=session["messages"][-2:])
        # 异步执行行动计划
        next_action = run_async(action_planner.plan_next_action, client)
        # 更新用户会话中的next_action字段
        session["next_action"] = next_action
        logging.info(f"初始下一步行动计划: {next_action}")
        
        # 通知前端下一步行动计划
        next_action_json = json.dumps({"type": "next_action", "data": next_action})
        yield AdditionalOutputs(next_action_json)
    except Exception as e:
        logging.error(f"规划初始下一步行动失败: {str(e)}")
        session["next_action"] = "share_memory"  # 失败时默认为分享记忆

# 定义一个异步函数来运行predict_emotion
async def run_predict_emotion(message, client=None):
    """
    异步运行predict_emotion函数
    
    参数:
        message (str): 用于情感分析的消息文本
        client (OpenAI): OpenAI客户端实例，可选
        
    返回:
        str: 预测的情感类型
    """
    return await predict_emotion(message, client)

# Removed process_llm_stream function as it's being replaced by GoogleLiveClient logic

# 定义echo函数，处理音频输入并返回音频输出
def echo(audio: tuple[int, np.ndarray], message: str, input_data: InputData, next_action = "", video_frames = None):
    # 获取用户会话状态
    session = get_user_session(input_data.webrtc_id)
    whisper_config = get_user_whisper_config(input_data.webrtc_id)
    logging.info(f"摄像头状态: {input_data.is_camera_on}")
    
    # 记录视频帧信息
    if video_frames and input_data.is_camera_on:
        num_frames = len(video_frames) if video_frames else 0
        logging.info(f"接收到 {num_frames} 帧视频数据")
    
    prompt = "[AI主动发起对话]next Action: " + next_action
    user_id = generate_unique_user_id(session["user_name"])
    
    # --- ASR Phase (Google Live API) ---
    if next_action == "": # Only do ASR if not an AI-triggered action
        stt_time = time.time()
        logging.info(f"User {input_data.webrtc_id} performing ASR using GoogleLiveClient")
        
        google_gemini_api_key = get_user_gemini_config(input_data.webrtc_id).get('api_key')
        if not google_gemini_api_key:
            logging.error(f"Google Gemini API key not found for {input_data.webrtc_id}. Skipping ASR.")
            return

        asr_model_name = get_user_ai_model(input_data.webrtc_id) # Use same model for ASR for now
        asr_client = GoogleLiveClient(
            api_key=google_gemini_api_key,
            model_name=asr_model_name,
            response_modalities=["TEXT"], # ASR only needs text
            enable_asr=True,
            enable_tts_transcription=False
        )
        
        asr_result = {}
        try:
            connected = run_async(asr_client.connect())
            if not connected:
                logging.error(f"Failed to connect to Google Live API for ASR for {input_data.webrtc_id}.")
                return

            audio_bytes = audio_to_bytes(audio)
            logging.info(f"Converted audio to {len(audio_bytes)} bytes for Google Live ASR.")
            run_async(asr_client.send_audio_chunk(audio_bytes))
            run_async(asr_client.send_activity_end())
            logging.info(f"ASR audio sent, awaiting transcription for {input_data.webrtc_id}...")

            async def receive_asr_prompt_helper(client: GoogleLiveClient, webrtc_id: str):
                final_prompt_text = ""
                ui_transcripts = []
                async for message in client.receive_messages():
                    logging.debug(f"ASR client message for {webrtc_id}: {message}")
                    processed = client.process_server_content_parts(message) # Use the static helper
                    
                    if processed.get("error"):
                        logging.error(f"ASR error from Google Live API for {webrtc_id}: {processed['error']}")
                        break
                    
                    if processed.get("input_transcription_text"):
                        transcript_segment = processed["input_transcription_text"]
                        ui_transcripts.append(transcript_segment) # For UI updates
                        if processed.get("input_transcription_is_final"):
                            final_prompt_text = transcript_segment # Overwrite with final if available
                            logging.info(f"ASR final segment for {webrtc_id}: '{final_prompt_text}'")
                            # Assuming the first final segment is the one we want for the prompt
                            break # Got the final ASR
                        else:
                            logging.info(f"ASR partial segment for {webrtc_id}: '{transcript_segment}'")
                return {"final_asr_prompt": final_prompt_text, "ui_transcripts": ui_transcripts}

            asr_result = run_async(receive_asr_prompt_helper(asr_client, input_data.webrtc_id))
            prompt = asr_result.get("final_asr_prompt", "")

        except Exception as e:
            logging.error(f"Error during Google Live ASR for {input_data.webrtc_id}: {type(e).__name__} - {e}")
            prompt = ""
        finally:
            if asr_client:
                run_async(asr_client.close())
                logging.info(f"Google Live ASR client closed for {input_data.webrtc_id}.")

        # Yield partial transcripts for UI
        if asr_result.get("ui_transcripts"):
            for ui_transcript in asr_result["ui_transcripts"]:
                yield AdditionalOutputs(json.dumps({"type": "transcript", "data": ui_transcript}))
        
        if not prompt:
            logging.info(f"Google Live ASR returned empty prompt for {input_data.webrtc_id}.")
            return
        logging.info(f"Google Live ASR final prompt for {input_data.webrtc_id}: '{prompt}' (Time: {time.time() - stt_time:.2f}s)")
        
        # Update session with user's ASR prompt
        session["messages"].append({"role": "user", "content": prompt})
        # Send transcript to UI (already done if we yielded partials, but can send final too)
        # yield AdditionalOutputs(json.dumps({"type": "transcript", "data": prompt})) # Redundant if partials sent
    
    # If next_action is set (AI initiated), 'prompt' is already populated from that.
    # If ASR was skipped due to no next_action, and ASR failed, 'prompt' would be empty.
    # If ASR succeeded, 'prompt' has the user's speech.

    if not prompt: # If prompt is still empty (e.g. AI-triggered action with no text, or ASR failed and next_action was also empty)
        logging.warning(f"No prompt available for LLM for {input_data.webrtc_id} (ASR might have failed or no input).")
        return

    # --- Unified ASR, LLM, TTS Phase using a single GoogleLiveClient session ---
    
    start_time = time.time() # Overall timer for the interaction
    final_asr_prompt = ""
    final_llm_response = ""

    # Get API key and model name once
    google_gemini_api_key = get_user_gemini_config(input_data.webrtc_id).get('api_key')
    if not google_gemini_api_key:
        logging.error(f"Google Gemini API key not found for {input_data.webrtc_id}. Cannot proceed.")
        return
    
    # Use new getters for Google Live API model and voice
    live_api_model_to_use = get_user_live_api_model(input_data.webrtc_id)
    live_api_voice_to_use = get_user_live_api_voice(input_data.webrtc_id)
    
    # Fetch the system prompt text that was formatted and stored in the session by get_user_session or handle_config_update.
    system_prompt_text_for_api = session.get("system_prompt_text_for_api", DEFAULT_SYSTEM_PROMPT)


    gemini_client = GoogleLiveClient(
        api_key=google_gemini_api_key,
        model_name=live_api_model_to_use, 
        response_modalities=["TEXT", "AUDIO"], 
        enable_asr=True,                      
        enable_tts_transcription=True, 
        system_instruction_text=system_prompt_text_for_api,
        voice_name=live_api_voice_to_use # Pass the selected voice name
    )

    try:
        logging.info(f"Attempting to connect single GoogleLiveClient for {input_data.webrtc_id} with System Instruction: '{system_prompt_text_for_api[:100]}...'")
        connected = run_async(gemini_client.connect())
        if not connected:
            logging.error(f"Failed to connect single GoogleLiveClient for {input_data.webrtc_id}.")
            return

        async def handle_live_interaction_loop(
            client: GoogleLiveClient, 
            initial_audio_data: tuple[int, np.ndarray], 
            current_session: dict, 
            webrtc_id: str,
            user_id_for_mem0: str 
        ):
            asr_prompt_finalized = "" 
            llm_response_parts = []
            tts_audio_items_list = []
            asr_ui_updates_list = [] 
            
            if next_action == "": 
                audio_bytes_for_asr = audio_to_bytes(initial_audio_data)
                logging.info(f"Sending initial audio ({len(audio_bytes_for_asr)} bytes) for ASR to {webrtc_id}.")
                await client.send_audio_chunk(audio_bytes_for_asr)
                await client.send_activity_end()
                logging.info(f"Initial audio and activity_end sent for ASR for {webrtc_id}.")
                has_sent_asr_to_llm = False
            else: 
                asr_prompt_finalized = prompt # This 'prompt' is from the outer scope, already set to next_action text
                logging.info(f"AI-triggered action for {webrtc_id}. Using prompt: '{asr_prompt_finalized}'")
                
                # Add AI-triggered user message to session history (formatted for API)
                # This is the first user message in this turn.
                # IMPORTANT: Use the Google API 'turns' structure: {"role": "user", "parts": [{"text": "..."}]}
                current_session["messages"].append({"role": "user", "parts": [{"text": asr_prompt_finalized}]})
                
                # Prepare turns for API: current message + prior history from session.
                # No Mem0 augmentation for AI-triggered actions for now.
                # History is already in current_session["messages"]
                turns_for_api = trim_conversation_history(list(current_session["messages"]), DEFAULT_MAX_CONTEXT_LENGTH)
                
                await client.send_text(turns=turns_for_api, turn_complete=True)
                has_sent_asr_to_llm = True 
                logging.info(f"Sent AI-triggered prompt with history to LLM for {webrtc_id}.")
            
            async for message in client.receive_messages():
                logging.debug(f"Unified client message for {webrtc_id}: {message}")
                processed = client.process_server_content_parts(message)

                if processed.get("error"):
                    logging.error(f"Error from Google Live API for {webrtc_id}: {processed['error']}")
                    break 

                if not has_sent_asr_to_llm and processed.get("input_transcription_text"):
                    asr_text_segment = processed["input_transcription_text"]
                    asr_ui_updates_list.append(asr_text_segment) 
                    
                    if processed.get("input_transcription_is_final"):
                        asr_prompt_finalized = asr_text_segment 
                        logging.info(f"Final ASR prompt for {webrtc_id}: '{asr_prompt_finalized}'")
                        
                        # Append actual user utterance to session messages (this is the "original" user input for this turn)
                        # Using the Google API 'turns' structure
                        current_session["messages"].append({"role": "user", "parts": [{"text": asr_prompt_finalized}]})
                        
                        mem0_config_local = get_user_mem0_config(webrtc_id)
                        memory_client_local = AsyncMemoryClient(api_key=mem0_config_local["api_key"])
                        search_results_local = await memory_client_local.search(query=asr_prompt_finalized, user_id=user_id_for_mem0, limit=3)
                        memories_text_local = "\n".join(m["memory"] for m in search_results_local if m.get("memory"))
                        
                        user_prompt_for_llm_augmented = asr_prompt_finalized
                        if memories_text_local:
                            user_prompt_for_llm_augmented = f"Relevant Memories/Facts from previous conversations (for your reference only, do not directly mention these unless relevant to the current query):\n{memories_text_local}\n\nUser's current message: {asr_prompt_finalized}"
                        
                        # Prepare turns for API: history (all messages *before* current original user prompt) + current *augmented* user prompt
                        # The history is current_session["messages"][:-1] (all up to, but not including, the just-added user prompt)
                        turns_for_api_history = list(current_session["messages"][:-1]) 
                        current_turn_for_api = [{"role": "user", "parts": [{"text": user_prompt_for_llm_augmented}]}]
                        # Note: The above `current_turn_for_api` replaces the last item if we consider full history.
                        # For Google's `turns` API, we send the history AND the current turn.
                        # The `turns` should represent the conversation up to the point of the current query.
                        # So, `turns_for_api_history` (already containing previous user/model turns)
                        # should be followed by the *new* user turn (augmented).
                        
                        final_turns_for_api = trim_conversation_history(turns_for_api_history + current_turn_for_api, DEFAULT_MAX_CONTEXT_LENGTH)
                        
                        await client.send_text(turns=final_turns_for_api, turn_complete=True)
                        has_sent_asr_to_llm = True
                        logging.info(f"Sent final ASR prompt with history/memories to LLM for {webrtc_id}.")
                
                if has_sent_asr_to_llm: 
                    if processed.get("llm_text"):
                        llm_response_parts.append(processed["llm_text"])
                    
                    if processed.get("tts_audio_bytes"):
                        tts_bytes = processed["tts_audio_bytes"]
                        sample_rate = 24000 
                        audio_arr = np.frombuffer(tts_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        tts_audio_items_list.append((sample_rate, audio_arr))

                    if processed.get("output_transcription_text"):
                        logging.info(f"TTS transcription for {webrtc_id}: {processed['output_transcription_text']}")

                    if processed.get("generation_complete"):
                        logging.info(f"LLM generation complete for {webrtc_id}.")
                        break 
            
            final_llm_response_text = "".join(llm_response_parts)
            if final_llm_response_text: 
                 # Using the Google API 'turns' structure
                 current_session["messages"].append({"role": "model", "parts": [{"text": final_llm_response_text}]})
            
            return {
                "final_asr_prompt": asr_prompt_finalized, 
                "asr_ui_updates": asr_ui_updates_list,
                "final_llm_response": final_llm_response_text,
                "tts_audio_items": tts_audio_items_list
            }

        interaction_results = run_async(
            handle_live_interaction_loop(gemini_client, audio, session, input_data.webrtc_id, user_id)
        )

        if interaction_results:
            final_asr_prompt = interaction_results.get("final_asr_prompt", prompt if next_action else "") 
            full_response = interaction_results.get("final_llm_response", "") 
            
            for asr_ui_update in interaction_results.get("asr_ui_updates", []):
                yield AdditionalOutputs(json.dumps({"type": "transcript", "data": asr_ui_update}))

            for tts_audio_item in interaction_results.get("tts_audio_items", []):
                yield tts_audio_item
            
            if full_response: 
                logging.info(f"Final LLM for {input_data.webrtc_id}: '{full_response[:100]}...'")
                
                user_input_for_mem0 = final_asr_prompt 
                if user_input_for_mem0: 
                    mem0_config = get_user_mem0_config(input_data.webrtc_id)
                    memory_client = AsyncMemoryClient(api_key=mem0_config["api_key"])
                    # Mem0 expects a list of {"role": str, "content": str}
                    # We use final_asr_prompt (original user) and full_response (model)
                    conversation_to_save_for_mem0 = [
                        {"role": "user", "content": user_input_for_mem0}, 
                        {"role": "assistant", "content": full_response} 
                    ]
                    run_async(memory_client.add(conversation_to_save_for_mem0, user_id=user_id))
                    logging.info(f"Saved conversation to Mem0 for {input_data.webrtc_id} (User: '{user_input_for_mem0[:50]}...', Assistant: '{full_response[:50]}...')")

            if full_response: 
                try:
                    openai_planning_client = get_user_openai_client(input_data.webrtc_id) 
                    # ActionPlanner expects history in OpenAI format.
                    # session["messages"] is now in Google API format.
                    # We need to adapt this or ActionPlanner. For now, just trim.
                    action_planner_history = trim_conversation_history(session["messages"], 5)
                    # Potentially convert action_planner_history to OpenAI format if ActionPlanner requires it strictly.
                    # For now, assume ActionPlanner can handle it or is adapted.
                    action_planner = ActionPlanner(conversation_history=action_planner_history) 
                    next_action_plan = run_async(action_planner.plan_next_action, openai_planning_client)
                    session["next_action"] = next_action_plan
                    logging.info(f"Next action plan for {input_data.webrtc_id}: {next_action_plan}")
                    yield AdditionalOutputs(json.dumps({"type": "next_action", "data": next_action_plan}))
                except Exception as e:
                    logging.error(f"ActionPlanner failed for {input_data.webrtc_id}: {e}")
                    session["next_action"] = "share_memory" 
        
        logging.info(f"Total interaction time for {input_data.webrtc_id}: {time.time() - start_time:.2f}s")

    except Exception as e:
        logging.error(f"Error during unified Google Live session for {input_data.webrtc_id}: {type(e).__name__} - {e}")
    finally:
        if gemini_client: 
            run_async(gemini_client.close())
            logging.info(f"Unified GoogleLiveClient closed for {input_data.webrtc_id}.")

# 创建一个包装函数来接收来自Stream的webrtc_id参数
def startup_wrapper(*args):
    logging.info(f"startup_wrapper: {args}")
    return start_up(args[1].webrtc_id)

# 使用echo函数直接作为回调
reply_handler = ReplyOnPause(echo,
    startup_fn=startup_wrapper,
    can_interrupt=True,
    model=vad_model
    )

# 创建Stream对象，用于处理WebRTC流
stream = Stream(reply_handler, 
            modality="audio",  # 设置模态为音频
            rtc_configuration=rtc_configuration,
            mode="send-receive",  # 设置模式为发送和接收
            time_limit=DEFAULT_TIME_LIMIT,
            concurrency_limit=DEFAULT_CONCURRENCY_LIMIT
        )

# 使用 lifespan 上下文管理器替代 on_event
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # 启动时执行的代码
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    yield
    # 关闭时执行的代码
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        logging.info("清理任务已取消")

# 创建FastAPI应用，使用lifespan参数
app = fastapi.FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置更新处理函数（用于处理用户配置更新）
def handle_config_update(webrtc_id: str, message: str, data: InputData): # Added type hints
    # data is an InputData instance, already updated in user_configs by routes.py
    if message == "config_updated": 
        logging.info(f"用户 {webrtc_id} 配置已更新: {data.model_dump_json(exclude_none=True, exclude={'frame_data'})}") # Log relevant parts of data
        
        # 1. Update session-specific information (prompts, languages) if the user has an active session
        if webrtc_id in user_sessions:
            session = user_sessions[webrtc_id]
            
            # Update session settings from InputData if new values are provided
            if data.voice_output_language is not None:
                session["voice_output_language"] = data.voice_output_language
            if data.text_output_language is not None:
                session["text_output_language"] = data.text_output_language
            if data.system_prompt is not None:
                session["raw_system_prompt"] = data.system_prompt # Store new raw system prompt
            if data.user_name is not None:
                session["user_name"] = data.user_name
            
            session["is_same_language"] = (session["voice_output_language"] == session["text_output_language"])
            
            # Regenerate system prompt as it might depend on the updated session settings or AI model
            # get_user_ai_model() will use the 'data' (latest config) to determine the model
            current_sys_prompt = generate_sys_prompt(
                voice_output_language=session["voice_output_language"],
                text_output_language=session["text_output_language"],
                is_same_language=session["is_same_language"],
                current_user_name=session["user_name"],
                system_prompt=session["system_prompt"],
                model=get_user_ai_model(webrtc_id) 
            )
            
            if session["messages"] and session["messages"][0]["role"] == "system":
                session["messages"][0]["content"] = current_sys_prompt
            else:
                session["messages"].insert(0, {"role": "system", "content": current_sys_prompt})
            logging.info(f"Session for {webrtc_id} updated with new system prompt and settings.")

        # 2. Update OpenAI client if LLM API key or base URL changed
        # The google_gemini_api_key is accessed via get_user_gemini_config(), not directly used for this OpenAI client.
        if webrtc_id in openai_clients:
            current_client = openai_clients[webrtc_id]
            
            # Determine the effective API key: use data if provided, else current client's, else default
            effective_api_key = DEFAULT_LLM_API_KEY # Start with global default
            if hasattr(current_client, 'api_key') and current_client.api_key:
                effective_api_key = current_client.api_key # Use current client's if exists
            if data.llm_api_key is not None: # Override with new data if provided
                effective_api_key = data.llm_api_key

            # Determine the effective base URL: use data if provided, else current client's, else default
            effective_base_url = DEFAULT_LLM_BASE_URL # Start with global default
            if hasattr(current_client, 'base_url') and current_client.base_url:
                client_base_url_str = str(current_client.base_url)
                if client_base_url_str != "None": # Check against str(None)
                    effective_base_url = client_base_url_str # Use current client's if exists and not None
            if data.llm_base_url is not None: # Override with new data if provided
                effective_base_url = data.llm_base_url
            
            # Re-initialize the client only if the effective settings are different from current client's,
            # or if an explicit update was provided via 'data'.
            # This avoids re-creating the client unnecessarily if settings remain the same.
            needs_update = False
            if not hasattr(current_client, 'api_key') or current_client.api_key != effective_api_key:
                needs_update = True
            if not hasattr(current_client, 'base_url') or str(current_client.base_url) != effective_base_url:
                needs_update = True
            
            if data.llm_api_key is not None or data.llm_base_url is not None: # Explicit intent to update
                needs_update = True

            if needs_update:
                openai_clients[webrtc_id] = OpenAI(
                    api_key=effective_api_key,
                    base_url=effective_base_url
                )
                logging.info(f"OpenAI client for {webrtc_id} re-initialized/updated.")
        elif data.llm_api_key is not None or data.llm_base_url is not None:
            # If client doesn't exist but keys are provided, create it.
            # This case might be covered by get_user_openai_client, but good to be robust.
             openai_clients[webrtc_id] = OpenAI(
                    api_key=data.llm_api_key if data.llm_api_key is not None else DEFAULT_LLM_API_KEY,
                    base_url=data.llm_base_url if data.llm_base_url is not None else DEFAULT_LLM_BASE_URL
                )
             logging.info(f"New OpenAI client for {webrtc_id} created due to config update.")

# 初始化路由器，传递配置处理函数
init_router(stream, rtc_configuration, handle_config_update)

# 挂载WebRTC流
stream.mount(app)

# 包含路由
app.include_router(router)

# 添加主函数，当脚本直接运行时启动uvicorn服务器
if __name__ == "__main__":
    import uvicorn
    logging.info("启动服务器，监听 0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)