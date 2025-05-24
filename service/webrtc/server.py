import fastapi 
# from fastapi.responses import FileResponse # Keep for potential future use, but not for main flow
from fastrtc import ReplyOnPause, Stream, AdditionalOutputs # audio_to_bytes might be replaced or adapted if it was OGG specific
import logging
# import time # Keep for logging if needed
import numpy as np
import asyncio
import os
# from io import BytesIO # Keep for potential use with audio bytes
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple, Union, TypedDict, Literal, AsyncGenerator # Added more types
from contextlib import asynccontextmanager
import base64
import wave # For PCM audio handling

import google.generativeai as genai
import google.generativeai.types as genai_types # For Blob, Part, etc.
# from google.generativeai.types import Content, Part, Blob # Specific types

# Custom utils & routing
from utils import run_async, generate_sys_prompt, generate_unique_user_id, stream_utils 
from ai import AI_MODEL # Kept for generate_sys_prompt model parameter
from ai.emotion import predict_emotion # Import for emotion prediction
from routes import router, init_router, get_user_config, InputData # InputData now has google_api_key, gemini_model_name
from fastapi.middleware.cors import CORSMiddleware # Already here

load_dotenv() # Already here

from humaware_vad import HumAwareVADModel # VAD model for detecting speech pauses
vad_model = HumAwareVADModel() # Already here

# Default environment variables & constants
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") 
DEFAULT_GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest") # Ensure this model supports Live API
DEFAULT_AI_MODEL = DEFAULT_GEMINI_MODEL_NAME # For generate_sys_prompt compatibility

# Configuration for Emotion Prediction (OpenAI gpt-4o-mini)
DEFAULT_EMOTION_OPENAI_API_KEY = os.getenv("EMOTION_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")) # Fallback to general OpenAI key
DEFAULT_EMOTION_OPENAI_BASE_URL = os.getenv("EMOTION_OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"))


DEFAULT_TIME_LIMIT = int(os.getenv("TIME_LIMIT", "600")) 
DEFAULT_CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "10")) 

DEFAULT_VOICE_OUTPUT_LANGUAGE = 'ja' 
DEFAULT_TEXT_OUTPUT_LANGUAGE = 'zh'  
DEFAULT_SYSTEM_PROMPT = """命运石之门(steins gate)的牧濑红莉栖(kurisu),一个天才少女,性格傲娇,不喜欢被叫克里斯蒂娜"""
DEFAULT_USER_NAME = "用户"
SESSION_TIMEOUT = timedelta(seconds=DEFAULT_TIME_LIMIT)
CLEANUP_INTERVAL = 60 

user_sessions: Dict[str, Dict[str, Any]] = {} 
user_sessions_last_active: Dict[str, float] = {} 
# Store the actual Live API session objects from google.generativeai
active_live_sessions: Dict[str, genai_types.ChatSession] = {} # Using official SDK's session type

# Configure the genai client globally with API key for Gemini
if DEFAULT_GOOGLE_API_KEY: # This is for Gemini
    genai.configure(api_key=DEFAULT_GOOGLE_API_KEY)
    logging.info("Google Generative AI client configured with API key for Gemini.")
else:
    logging.warning("GOOGLE_API_KEY not found in environment for Gemini. Live API calls may fail.")

# Global client instance (optional, methods can be called directly on genai module after configure)
# client = genai.Client() # Not strictly necessary for aio.live.connect if genai is configured

def audio_numpy_to_pcm_bytes(audio_np: np.ndarray, input_sample_rate: int, target_sample_rate: int = 16000) -> bytes:
    if not isinstance(audio_np, np.ndarray): # Should not happen with fastRTC
        logging.error(f"audio_numpy_to_pcm_bytes: Expected numpy array, got {type(audio_np)}")
        return b''
    if audio_np.size == 0: # Handle empty audio input after potential resampling
        return b''
            
    return audio_np.tobytes()

async def cleanup_expired_sessions():
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            current_time = time.time()
            expired_ids = [
                webrtc_id for webrtc_id, last_active in list(user_sessions_last_active.items())
                if current_time - last_active > SESSION_TIMEOUT.total_seconds()
            ]
            for webrtc_id in expired_ids:
                logging.info(f"清理过期用户会话: {webrtc_id}")
                user_sessions.pop(webrtc_id, None)
                user_sessions_last_active.pop(webrtc_id, None)
                if webrtc_id in active_live_sessions:
                    logging.info(f"关闭并清理过期Live API会话: {webrtc_id}")
                    try:
                        # The official SDK session from aio.live.connect() might not have a close_async or close method.
                        # Typically, these sessions are managed by context managers or are implicitly closed when out of scope
                        # or when the underlying connection is severed.
                        # For now, just removing from our tracking dict. The connection might time out on server side.
                        # If a close method becomes available, it should be used here.
                        # await active_live_sessions[webrtc_id].close() # Or close_async()
                        logging.info(f"Live API session for {webrtc_id} removed from active pool. Actual closing depends on SDK behavior.")
                    except Exception as e: # Catch any error during hypothetical close
                        logging.error(f"关闭Live API会话时出错 for {webrtc_id}: {e}")
                    del active_live_sessions[webrtc_id]
            logging.info(f"清理完成，当前活跃用户会话数: {len(user_sessions)}, 当前活跃Live API会话数: {len(active_live_sessions)}")
        except Exception as e:
            logging.error(f"清理过期会话时出错: {e}")

def get_user_session(webrtc_id: str) -> Dict[str, Any]: 
    user_sessions_last_active[webrtc_id] = time.time()
    if webrtc_id not in user_sessions:
        config = get_user_config(webrtc_id) 
        sys_prompt_template = config.system_prompt if config and config.system_prompt else DEFAULT_SYSTEM_PROMPT
        user_name = config.user_name if config and config.user_name else DEFAULT_USER_NAME
        model_name = config.gemini_model_name if config and config.gemini_model_name else DEFAULT_GEMINI_MODEL_NAME

        sys_prompt_content = generate_sys_prompt(
            voice_output_language=config.voice_output_language if config and config.voice_output_language else DEFAULT_VOICE_OUTPUT_LANGUAGE,
            text_output_language=config.text_output_language if config and config.text_output_language else DEFAULT_TEXT_OUTPUT_LANGUAGE,
            is_same_language=( (config.voice_output_language if config else None) == (config.text_output_language if config else None) ),
            current_user_name=user_name,
            system_prompt=sys_prompt_template, 
            model=model_name 
        )
        user_sessions[webrtc_id] = {
            "system_prompt_content": sys_prompt_content,
            "voice_output_language": config.voice_output_language if config else DEFAULT_VOICE_OUTPUT_LANGUAGE,
            "text_output_language": config.text_output_language if config else DEFAULT_TEXT_OUTPUT_LANGUAGE,
            "system_prompt_template": sys_prompt_template, 
            "user_name": user_name,
            "next_action": None, 
            # "last_live_api_handle" - The official SDK might not use explicit handles this way. Context is part of the session.
            "conversation_history": [genai_types.Content(role="system", parts=[genai_types.Part(text=sys_prompt_content)])]
        }
    
    session = user_sessions[webrtc_id]
    session.setdefault("conversation_history", [genai_types.Content(role="system", parts=[genai_types.Part(text=session.get("system_prompt_content", ""))])])
    if "system_prompt" in session and "system_prompt_template" not in session : 
        session["system_prompt_template"] = session.pop("system_prompt")
    session.setdefault("system_prompt_template", DEFAULT_SYSTEM_PROMPT)
    if "system_prompt_content" not in session: 
        model_name_for_prompt = get_user_ai_model(webrtc_id)
        session["system_prompt_content"] = generate_sys_prompt(
            voice_output_language=session.get("voice_output_language", DEFAULT_VOICE_OUTPUT_LANGUAGE),
            text_output_language=session.get("text_output_language", DEFAULT_TEXT_OUTPUT_LANGUAGE),
            is_same_language=session.get("is_same_language", True),
            current_user_name=session.get("user_name", DEFAULT_USER_NAME),
            system_prompt=session["system_prompt_template"],
            model=model_name_for_prompt
        )
    return session

# Renamed to reflect it's using the official google.generativeai Live API
async def get_google_live_session(webrtc_id: str) -> genai_types.ChatSession: # Type hint with official SDK type
    user_sessions_last_active[webrtc_id] = time.time() 

    if webrtc_id in active_live_sessions:
        return active_live_sessions[webrtc_id]

    if not genai.get_api_key(): # Check if API key is configured
        raise ConnectionError("Google API Key not configured for Generative AI client.")

    user_session_data = get_user_session(webrtc_id) 
    config_from_fe = get_user_config(webrtc_id) 
    
    model_to_use = config_from_fe.gemini_model_name if config_from_fe and config_from_fe.gemini_model_name else DEFAULT_GEMINI_MODEL_NAME
    user_session_data["model_name_in_use"] = model_to_use 

    # Configuration for the official Live API
    # The official `client.aio.live.connect` takes these as direct parameters or nested config objects.
    # Based on `Get_started_LiveAPI.ipynb`, it's direct parameters.
    try:
        logging.info(f"Attempting to connect to Google Live API for {webrtc_id} with model {model_to_use}")
        # The notebook uses client.aio.live.connect()
        # Assuming global `genai` is configured, we can use `genai.Client().aio.live.connect()`
        # or if the SDK structure is flat after `import google.generativeai as genai`, it might be `genai.live.connect()`
        # Let's assume `genai.Client().aio.live.connect()` as per prompt.
        # If `genai.Client()` needs to be instantiated:
        client = genai.Client() # This might pick up global config or need specific auth.
        
        # History needs to be in genai_types.Content format
        history_for_api = list(user_session_data["conversation_history"]) # Make a copy

        live_session = await client.aio.live.connect(
            model=model_to_use,
            system_instruction=genai_types.Content(role="system", parts=[genai_types.Part(text=user_session_data["system_prompt_content"])]),
            history=history_for_api if history_for_api else None, # Pass existing history if any
            input_audio_config=genai_types.AudioConfig(sample_rate_hertz=16000), # PCM is default encoding
            output_audio_config=genai_types.AudioConfig(sample_rate_hertz=24000), # PCM is default encoding
            response_modalities=[genai_types.ResponseModality.AUDIO], # Primary output is AUDIO
            output_audio_transcription_config=genai_types.OutputAudioTranscriptionConfig() # Enable transcription of generated audio
        )
        active_live_sessions[webrtc_id] = live_session
        logging.info(f"Google Live API session established for {webrtc_id}")
        return live_session
    except Exception as e:
        logging.error(f"Failed to connect to Google Live API for {webrtc_id}: {e}")
        if webrtc_id in active_live_sessions: # Should not be the case if connect failed, but defensive
            del active_live_sessions[webrtc_id]
        raise 

def get_user_ai_model(webrtc_id: str) -> str: 
    config = get_user_config(webrtc_id)
    if config and config.gemini_model_name:
        return config.gemini_model_name
    # Fallback for older InputData that might have 'ai_model'
    if config and hasattr(config, 'ai_model') and config.ai_model:
         return config.ai_model 
    return DEFAULT_GEMINI_MODEL_NAME

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

async def start_up(webrtc_id: str) -> AsyncGenerator[Union[bytes, AdditionalOutputs], None]:
    user_session_data = get_user_session(webrtc_id)
    logging.info(f"用户会话数据 (startup Live API): {user_session_data.get('user_name', 'Unknown User')}")

    try:
        live_session = await get_live_gemini_session(webrtc_id)
        initial_user_content = "self_motivated_greeting" 
        logging.info(f"为 {webrtc_id} 发送初始自驱动消息给Live API: '{initial_user_content}'")

        await live_session.send_client_content_async(
            text=initial_user_content, 
            turn_complete=True 
        )
        
        user_session_data["conversation_history"].append({"role": "user", "parts": [{"text": initial_user_content}]})

        welcome_text_parts = []
        produced_audio_output = False

        async for event in live_session.receive_async(): 
            event_type = event.get("event_type")
            if event_type == "TEXT_CHUNK":
                text_chunk = event.get("text_chunk", "")
                welcome_text_parts.append(text_chunk)
                yield AdditionalOutputs(json.dumps({"type": "text_chunk", "data": text_chunk}))
            elif event_type == "AUDIO_CHUNK": 
                audio_chunk_bytes = event.get("audio_chunk")
                if audio_chunk_bytes:
                    yield audio_chunk_bytes 
                    produced_audio_output = True
            elif event_type == "TRANSCRIPT": 
                 transcript_data = {"type": "transcript", "data": event.get("transcript", ""), "is_final": event.get("is_final", False)}
                 yield AdditionalOutputs(json.dumps(transcript_data))
            elif event_type == "TURN_COMPLETE" and event.get("is_final"):
                if event.get("last_handle"): 
                    user_session_data["last_live_api_handle"] = event["last_handle"]
                    logging.info(f"Received last_handle for {webrtc_id} (startup): {event['last_handle']}")
                break 

        final_welcome_text = "".join(welcome_text_parts)
        if final_welcome_text:
            yield AdditionalOutputs(json.dumps({"type": "assistant_message", "data": final_welcome_text}))
        
        user_session_data["conversation_history"].append({"role": "model", "parts": [{"text": final_welcome_text, "audio_present": produced_audio_output}]})
        
        logging.info(f"Live API 启动响应 for {webrtc_id}: Text='{final_welcome_text if final_welcome_text else '[No text]'}', Audio Produced={produced_audio_output}")
        
        user_session_data["next_action"] = "share_memory" 
        yield AdditionalOutputs(json.dumps({"type": "next_action", "data": user_session_data["next_action"]}))

    except Exception as e:
        logging.error(f"Live API startup for {webrtc_id} failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        yield AdditionalOutputs(json.dumps({"type": "error", "data": f"AI Connection Error: {str(e)}"}))

async def echo( 
    audio: Optional[tuple[int, np.ndarray]], 
    message: str,  
    input_data: InputData, 
    next_action: str = "", 
    video_frames: Optional[List[Dict[str, Any]]] = None
) -> AsyncGenerator[Union[bytes, AdditionalOutputs], None]:
    
    user_session_data = get_user_session(input_data.webrtc_id)
    # logging.debug(f"echo Live API: webrtc_id={input_data.webrtc_id}, audio_present={audio is not None}, msg='{message}', next_action='{next_action}'")

    try:
        live_session = await get_live_gemini_session(input_data.webrtc_id)
        
        if next_action: 
            logging.info(f"用户 {input_data.webrtc_id} AI触发对话 (Live API): {next_action}")
            await live_session.send_client_content_async(text=next_action, turn_complete=True)
            user_session_data["conversation_history"].append({"role": "user", "parts": [{"text": next_action}]})

        elif audio and audio[1].size > 0:
            sample_rate, audio_np = audio 
            audio_pcm_16k_bytes = audio_numpy_to_pcm_bytes(audio_np, sample_rate, target_sample_rate=16000)
            
            if audio_pcm_16k_bytes:
                await live_session.send_realtime_input_async(audio_bytes=audio_pcm_16k_bytes)
                # logging.debug(f"User {input_data.webrtc_id} sent audio (PCM 16kHz) to Live API, {len(audio_pcm_16k_bytes)} bytes.")
                # Simplified history update for audio chunks
                if not user_session_data["conversation_history"] or user_session_data["conversation_history"][-1].get("role") != "user_audio_stream":
                    user_session_data["conversation_history"].append({"role": "user_audio_stream", "parts": [{"audio_bytes_sent": len(audio_pcm_16k_bytes)}]})
                else:
                    user_session_data["conversation_history"][-1]["parts"][0]["audio_bytes_sent"] += len(audio_pcm_16k_bytes)
            else:
                logging.warning(f"User {input_data.webrtc_id}: Audio processing resulted in empty bytes, not sending.")

            if message == "silence": # VAD triggered, user stopped speaking
                 logging.info(f"User {input_data.webrtc_id} detected silence (VAD), sending turn_complete=True to Live API.")
                 await live_session.send_client_content_async(turn_complete=True)
                 if user_session_data["conversation_history"] and user_session_data["conversation_history"][-1].get("role") == "user_audio_stream":
                    user_session_data["conversation_history"][-1]["parts"][0]["turn_ended_with_silence"] = True
        
        if input_data.is_camera_on and video_frames:
            # logging.info(f"User {input_data.webrtc_id} sending {len(video_frames)} video frames to Live API.")
            for frame_info in video_frames:
                frame_b64 = frame_info['frame_data']
                image_bytes = base64.b64decode(frame_b64)
                await live_session.send_realtime_input_async(image_bytes=image_bytes, mime_type="image/jpeg") # Assuming API supports this

        full_response_text_parts = []
        produced_audio_output = False
        
        async for event in live_session.receive_async(): 
            event_type = event.get("event_type")
            if event_type == "TEXT_CHUNK":
                text_chunk = event.get("text_chunk", "")
                full_response_text_parts.append(text_chunk)
                yield AdditionalOutputs(json.dumps({"type": "text_chunk", "data": text_chunk}))
            elif event_type == "AUDIO_CHUNK": 
                audio_chunk_bytes = event.get("audio_chunk")
                if audio_chunk_bytes:
                    yield audio_chunk_bytes 
                    produced_audio_output = True
            elif event_type == "TRANSCRIPT": 
                 transcript_data = {"type": "transcript", "data": event.get("transcript", ""), "is_final": event.get("is_final", False)}
                 yield AdditionalOutputs(json.dumps(transcript_data))
            elif event_type == "TURN_COMPLETE" and event.get("is_final"):
                if event.get("last_handle"): 
                    user_session_data["last_live_api_handle"] = event["last_handle"]
                    # logging.info(f"Received last_handle for {input_data.webrtc_id} (echo): {event['last_handle']}")
                break 

        final_response_text = "".join(full_response_text_parts)
        if final_response_text:
            yield AdditionalOutputs(json.dumps({"type": "assistant_message", "data": final_response_text}))
        
        user_session_data["conversation_history"].append({"role": "model", "parts": [{"text": final_response_text, "audio_present": produced_audio_output}]})

        if not final_response_text and not produced_audio_output and (next_action or (audio and audio[1].size > 0)):
            logging.info(f"Live API for {input_data.webrtc_id} produced no text or audio output for this turn.")
        
        user_session_data["next_action"] = "share_memory" 
        yield AdditionalOutputs(json.dumps({"type": "next_action", "data": user_session_data["next_action"]}))

    except Exception as e:
        logging.error(f"Live API echo processing error for {input_data.webrtc_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        yield AdditionalOutputs(json.dumps({"type": "error", "data": f"AI Processing Error: {str(e)}"}))

def startup_wrapper_sync(*args: Any) -> Any:
    webrtc_id_to_use = ""
    if len(args) > 1: # Try to determine webrtc_id from common fastRTC call patterns
        if isinstance(args[1], str): webrtc_id_to_use = args[1]
        elif hasattr(args[1], 'webrtc_id'): webrtc_id_to_use = args[1].webrtc_id
        elif len(args) > 3 and hasattr(args[3], 'webrtc_id'): webrtc_id_to_use = args[3].webrtc_id 
    
    if not webrtc_id_to_use: # Fallback if ID couldn't be determined
        logging.error(f"startup_wrapper_sync: Could not determine webrtc_id from args: {args}")
        return iter([]) 
        
    # logging.info(f"startup_wrapper_sync called for webrtc_id: {webrtc_id_to_use}")
    return stream_utils.async_generator_to_sync_iterator(start_up(webrtc_id_to_use))

def echo_wrapper_sync(audio: Optional[tuple[int, np.ndarray]], message: str, input_data: InputData, 
                      next_action: str = "", video_frames: Optional[List[Dict[str, Any]]] = None) -> Any:
    return stream_utils.async_generator_to_sync_iterator(
        echo(audio, message, input_data, next_action, video_frames)
    )

reply_handler = ReplyOnPause(echo_wrapper_sync, 
    startup_fn=startup_wrapper_sync, 
    can_interrupt=True, 
    model=vad_model 
)