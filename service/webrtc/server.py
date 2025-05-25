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
import re # For parsing emotion tags

# Custom utils & routing
from utils import run_async, generate_sys_prompt, generate_unique_user_id, stream_utils
# Removed: process_llm_stream, ai_stream, predict_emotion, transcribe, text_to_speech_stream, AsyncMemoryClient
from ai import AI_MODEL # Kept for generate_sys_prompt model parameter
from routes import router, init_router, get_user_config, InputData 
from fastapi.middleware.cors import CORSMiddleware 

load_dotenv() 

from humaware_vad import HumAwareVADModel # VAD model for detecting speech pauses
vad_model = HumAwareVADModel() # Already here

# Default environment variables & constants
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") 
DEFAULT_GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest") # Ensure this model supports Live API
DEFAULT_AI_MODEL = DEFAULT_GEMINI_MODEL_NAME # For generate_sys_prompt compatibility

# DEFAULT_EMOTION_OPENAI_API_KEY and DEFAULT_EMOTION_OPENAI_BASE_URL are removed as emotion prediction is now via Gemini prompt.

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

    if not genai.get_api_key(): 
        config_from_fe = get_user_config(webrtc_id)
        api_key_to_use = config_from_fe.google_api_key if config_from_fe and config_from_fe.google_api_key else DEFAULT_GOOGLE_API_KEY
        if not api_key_to_use:
            raise ConnectionError("Google API Key not configured for Generative AI client.")
        genai.configure(api_key=api_key_to_use) # Configure if not already done globally

    user_session_data = get_user_session(webrtc_id) 
    config_from_fe = get_user_config(webrtc_id) 
    
    model_to_use = config_from_fe.gemini_model_name if config_from_fe and config_from_fe.gemini_model_name else DEFAULT_GEMINI_MODEL_NAME
    user_session_data["model_name_in_use"] = model_to_use 

    try:
        logging.info(f"Attempting to connect to Google Live API for {webrtc_id} with model {model_to_use}")
        
        # Use GenerativeModel instance for live connection as per typical SDK patterns for newer models
        # if client.aio.live.connect is not the intended path for gemini-1.5+
        model_instance = genai.GenerativeModel(
            model_name=model_to_use,
            system_instruction=user_session_data["system_prompt_content"] 
            # history can be passed here or in start_chat/connect_live
        )
        
        # The exact method for starting a live session might vary.
        # `connect_live()` is a placeholder if `start_chat()` doesn't return a live-compatible session.
        # The prompt mentioned `model.connect_live()` or `client.aio.live.connect`.
        # Assuming `GenerativeModel` has a method like `start_chat` that could be used for live interaction
        # or a specific `connect_live`. Given the context, will use a generic `start_chat` and assume it's live-compatible
        # if the SDK merges these concepts, or that a specific `connect_live` method would exist.
        # For this refactor, let's assume `model_instance.start_chat(history=...)` is the path and returns a live-capable session.
        # However, the original prompt for frontend used `ai.live.connect`, implying a direct live connection method.
        # Let's stick to the previous `client.aio.live.connect` as it's closer to the prompt's intent for a dedicated live API.
        client = genai.Client() 
        history_for_api = list(user_session_data["conversation_history"])

        live_session = await client.aio.live.connect(
            model=model_to_use, # Model name might need 'models/' prefix depending on SDK version
            system_instruction=genai_types.Content(role="system", parts=[genai_types.Part(text=user_session_data["system_prompt_content"])]),
            history=history_for_api if history_for_api else None,
            input_audio_config=genai_types.AudioConfig(sample_rate_hertz=16000), 
            output_audio_config=genai_types.AudioConfig(sample_rate_hertz=24000),
            response_modalities=[genai_types.ResponseModality.AUDIO], 
            output_audio_transcription_config=genai_types.OutputAudioTranscriptionConfig() 
        )
        active_live_sessions[webrtc_id] = live_session
        logging.info(f"Google Live API session established for {webrtc_id}")
        return live_session
    except Exception as e:
        logging.error(f"Failed to connect to Google Live API for {webrtc_id}: {e}")
        if webrtc_id in active_live_sessions: 
            del active_live_sessions[webrtc_id]
        raise 

def get_user_ai_model(webrtc_id: str) -> str: # This will primarily return the Gemini model name
    config = get_user_config(webrtc_id)
    if config and config.gemini_model_name: # Prioritize gemini_model_name from InputData
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
    logging.info(f"Placeholder start_up for {webrtc_id}. User: {user_session_data.get('user_name', 'Unknown')}")
    # This function is a placeholder. 
    # Actual Gemini Live API interaction for startup will be implemented in the next subtask.
    try:
        yield AdditionalOutputs(json.dumps({"type": "system_message", "data": "System ready. Placeholder for Gemini Live API startup."}))
        yield AdditionalOutputs(json.dumps({"type": "emotion", "data": "neutral"})) 
        yield AdditionalOutputs(json.dumps({"type": "next_action", "data": "initial_greeting_done"})) 
        
        if False: 
            yield b'' 
    except Exception as e:
        logging.error(f"Error in placeholder start_up for {webrtc_id}: {e}")
        yield AdditionalOutputs(json.dumps({"type": "error", "data": f"Startup error: {str(e)}"}))
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
    # This function will be heavily refactored in the next subtask to use Gemini Live API.
    # For now, it's a placeholder after removing old service calls.
    
    # Log inputs for now
    logging.info(f"Echo called for {input_data.webrtc_id}. Message: {message}, Next Action: {next_action}")
    if audio:
        logging.info(f"Audio received: {len(audio[1])} samples at {audio[0]} Hz")
    if video_frames:
        logging.info(f"Video frames received: {len(video_frames)}")

    # Placeholder: Simulate processing and yield some outputs
    # In the next subtask, this will be replaced with actual Gemini Live API interaction.
    
    # Simulate user speech transcript if audio was present (mimicking STT)
    if audio and audio[1].size > 0 and not next_action:
        simulated_transcript = f"User said something (audio length {len(audio[1])})"
        yield AdditionalOutputs(json.dumps({"type": "transcript", "data": simulated_transcript, "is_final": True}))
        # Add to history (placeholder, real STT would be used)
        user_session_data["conversation_history"].append(genai_types.Content(role="user", parts=[genai_types.Part(text=simulated_transcript)]))


    # Simulate AI response (text, audio, emotion)
    ai_response_text = f"AI responding to '{message if not next_action else next_action}'... (placeholder)"
    # Simulate emotion tag parsing (as if AI included it)
    emotion_to_send = "neutral"
    emotion_match = re.search(r"\[(neutral|joy|anger|sadness|shy|surprised|thinking)\]", ai_response_text, re.IGNORECASE)
    if emotion_match:
        emotion_to_send = emotion_match.group(1).lower()
        ai_response_text = ai_response_text.replace(emotion_match.group(0), "").lstrip()

    yield AdditionalOutputs(json.dumps({"type": "text_chunk", "data": ai_response_text}))
    yield AdditionalOutputs(json.dumps({"type": "assistant_message", "data": ai_response_text}))
    yield AdditionalOutputs(json.dumps({"type": "emotion", "data": emotion_to_send}))

    # Simulate sending some audio bytes (e.g., silent audio if nothing else)
    # This would be actual AI speech audio from Gemini.
    # Creating a short silent audio chunk for placeholder. 1 sec of 24kHz 16-bit mono.
    silent_audio_duration_ms = 200 
    num_frames = int(24000 * (silent_audio_duration_ms / 1000))
    silent_audio_bytes = b'\x00\x00' * num_frames 
    yield silent_audio_bytes 
    
    # Update history with AI's placeholder response
    user_session_data["conversation_history"].append(genai_types.Content(role="model", parts=[genai_types.Part(text=ai_response_text)]))

    # Placeholder for next action
    user_session_data["next_action"] = "another_placeholder_action"
    yield AdditionalOutputs(json.dumps({"type": "next_action", "data": user_session_data["next_action"]}))

    # Ensure the generator actually yields if no conditions are met above
    if not (audio and audio[1].size > 0 and not next_action): # if not already yielded transcript
        if not next_action: # if not AI triggered
             pass # No specific yield needed for pure silence without action
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