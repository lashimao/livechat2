# 导入必要的库和模块
from fastapi import APIRouter, Depends, Request, Body
from fastapi.responses import StreamingResponse
import logging
import json
import os
from typing import Dict, List, Any, Optional, cast
from pydantic import BaseModel
from fastrtc import ReplyOnPause


# 添加一个数据模型来接收前端传入的配置
class InputData(BaseModel):
    webrtc_id: str
    llm_api_key: Optional[str] = None
    whisper_api_key: Optional[str] = None
    siliconflow_api_key: Optional[str] = None
    google_gemini_api_key: Optional[str] = None
    live_api_model_name: Optional[str] = None # For Google Live API model
    live_api_voice_name: Optional[str] = None # For Google Live API TTS voice
    llm_base_url: Optional[str] = None
    whisper_base_url: Optional[str] = None
    whisper_model: Optional[str] = None  # 添加ASR模型字段
    siliconflow_voice: Optional[str] = None
    ai_model: Optional[str] = None
    voice_output_language: Optional[str] = None
    text_output_language: Optional[str] = None
    system_prompt: Optional[str] = None
    user_name: Optional[str] = None
    max_context_length: Optional[int] = None
    mem0_api_key: Optional[str] = None
    next_action: Optional[str] = None
    is_camera_on: Optional[bool] = False  # 添加摄像头状态字段

# 摄像头状态请求模型
class CameraStateRequest(BaseModel):
    webrtc_id: str
    is_camera_on: bool  # True为开启，False为关闭

# 视频帧数据模型
class VideoFrameData(BaseModel):
    webrtc_id: str
    frame_data: str  # base64编码的视频帧数据
    timestamp: Optional[float] = None  # 可选的时间戳

# 添加用于内置服务的简化数据模型
class BuiltinServiceRequest(BaseModel):
    webrtc_id: str
    ai_model: Optional[str] = None
    whisper_model: Optional[str] = None  # 添加ASR模型字段
    voice_output_language: Optional[str] = None
    text_output_language: Optional[str] = None
    system_prompt: Optional[str] = None
    user_name: Optional[str] = None
    max_context_length: Optional[int] = None

# 创建一个字典来存储每个用户的配置
user_configs = {}

# 创建一个字典来存储每个用户的最新视频帧
user_video_frames = {}  # 格式: {webrtc_id: [最新帧, 次新帧]}

# 创建路由器
router = APIRouter()

# 全局变量，将在server.py中被初始化
stream = None
rtc_configuration = None  # TURN服务器配置变量
handle_config_update = None  # 配置更新处理函数

# 初始化函数，用于从server.py接收必要的对象
def init_router(stream_obj, rtc_config=None, config_handler=None):
    global stream, rtc_configuration, handle_config_update
    stream = stream_obj
    rtc_configuration = rtc_config  # 保存TURN服务器配置
    handle_config_update = config_handler  # 保存配置更新处理函数
    logging.info("路由器已初始化")

@router.get("/reset/{webrtc_id}")
async def reset(webrtc_id: str):
    from server import user_sessions, get_user_session
    logging.info(f"重置用户 {webrtc_id} 的聊天")
    if webrtc_id in user_sessions:
        user_sessions[webrtc_id]["messages"] = [user_sessions[webrtc_id]["messages"][0]]  # 保留系统提示
    else:
        get_user_session(webrtc_id)  # 如果不存在则创建新会话
    return {"status": "success"}

# 提供TURN服务器配置的路由
@router.get("/webrtc/ice-config")
async def get_ice_config():
    logging.info("客户端请求ICE配置")
    if rtc_configuration:
        return rtc_configuration
    else:
        return {"iceServers": []}  # 如果没有配置，返回空的ICE服务器列表

# 提供服务端和客户端之间的通用通信SSE流
@router.get("/events")
async def events(webrtc_id: str):
    async def output_stream():
        logging.info(f"客户端 {webrtc_id} 已连接到事件流")
        try:
            async for output in stream.output_stream(webrtc_id):
                if output.args and len(output.args) > 0:
                    # 获取输出数据，这已经是JSON字符串
                    json_data = output.args[0]
                    try:
                        # 尝试解析JSON，确保它是有效的JSON
                        event_data = json.loads(json_data)
                        logging.info(f"发送事件数据: {event_data}")
                        # 直接发送JSON数据
                        yield f"event: message\ndata: {json_data}\n\n"
                    except json.JSONDecodeError:
                        # 如果不是有效的JSON，则将其作为error类型的数据发送
                        # 这是为了向后兼容
                        logging.info(f"发送传统错误数据: {json_data}")
                        legacy_data = json.dumps({"type": "error", "data": json_data})
                        yield f"event: message\ndata: {legacy_data}\n\n"
        except Exception as e:
            logging.error(f"事件流错误: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    # 设置正确的响应头
    return StreamingResponse(
        output_stream(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

# 添加接收前端配置的接口
@router.post("/input_hook")
async def input_hook(data: InputData):
    logging.info(f"接收到用户 {data.webrtc_id} 的配置")
    # 将用户配置存储到字典中
    user_configs[data.webrtc_id] = data
    
    # 在提供给 stream.set_input 前，先在内部处理配置更新
    if handle_config_update:
        handle_config_update(data.webrtc_id, "config_updated", data)
    
    # 使用 set_input 将配置传递给 Stream 对象
    # 这样修改不会改变使用方式，但确保配置会被正确处理
    stream.set_input(data.webrtc_id, "config_updated", data)
    return {"status": "success"}

# 添加设置服务请求的接口
@router.post("/use_builtin_service")
async def use_builtin_service(data: BuiltinServiceRequest):
    logging.info(f"接收到用户 {data.webrtc_id} 的服务请求")
    
    # 获取当前配置以保留摄像头状态
    current_config = get_user_config(data.webrtc_id)
    is_camera_on = current_config.is_camera_on if current_config else False
    
    # 从环境变量中获取配置
    built_in_config = InputData(
        webrtc_id=data.webrtc_id,
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        whisper_api_key=os.environ.get("WHISPER_API_KEY", ""),
        siliconflow_api_key=os.environ.get("SILICONFLOW_API_KEY", ""),
        llm_base_url=os.environ.get("LLM_BASE_URL", ""),
        whisper_base_url=os.environ.get("WHISPER_BASE_URL", ""),
        whisper_model=data.whisper_model or os.environ.get("WHISPER_MODEL", "whisper-large-v3"),  # 使用ASR模型字段
        siliconflow_voice=os.environ.get("SILICONFLOW_VOICE", ""),
        ai_model=data.ai_model or os.environ.get("AI_MODEL", "gpt-4o-mini"),
        voice_output_language=data.voice_output_language or os.environ.get("VOICE_OUTPUT_LANGUAGE", "zh"),
        text_output_language=data.text_output_language or os.environ.get("TEXT_OUTPUT_LANGUAGE", "zh"),
        system_prompt=data.system_prompt or os.environ.get("SYSTEM_PROMPT", ""),
        user_name=data.user_name or os.environ.get("USER_NAME", ""),
        max_context_length=data.max_context_length or os.environ.get("MAX_CONTEXT_LENGTH", 20),
        mem0_api_key=os.environ.get("MEM0_API_KEY", ""),
        is_camera_on=is_camera_on  # 保留原有的摄像头状态
    )
    
    # 将内置配置存储到字典中
    user_configs[data.webrtc_id] = built_in_config
    
    # 在提供给 stream.set_input 前，先在内部处理配置更新
    if handle_config_update:
        handle_config_update(data.webrtc_id, "config_updated", built_in_config)
    
    # 使用 set_input 将配置传递给 Stream 对象
    stream.set_input(data.webrtc_id, "config_updated", built_in_config)
    
    logging.info(f"用户 {data.webrtc_id} 已请求服务")
    return {"status": "success", "message": "请求服务成功"}

@router.post("/ai-trigger")
async def ai_trigger(data: InputData):
    logging.info(f"接收到用户 {data.webrtc_id} 的AI触发请求")
    config = get_user_config(data.webrtc_id)
    logging.info(f"Next Action: {data.next_action}")
    stream.set_input(data.webrtc_id, "config_updated", config, data.next_action)
    cast(ReplyOnPause, stream.handlers[data.webrtc_id]).trigger_response()
    return {"status": "success", "message": "AI触发请求已接收"}

@router.post("/ai-trigger-reset")
async def ai_trigger_reset(data: InputData):
    logging.info(f"接收到用户 {data.webrtc_id} 的reset请求")
    config = get_user_config(data.webrtc_id)
    stream.set_input(data.webrtc_id, "config_updated", config, "")
    return {"status": "success", "message": "reset请求已接收"}

# 简化：控制摄像头状态的接口
@router.post("/camera-state")
async def set_camera_state(data: CameraStateRequest):
    logging.info(f"接收到用户 {data.webrtc_id} 的摄像头状态更新: {data.is_camera_on}")
    
    # 获取用户当前配置
    config = get_user_config(data.webrtc_id)
    
    if config:
        # 更新配置中的摄像头状态
        config.is_camera_on = data.is_camera_on
        # 将更新后的配置存储回字典
        user_configs[data.webrtc_id] = config
        
        # 在提供给 stream.set_input 前，如有必要先在内部处理配置更新
        if handle_config_update:
            handle_config_update(data.webrtc_id, "config_updated", config)
        
        # 使用 set_input 将更新后的配置传递给 Stream 对象
        stream.set_input(data.webrtc_id, "config_updated", config)
        
        logging.info(f"用户 {data.webrtc_id} 的摄像头状态已同步到配置: {data.is_camera_on}")
    else:
        logging.warning(f"找不到用户 {data.webrtc_id} 的配置，无法更新摄像头状态")
    
    return {"status": "success", "message": "摄像头状态已更新并同步到配置"}

# 简化：接收视频帧的接口
@router.post("/video-frame")
async def receive_video_frame(data: VideoFrameData):
    logging.info(f"接收到用户 {data.webrtc_id} 的视频帧")
    
    # 获取用户配置
    config = get_user_config(data.webrtc_id)
    
    # 仅在用户配置存在且摄像头开启时处理视频帧
    if config and config.is_camera_on:
        # 获取当前用户的视频帧列表，如果不存在则创建空列表
        if data.webrtc_id not in user_video_frames:
            user_video_frames[data.webrtc_id] = []
        
        # 添加新帧到列表开头（最新的放在前面）
        user_video_frames[data.webrtc_id].insert(0, {
            "frame_data": data.frame_data,
            "timestamp": data.timestamp or 0
        })
        
        # 仅保留最新的2帧
        if len(user_video_frames[data.webrtc_id]) > 2:
            user_video_frames[data.webrtc_id] = user_video_frames[data.webrtc_id][:2]
        
        # 通过set_input传递最新的视频帧数组（作为第五个参数）
        # 前四个参数分别是：用户ID，事件类型，配置对象，next_action
        stream.set_input(
            data.webrtc_id, 
            "config_updated", 
            config, 
            "", 
            user_video_frames[data.webrtc_id]
        )
        
        logging.info(f"用户 {data.webrtc_id} 的视频帧已传递到stream（共{len(user_video_frames[data.webrtc_id])}帧）")
        return {"status": "success", "message": "视频帧已接收并处理"}
    else:
        # 摄像头未开启，不处理视频帧
        if not config:
            logging.warning(f"找不到用户 {data.webrtc_id} 的配置，无法处理视频帧")
        elif not config.is_camera_on:
            logging.info(f"用户 {data.webrtc_id} 的摄像头未开启，忽略视频帧")
            
        # 清除之前可能保存的视频帧
        if data.webrtc_id in user_video_frames:
            user_video_frames[data.webrtc_id] = []
            
        return {"status": "ignored", "message": "摄像头未开启，视频帧已忽略"}
    
# 获取用户配置的函数
def get_user_config(webrtc_id: str) -> Optional[InputData]:
    return user_configs.get(webrtc_id)