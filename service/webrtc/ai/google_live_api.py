# service/webrtc/ai/google_live_api.py
import asyncio
import websockets
import json
import logging
import os
import base64
from typing import Optional, List, Dict, Any, AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)

# Default model name for Google Live API, can be overridden by environment variable
DEFAULT_GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-1.5-flash-latest") # Ensure this is a live-capable model

class GoogleLiveClient:
    def __init__(self, 
                 api_key: str, 
                 model_name: Optional[str] = None, 
                 response_modalities: Optional[List[str]] = None,
                 enable_asr: bool = True,
                 enable_tts_transcription: bool = False,
                 system_instruction_text: Optional[str] = None,
                 tools_config: Optional[List[Dict[str, Any]]] = None,
                 voice_name: Optional[str] = None): # Added voice_name
        self.api_key = api_key
        self.model_name = model_name if model_name else DEFAULT_GEMINI_LIVE_MODEL
        self.system_instruction_text = system_instruction_text
        self.tools_config = tools_config
        self.voice_name = voice_name # Store voice_name
        
        # Updated WebSocket URL, API key is now passed in headers
        self.websocket_url = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        # Ensure response_modalities always includes TEXT if AUDIO is requested for TTS,
        # as text often accompanies audio. API might enforce this.
        if response_modalities and "AUDIO" in response_modalities and "TEXT" not in response_modalities:
            self.response_modalities = response_modalities + ["TEXT"]
        elif response_modalities:
            self.response_modalities = response_modalities
        else:
            self.response_modalities = ["TEXT"] # Default
            
        self.enable_asr = enable_asr
        self.enable_tts_transcription = enable_tts_transcription
        
        logging.info(f"GoogleLiveClient initialized for model {self.model_name}. ASR: {self.enable_asr}, TTS Transcription: {self.enable_tts_transcription}, System Instruction: {'Provided' if self.system_instruction_text else 'Not provided'}")

    async def connect(self, custom_setup_payload: Optional[Dict[str, Any]] = None, connect_timeout: float = 10.0, setup_timeout: float = 10.0) -> bool:
        """
        Establishes a WebSocket connection and sends the BidiGenerateContentSetup message.
        custom_setup_payload: Allows overriding the 'setup' part of the BidiGenerateContentSetup message.
        connect_timeout: Timeout in seconds for establishing the WebSocket connection.
        setup_timeout: Timeout in seconds for receiving the BidiGenerateContentSetupComplete message.
        Returns: True if connection and setup succeeded, False otherwise.
        """
        if self.ws_connection and not self.ws_connection.closed:
            logging.info("WebSocket connection already established.")
            return True
            
        try:
            headers = {"X-Goog-Api-Key": self.api_key, "User-Agent": "PythonWebSocketClient/1.0"} # Added User-Agent
            
            logging.info(f"Attempting to connect to WebSocket: {self.websocket_url}")
            self.ws_connection = await asyncio.wait_for(
                websockets.connect(self.websocket_url, extra_headers=headers),
                timeout=connect_timeout
            )
            logging.info("Successfully connected to Google Live API WebSocket.")

            # Construct BidiGenerateContentSetup message
            setup_config: Dict[str, Any] = { # Renamed to setup_config for clarity before merge
                "model": f"models/{self.model_name}",
                "generationConfig": { 
                    "responseModalities": self.response_modalities 
                }
            }
            if self.enable_asr:
                setup_config["inputAudioTranscription"] = {} 
            if self.enable_tts_transcription:
                setup_config["outputAudioTranscription"] = {} 
            
            if self.system_instruction_text: 
                setup_config["systemInstruction"] = {"parts": [{"text": self.system_instruction_text}]}
            
            if self.tools_config: 
                setup_config["tools"] = self.tools_config
                logging.info(f"Tools configuration being sent: {json.dumps(self.tools_config)}")
            
            if self.voice_name: # Add speechConfig if voice_name is provided
                if "generationConfig" not in setup_config:
                    setup_config["generationConfig"] = {} # Should already exist due to responseModalities
                if "speechConfig" not in setup_config["generationConfig"]: # camelCase
                    setup_config["generationConfig"]["speechConfig"] = {}
                setup_config["generationConfig"]["speechConfig"]["voiceConfig"] = { # camelCase
                    "prebuiltVoiceConfig": { # camelCase
                        "voiceName": self.voice_name # camelCase
                    }
                }
                logging.info(f"TTS voice '{self.voice_name}' configured for the session.")


            # Allow overriding the entire setup_config if custom_setup_payload is provided
            # Note: custom_setup_payload would override the dynamically constructed setup_config including systemInstruction and tools
            final_setup_content_dict = custom_setup_payload if custom_setup_payload else setup_config
            initial_setup_message_dict = {"setup": final_setup_content_dict} 
            
            await self.ws_connection.send(json.dumps(initial_setup_message_dict))
            logging.info(f"Sent BidiGenerateContentSetup message: {json.dumps(initial_setup_message_dict)}")
            
            # Wait for BidiGenerateContentSetupComplete message
            response_str = await asyncio.wait_for(self.ws_connection.recv(), timeout=setup_timeout)
            response_json = json.loads(response_str)

            if response_json.get("setupComplete"): # camelCase
                logging.info(f"Received BidiGenerateContentSetupComplete: {response_json}")
                return True
            else:
                logging.error(f"Did not receive SetupComplete or received an error in setup. Response: {response_json}")
                await self.close() # Close connection if setup failed
                return False

        except asyncio.TimeoutError:
            logging.error(f"Timeout during WebSocket connect or setup phase (connect_timeout={connect_timeout}s, setup_timeout={setup_timeout}s).")
        except websockets.exceptions.InvalidURI:
            logging.error(f"Invalid WebSocket URI: {self.websocket_url}")
        except websockets.exceptions.ConnectionClosedError as e: # More specific
            logging.error(f"WebSocket connection closed with error during connect/setup: {e}")
        except websockets.exceptions.SecurityError as e:
            logging.error(f"WebSocket security error (e.g., SSL/TLS issue) during connect/setup: {e}")
        except ConnectionRefusedError as e:
            logging.error(f"WebSocket connection refused during connect/setup: {e}")
        except Exception as e:
            logging.error(f"Failed to connect or setup Google Live API: {e}")
        
        # Ensure connection is closed and marked as None on any failure path
        if self.ws_connection and not self.ws_connection.closed:
            await self.close() # This will also set self.ws_connection to None
        else:
            self.ws_connection = None 
        return False

    async def send_text(self, text: str, turn_complete: bool = True):
        """
        Sends a text message using BidiGenerateContentClientContent structure.
        """
        if not self.ws_connection or self.ws_connection.closed:
            logging.error("WebSocket connection is not established or closed. Cannot send text.")
            return

        try:
            message = {
                "clientContent": { # camelCase
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                    "turnComplete": turn_complete # camelCase
                }
            }
            await self.ws_connection.send(json.dumps(message))
            logging.info(f"Sent text message: {text}")
        except websockets.exceptions.ConnectionClosed as e: # More specific
            logging.error(f"Error sending text message: Connection closed. {e}")
            await self.close() 
        except Exception as e:
            logging.error(f"Error sending text message: {e}")

    async def send_audio_chunk(self, audio_chunk: bytes, mime_type: str = "audio/pcm;rate=16000"):
        """
        Sends an audio chunk using BidiGenerateContentRealtimeInput structure.
        Audio data is base64 encoded.
        """
        if not self.ws_connection or self.ws_connection.closed:
            logging.error("WebSocket connection is not established or closed. Cannot send audio.")
            return

        try:
            encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
            message = {
                "realtimeInput": { # camelCase
                    "audio": {
                        "mime_type": mime_type, # camelCase for consistency, though API might be flexible
                        "data": encoded_audio
                    }
                }
            }
            await self.ws_connection.send(json.dumps(message))
            # Use logging.debug for frequent messages like audio chunks if needed
            # logging.debug(f"Sent audio chunk ({len(audio_chunk)} bytes, type {mime_type}).") 
        except websockets.exceptions.ConnectionClosed as e: # More specific
            logging.error(f"Error sending audio chunk: Connection closed. {e}")
            await self.close()
        except Exception as e:
            logging.error(f"Error sending audio chunk: {e}")

    async def send_activity_end(self):
        """
        Sends an activity end marker to the Google Live API.
        This signals that no more realtimeInput (e.g., audio) will be sent for the current activity.
        """
        if not self.ws_connection or self.ws_connection.closed:
            logging.error("WebSocket connection is not established or closed. Cannot send activity end.")
            return
        try:
            message = {"realtimeInput": {"activityEnd": {}}} # activityEnd is an empty object
            await self.ws_connection.send(json.dumps(message))
            logging.info("Sent activity end marker.")
        except websockets.exceptions.ConnectionClosed as e:
            logging.error(f"Error sending activity end marker: Connection closed. {e}")
            await self.close()
        except Exception as e:
            logging.error(f"Error sending activity end marker: {e}")

    async def receive_messages(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Continuously listens for messages from the API and yields them as parsed JSON.
        """
        if not self.ws_connection: # Check if None (not just closed)
            logging.error("WebSocket connection is not established. Cannot receive messages.")
            raise StopAsyncIteration # Signal that iteration cannot proceed

        if self.ws_connection.closed:
            logging.warning("WebSocket connection is closed. Cannot receive messages.")
            raise StopAsyncIteration

        try:
            async for message_str in self.ws_connection:
                try:
                    message_json = json.loads(message_str)
                    # logging.debug(f"Received raw message: {message_json}") 
                    yield message_json
                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON from message: {message_str}")
        except websockets.exceptions.ConnectionClosed as e: # More specific
            logging.info(f"WebSocket connection closed while receiving messages: {e.reason} (code: {e.code})")
        except Exception as e:
            logging.error(f"Error during message receiving loop: {type(e).__name__} - {e}")
        finally:
            logging.info("Receive loop ended.")
            # Connection state will be handled by the caller or by subsequent calls trying to use it.
            # If the loop ends due to connection closure, self.ws_connection.closed will be True.

    @staticmethod
    def process_server_content_parts(message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a server message to extract relevant data from 'serverContent'.
        Specifically looks for modelTurn parts (text, audio) and input/output transcriptions.

        Args:
            message: The raw message dictionary received from the WebSocket.

        Returns:
            A dictionary containing extracted data, e.g.:
            {
                "llm_text": "...",
                "tts_audio_bytes": b"...", (raw audio bytes)
                "tts_audio_mime_type": "audio/mpeg" or "audio/pcm",
                "input_transcription_text": "...",
                "input_transcription_is_final": False,
                "output_transcription_text": "...",
                "turn_complete": False, (model's turn complete flag)
                "generation_complete": False,
                "error": "..." (if any error in serverContent)
            }
        """
        processed_data = {
            "llm_text": None,
            "tts_audio_bytes": None,
            "tts_audio_mime_type": None,
            "input_transcription_text": None,
            "input_transcription_is_final": False,
            "output_transcription_text": None,
                "turn_complete": False, # model's turn completion
                "generation_complete": False, # overall generation completion
                "tool_calls": None, # For function calling
            "error": None
        }

        server_content = message.get("serverContent")
        if not server_content:
            return processed_data 

        if "error" in server_content:
            processed_data["error"] = server_content["error"]
            logging.error(f"Server message contained an error: {server_content['error']}")
            # Do not return immediately, other parts might still be present (e.g. partial text before error)
        
        # Process modelTurn for LLM text, TTS audio, and Tool Calls
        model_turn = server_content.get("modelTurn")
        if model_turn and "parts" in model_turn:
            accumulated_text = []
            for part in model_turn["parts"]:
                if "text" in part:
                    accumulated_text.append(part["text"])
                elif "inline_data" in part: 
                    inline_data = part["inline_data"]
                    if "data" in inline_data and "mime_type" in inline_data:
                        try:
                            processed_data["tts_audio_bytes"] = base64.b64decode(inline_data["data"])
                            processed_data["tts_audio_mime_type"] = inline_data["mime_type"]
                        except Exception as e:
                            logging.error(f"Error decoding base64 audio data: {e}")
                elif "toolCall" in part: # camelCase for toolCall
                    if processed_data["tool_calls"] is None:
                        processed_data["tool_calls"] = []
                    # Extract id, name, args. Args is already a dict.
                    tool_call_data = part["toolCall"]
                    processed_data["tool_calls"].append({
                        "id": tool_call_data.get("id"),
                        "name": tool_call_data.get("name"),
                        "args": tool_call_data.get("args", {}) # args should be a dict
                    })
                    logging.info(f"Extracted tool call: ID={tool_call_data.get('id')}, Name={tool_call_data.get('name')}")

            if accumulated_text: # This is the LLM's textual response part
                processed_data["llm_text"] = "".join(accumulated_text)

        # Process inputTranscription for ASR results (remains unchanged)
        input_transcription = server_content.get("inputTranscription")
        if input_transcription:
            processed_data["input_transcription_text"] = input_transcription.get("text")
            processed_data["input_transcription_is_final"] = input_transcription.get("isFinal", False)
            if input_transcription.get("finalText"): # Prefer finalText if available and isFinal
                 processed_data["input_transcription_text"] = input_transcription["finalText"]


        # Process outputTranscription for TTS transcription (if enabled)
        output_transcription = server_content.get("outputTranscription")
        if output_transcription:
            processed_data["output_transcription_text"] = output_transcription.get("text")

        # Check for turn/generation completion flags
        if server_content.get("turnComplete", False): # Model's turn complete
            processed_data["turn_complete"] = True
        if server_content.get("generationComplete", False):
            processed_data["generation_complete"] = True
            
        return processed_data

    async def close(self):
        """
        Gracefully closes the WebSocket connection.
        """
        if self.ws_connection and not self.ws_connection.closed:
            try:
                await self.ws_connection.close()
                logging.info("WebSocket connection successfully closed.")
            except websockets.exceptions.ConnectionClosed: # Catch if already closed during operation
                logging.info("WebSocket connection was already closed.")
            except Exception as e:
                logging.error(f"Error closing WebSocket connection: {e}")
        else:
            logging.info("WebSocket connection was not open or already marked closed.")
        self.ws_connection = None # Crucial: ensure it's None after closing attempt

    async def send_text(self, turns: list[dict], turn_complete: bool = True): # Signature updated
        """
        Sends a text message using BidiGenerateContentClientContent structure.
        'turns' is a list of turn objects e.g. [{"role": "user", "parts": [{"text": "..."}]}]
        """
        if not self.ws_connection or self.ws_connection.closed:
            logging.error("WebSocket connection is not established or closed. Cannot send text.")
            return

        try:
            message = {
                "clientContent": { # camelCase
                    "turns": turns, # Use the provided turns list
                    "turnComplete": turn_complete # camelCase
                }
            }
            await self.ws_connection.send(json.dumps(message))
            # Log only the number of turns or a summary, not potentially large 'turns' content
            logging.info(f"Sent text message with {len(turns)} turn(s). Last turn summary: '{turns[-1]['parts'][0]['text'][:50]}...' if available.")
        except websockets.exceptions.ConnectionClosed as e: # More specific
            logging.error(f"Error sending text message: Connection closed. {e}")
            await self.close() 
        except Exception as e:
            logging.error(f"Error sending text message: {e}")

# Example Usage (for testing purposes, can be removed or adapted later)
async def main_test():
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        logging.error("GOOGLE_GEMINI_API_KEY environment variable not set.")
        return

    # Use the module-level default or an environment override
    live_model_name = os.getenv("GEMINI_LIVE_MODEL_TEST", DEFAULT_GEMINI_LIVE_MODEL) 
    
    # Example: Initialize with ASR enabled, TEXT and AUDIO responses, and a system instruction
    client = GoogleLiveClient(
        api_key=api_key, 
        model_name=live_model_name, 
        response_modalities=["TEXT", "AUDIO"], 
        enable_asr=True,
        enable_tts_transcription=False,
        system_instruction_text="You are a helpful assistant that loves to talk about space."
    )
    
    # No custom_setup_payload for this test, use defaults from __init__
    if await client.connect(): 
        logging.info("Connection and setup successful, listening for responses...")
        
        # Example: Send an initial text message (now using the 'turns' format)
        example_turns = [{"role": "user", "parts": [{"text": "Hello, Gemini Live! Tell me a short story about a brave astronaut."}]}]
        await client.send_text(turns=example_turns)
        
        # Example: Simulate sending a few audio chunks (replace with actual audio source)
        # This is just for testing the send_audio_chunk method's payload structure.
        # In a real application, you'd get audio from a microphone or stream.
        # for i in range(3):
        #     await asyncio.sleep(0.5) # Simulate audio interval
        #     dummy_audio_chunk = os.urandom(1024) # 1KB of dummy audio data
        #     await client.send_audio_chunk(dummy_audio_chunk, mime_type="audio/opus") # Assuming Opus
        #     logging.info(f"Sent dummy audio chunk {i+1}")

        # await client.send_text("What is the weather like today?", turn_complete=True)

        try:
            async for message in client.receive_messages():
                logging.info(f"Received from API: {message}")
                # Add more sophisticated message processing based on actual API responses
                # For example, checking message.get("serverContent", {}).get("candidates") etc.
                if message.get("serverContent", {}).get("error"):
                    logging.error(f"Server returned an error: {message['serverContent']['error']}")
                    break 
                # Add logic here to stop listening if a "final" response is detected, if applicable
                # Or just run for a certain number of messages for a test
                # if some_condition_to_stop:
                #    break

        except asyncio.CancelledError:
            logging.info("Receiver task cancelled.")
        except StopAsyncIteration: # Can happen if receive_messages is called on a bad connection
            logging.warning("Receiver stopped due to connection issue.")
        finally:
            await client.close()
    else:
        logging.error("Failed to connect to Google Live API.")

if __name__ == "__main__":
    # This part is for testing the client directly.
    # To run this test:
    # 1. Ensure GOOGLE_GEMINI_API_KEY is set in your environment.
    # 2. Optionally, set GEMINI_LIVE_MODEL_TEST if you want to test a specific live-capable model.
    # 3. Run the script: python service/webrtc/ai/google_live_api.py
    
    # To enable the test, uncomment the line below:
    # asyncio.run(main_test()) # Ensure this test is updated if send_text signature changes significantly in usage.
    
    logging.info("GoogleLiveClient class updated. Uncomment 'asyncio.run(main_test())' in __main__ to run a connection test.")
