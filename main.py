import base64
import io
import os
import re
from typing import Optional
import numpy as np
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, Form 
import uuid
from typing import Dict, List
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import asyncio
from scipy.io.wavfile import write as write_wav


from server.modules.llm_module import LLMModule
from server.modules.stt_module import STTModule
from server.modules.tts_module import TTSModule
from server.utils import logger

# Initialize the FastAPI app
app = FastAPI()

logger = logger.setup_logger("main")

app.mount("/public", StaticFiles(directory="./public"), name="public")
# # Add CORS middleware to allow all origins, methods, and headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Create an instance of the STTModule
stt_module = STTModule()
llm_module = LLMModule()
tts_module = TTSModule()



# In-memory storage for chat sessions. This will reset on server restart.
# In a production environment, you would use a database like Firestore for persistence.
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

transcribed_text: Optional[str] = None

session_id: Optional[str] = None
if session_id is None:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = []


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "./public/index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/stt")
async def process_audio(
    audio: UploadFile = File(...)):
    """
    Receives an audio file, transcribes it, and then generates a response from the LLM.
    It manages the chat history for each session.
    """
    try:
        # Read the uploaded audio file directly into a bytes object
        audio_bytes = await audio.read()

        # Step 1: Transcribe the audio bytes using the STTModule
        global transcribed_text 
        transcribed_text = await stt_module.invoke(audio_bytes)
        
        if not transcribed_text:
            llm_response = "I'm sorry, I couldn't hear you. Please try again."
            chat_sessions[session_id].append({"role": "assistant", "content": llm_response})
            return {"transcription": transcribed_text, "session_id": session_id}

        # Append the new user message to the chat history
        chat_sessions[session_id].append({"role": "user", "content": transcribed_text})

        # # Step 2: Pass the transcribed text and the full chat history to the LLMModule
        # llm_response = await llm_module.invoke(chat_history=chat_sessions[session_id])
        # print("-"* 20)
        # print(f"LLM Response: {llm_response}")

        # # Step 3: Append the LLM's response to the chat history
        # chat_sessions[session_id].append({"role": "assistant", "content": llm_response})
        # print("-"* 20)
        # print(f"chat history: {chat_sessions[session_id]}")

        # # Split the string by the <|assistant|> tag and take the last element
        # text_for_tts = llm_response.split('<|assistant|>')[-1].strip()

        # # text_for_tts = extract_answer_from_assistant_response(text_for_tts)
        # print("-"*30)
        # print(f"Text to pass to TTS module: '{text_for_tts}'")

        # # Step 4: Generate audio from the LLM's response using the TTSModule
        # tts_audio_bytes = await tts_module.invoke(text=text_for_tts)
        # print("-"*30)
        # print("-"*30)
        # print(f"TTS Audio Bytes: {tts_audio_bytes}")

        # audio_data_base64 = ""
        # if tts_audio_bytes is not None and isinstance(tts_audio_bytes, np.ndarray):
        #     try:
        #         # The TTS module returns a NumPy array. Convert it to a WAV file in memory.
        #         buffer = io.BytesIO()
        #         # Assuming a sample rate of 22050 Hz from your Coqui configuration
        #         write_wav(buffer, 22050, tts_audio_bytes.astype(np.float32))
        #         buffer.seek(0)
                
        #         # Now, encode the buffer's contents to base64
        #         audio_data_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        #         print("-"*30)
        #         print("Base64 encoding successful.")
        #         print(f"Generated base64 audio data of length: {len(audio_data_base64)}")
        #         print("-"*30)
        #     except Exception as e:
        #         print(f"Error converting NumPy array to WAV and encoding to base64: {e}")
        # elif tts_audio_bytes and isinstance(tts_audio_bytes, bytes):
        #     # This handles cases where a different TTS module (e.g., Gemini API) returns raw bytes
        #     try:
        #         audio_data_base64 = base64.b64encode(tts_audio_bytes).decode('utf-8')
        #         print("-"*30)
        #         print("Base64 encoding successful.")
        #         print(f"Generated base64 audio data of length: {len(audio_data_base64)}")
        #         print("-"*30)
        #     except Exception as e:
        #         print(f"Error during base64 encoding: {e}")
        # else:
        #     print("TTS module returned invalid or no audio data.")
        
        return {"transcription": transcribed_text, "session_id": session_id}
    except Exception as e:
        # Log the error and return a helpful message
        print(f"Error during transcription or LLM generation: {e}")
        return {"error": "Failed to process audio or generate a response."}
    
@app.post("/llm_and_tts")
async def generate_response():
    """
    Receives an audio file, transcribes it, and then generates a response from the LLM.
    It manages the chat history for each session.
    """
    try:
        # # Read the uploaded audio file directly into a bytes object
        # audio_bytes = await audio.read()

        # # Step 1: Transcribe the audio bytes using the STTModule
        # transcribed_text = await stt_module.invoke(audio_bytes)
        
        # if not transcribed_text:
        #     llm_response = "I'm sorry, I couldn't hear you. Please try again."
        #     chat_sessions[session_id].append({"role": "assistant", "content": llm_response})
        #     return {"transcription": "No speech detected.", "llm_response": llm_response, "session_id": session_id}

        # # Append the new user message to the chat history
        # chat_sessions[session_id].append({"role": "user", "content": transcribed_text})

        # Step 2: Pass the transcribed text and the full chat history to the LLMModule
        llm_response = await llm_module.invoke(chat_history=chat_sessions[session_id])
        print("-"* 20)
        print(f"LLM Response: {llm_response}")

        # Step 3: Append the LLM's response to the chat history
        chat_sessions[session_id].append({"role": "assistant", "content": llm_response})
        print("-"* 20)
        print(f"chat history: {chat_sessions[session_id]}")

        # Split the string by the <|assistant|> tag and take the last element
        text_for_tts = llm_response.split('<|assistant|>')[-1].strip()

        # text_for_tts = extract_answer_from_assistant_response(text_for_tts)
        print("-"*30)
        print(f"Text to pass to TTS module: '{text_for_tts}'")

        # Step 4: Generate audio from the LLM's response using the TTSModule
        tts_audio_bytes = await tts_module.invoke(text=text_for_tts)
        print("-"*30)
        print("-"*30)
        print(f"TTS Audio Bytes: {tts_audio_bytes}")

        audio_data_base64 = ""
        if tts_audio_bytes is not None and isinstance(tts_audio_bytes, np.ndarray):
            try:
                # The TTS module returns a NumPy array. Convert it to a WAV file in memory.
                buffer = io.BytesIO()
                # Assuming a sample rate of 22050 Hz from your Coqui configuration
                write_wav(buffer, 22050, tts_audio_bytes.astype(np.float32))
                buffer.seek(0)
                
                # Now, encode the buffer's contents to base64
                audio_data_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                print("-"*30)
                print("Base64 encoding successful.")
                print(f"Generated base64 audio data of length: {len(audio_data_base64)}")
                print("-"*30)
            except Exception as e:
                print(f"Error converting NumPy array to WAV and encoding to base64: {e}")
        elif tts_audio_bytes and isinstance(tts_audio_bytes, bytes):
            # This handles cases where a different TTS module (e.g., Gemini API) returns raw bytes
            try:
                audio_data_base64 = base64.b64encode(tts_audio_bytes).decode('utf-8')
                print("-"*30)
                print("Base64 encoding successful.")
                print(f"Generated base64 audio data of length: {len(audio_data_base64)}")
                print("-"*30)
            except Exception as e:
                print(f"Error during base64 encoding: {e}")
        else:
            print("TTS module returned invalid or no audio data.")
        
        
        return {"llm_response": text_for_tts, "audio_data": audio_data_base64, "session_id": session_id}
    except Exception as e:
        # Log the error and return a helpful message
        print(f"Error during transcription or LLM generation: {e}")
        return {"error": "Failed to process audio or generate a response."}


# if __name__ == "__main__":
#     # Run the server with Uvicorn.
#     uvicorn.run(app, host="0.0.0.0", port=8000)