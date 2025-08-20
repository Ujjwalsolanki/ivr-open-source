# server/modules/stt_module.py

import asyncio
import os
import logging
import io
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly # For audio resampling
from faster_whisper import WhisperModel
from typing import Any

# Import the Runnable base class and the setup_logger utility
from .runnable_basemodel import Runnable
from ..utils.logger import setup_logger

# Initialize logger for the STT module
# This logger will output to console and the ivr_system.log file
stt_logger = setup_logger('STTModule', level=logging.INFO) # Set to logging.DEBUG for more verbose output during development

class STTModule(Runnable):
    """
    A runnable module for Speech-to-Text transcription using Faster Whisper.
    It inherits from the Runnable abstract base class and implements the
    asynchronous invoke method.
    """
    def __init__(self):
        """
        Initializes the STTModule and prepares to load the Faster Whisper model.

        Args:
            model_size (str): The size of the Whisper model to use.
                              Options include "tiny", "base", "small", "medium", "large".
                              For better performance, consider "base" or "small" if resources allow.
            device (str): The device to run the model on. Can be "cpu" or "cuda" (for GPU).
                          Ensure you have the correct PyTorch installation for CUDA if using "cuda".
            compute_type (str): The compute type for the model.
                                Options include "int8", "float16", "float32".
                                "int8" is generally faster and uses less memory but might have
                                a slight impact on accuracy. "float16" is a good balance for GPUs.
        """
        self.model_size = "tiny"
        self.device = "cpu"
        self.compute_type = "int8"

        # Hardcode the model download directory to server/models/faster-whisper/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        self.model_download_dir = os.path.join(project_root, 'models', 'faster-whisper')

        self.model = None  # Model will be loaded lazily on first invocation
        stt_logger.info(
            f"STTModule initialized with model_size='{self.model_size}', device='{self.device}', "
            f"compute_type='{self.compute_type}'. Model will be loaded from/downloaded to: '{self.model_download_dir}'. "
            "Model will be loaded on first use."
        )

    async def _load_model(self):
        """
        Asynchronously loads the Faster Whisper model if it hasn't been loaded yet.
        This method ensures the model is only loaded once, optimizing resource usage.
        """
        if self.model is None:
            stt_logger.info(
                f"Loading Faster Whisper model: '{self.model_size}' on '{self.device}' "
                f"with '{self.compute_type}' compute type from directory: '{self.model_download_dir}'..."
            )
            try:
                # The WhisperModel constructor handles downloading the model to the specified
                # download_dir if it's not already present.
                # self.model = await asyncio.to_thread(
                #     WhisperModel("tiny",
                #                device="cpu",
                #                compute_type="int8",
                #                num_workers=10,
                #                cpu_threads=4,
                #                download_root=self.model_download_dir))
                self.model = await asyncio.to_thread(
                    WhisperModel,
                    self.model_size,
                    device=self.device, 
                    compute_type=self.compute_type,
                    download_root=self.model_download_dir,
                    # Added performance-tuning parameters
                    num_workers=10,
                    cpu_threads=4
                )
                stt_logger.info("Faster Whisper model loaded successfully.")
            except Exception as e:
                stt_logger.error(f"Failed to load Faster Whisper model: {e}", exc_info=True)
                raise # Re-raise the exception to propagate the error

    async def invoke(self, audio_bytes: bytes) -> str:
        """
        Transcribes audio from a bytes object using the loaded Faster Whisper model.
        This method is designed to be asynchronous and processes audio directly in memory.

        Args:
            audio_bytes (bytes): The raw audio data as a bytes object. This typically
                                 comes directly from the frontend (e.g., a WebM or OGG blob).

        Returns:
            str: The concatenated transcribed text from the audio data.

        Raises:
            ValueError: If the audio data cannot be processed (e.g., unsupported format).
            Exception: Catches and logs any other exceptions that occur during
                       the transcription process, then re-raises them.
        """
        await self._load_model() # Ensure the model is loaded before transcription

        if not audio_bytes:
            stt_logger.warning("Received empty audio_bytes for transcription.")
            return ""

        stt_logger.info(f"Starting transcription for audio stream (bytes, length: {len(audio_bytes)}).")
        transcribed_text = ""
        try:
            # Create a BytesIO object from the incoming bytes
            audio_stream = io.BytesIO(audio_bytes)

            # Read audio data using soundfile. This handles various formats (WebM, OGG, WAV etc.)
            # The dtype='float32' argument is crucial to ensure the correct data type for the model.
            audio_data_np, samplerate = sf.read(audio_stream, dtype='float32')

            # Convert to mono if stereo
            if audio_data_np.ndim > 1:
                audio_data_np = np.mean(audio_data_np, axis=1)
                stt_logger.debug("Converted stereo audio to mono.")

            # Resample to 16kHz if necessary, as Faster Whisper prefers this
            if samplerate != 16000:
                stt_logger.info(f"Resampling audio from {samplerate}Hz to 16000Hz.")
                audio_data_np = resample_poly(audio_data_np, 16000, samplerate).astype(np.float32) ## we have to convert to float32 after resampling
                samplerate = 16000 # Update samplerate after resampling
            else:
                stt_logger.debug(f"Audio already at 16kHz sample rate.")

            # Faster Whisper's transcribe method can accept a NumPy array directly.
            # We use asyncio.to_thread to run this synchronous operation in a separate thread,
            # preventing it from blocking the main event loop.
            segments, info = await asyncio.to_thread(
                self.model.transcribe,
                audio_data_np, # Pass the NumPy array directly
                beam_size=5,  # Example: Number of beams in beam search. Can be tuned.
                vad_filter=True # Example: Enable voice activity detection to filter out silence
            )

            stt_logger.debug(f"Detected language: {info.language} with probability {info.language_probability:.4f}")

            # Iterate over the segments and concatenate the text
            for segment in segments:
                transcribed_text += segment.text + " "
                stt_logger.debug(f"Segment: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

            transcribed_text = transcribed_text.strip() # Remove leading/trailing whitespace
            stt_logger.info(f"Transcription complete. Result: '{transcribed_text}'")
            return transcribed_text
        except Exception as e:
            stt_logger.error(f"An error occurred during Faster Whisper transcription from bytes: {e}", exc_info=True)
            raise # Re-raise the exception after logging
