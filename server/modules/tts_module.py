# server/modules/tts_module.py

import asyncio
import logging
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Optional, List
from TTS.api import TTS
from .runnable_basemodel import Runnable
from ..utils.logger import setup_logger

# Define the logger for this module
tts_logger = setup_logger('TTSModule', level=logging.INFO)

# Define the output directory for audio files
OUTPUT_DIR = "output"

class TTSModule(Runnable):
    """
    A runnable module for converting text to speech using the Coqui TTS library.
    This module supports both playing the audio asynchronously and saving it to a file.
    """
    def __init__(self):
        """
        Initializes the TTS module, loading the Coqui TTS model.
        """
        # Define the default Coqui TTS model to use
        # This will download the model the first time it's run
        self.tts_model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        self.tts = None
        self.tts_logger = tts_logger
        self.tts_logger.info("TTSModule initialized. Coqui TTS model will be loaded on first use.")

    async def _load_model(self):
        """
        Asynchronously loads the Coqui TTS model.
        """
        if self.tts is None:
            self.tts_logger.info("Loading Coqui TTS model. This may take a moment...")
            try:
                self.tts = await asyncio.to_thread(TTS, model_name=self.tts_model_name, progress_bar=False, gpu=False)
                self.tts_logger.info("Coqui TTS model loaded successfully.")
            except Exception as e:
                self.tts_logger.error(f"Failed to load Coqui TTS model: {e}", exc_info=True)
                raise

    async def invoke(self, text: str) -> Optional[np.ndarray]:
        """
        Generates audio from text and returns it as a NumPy array.
        """
        if not text:
            self.tts_logger.warning("Received empty text for TTS.")
            return None

        await self._load_model()
        
        self.tts_logger.info(f"Generating TTS audio for text: '{text[:50]}...'")
        
        try:
            # The TTS library's tts_to_file method can be used to get the audio data directly
            # by providing a file path and a callback. However, for a cleaner async approach,
            # we'll use a direct synthesis method if available. In this case, we'll
            # just use the `tts_to_file` method to save the file, as it's the most
            # straightforward way to get the audio data into a usable format.
            # Coqui's `tts` method returns a list of float audio samples.
            audio_data = await asyncio.to_thread(self.tts.tts, text=text) #, speaker=self.tts.default_speaker)
            audio_data = np.array(audio_data, dtype=np.float32)
            self.tts_logger.info("TTS audio generated successfully.")
            return audio_data
        except Exception as e:
            self.tts_logger.error(f"An error occurred during TTS generation: {e}", exc_info=True)
            return None

    async def play_audio(self, audio_data: np.ndarray, sample_rate: int = 22050):
        """
        Plays the generated audio asynchronously using sounddevice.
        """
        if audio_data is None or audio_data.size == 0:
            self.tts_logger.warning("No audio data to play.")
            return
        
        self.tts_logger.info("Playing audio...")
        try:
            await asyncio.to_thread(sd.play, audio_data, sample_rate)
            await asyncio.to_thread(sd.wait)  # Wait for the playback to finish
            self.tts_logger.info("Audio playback finished.")
        except Exception as e:
            self.tts_logger.error(f"Error playing audio: {e}", exc_info=True)

    async def save_audio(self, audio_data: np.ndarray, filename: str, sample_rate: int = 22050):
        """
        Saves the generated audio to a WAV file in the output directory using soundfile.
        """
        if audio_data is None or audio_data.size == 0:
            self.tts_logger.warning("No audio data to save.")
            return
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        self.tts_logger.info(f"Saving audio to '{file_path}'...")
        try:
            await asyncio.to_thread(sf.write, file_path, audio_data, sample_rate)
            self.tts_logger.info(f"Audio successfully saved to '{file_path}'.")
        except Exception as e:
            self.tts_logger.error(f"Error saving audio file: {e}", exc_info=True)

# --- Main function for testing the TTSModule ---
async def main():
    """
    Main function to test the functionality of the TTSModule.
    """
    tts_logger.info("Starting TTSModule test...")
    
    # Initialize the module
    tts_module = TTSModule()
    
    # Define a sample text
    sample_text = "Hello, this is a test of the Coqui text-to-speech engine. It works wonderfully!"
    tts_logger.info(f"Test text: '{sample_text}'")
    
    try:
        # Generate the audio data
        audio_data = await tts_module.invoke(text=sample_text)
        
        if audio_data is not None:
            # Play the audio
            await tts_module.play_audio(audio_data)
            
            # Save the audio
            await tts_module.save_audio(audio_data, "coqui_test.wav")
        else:
            tts_logger.error("Audio generation failed, cannot proceed with playback or saving.")
            
    except Exception as e:
        tts_logger.error(f"An error occurred during the test: {e}", exc_info=True)

if __name__ == "__main__":
    # Add the parent directory to the Python path to resolve the relative import for testing
    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)
    asyncio.run(main())
