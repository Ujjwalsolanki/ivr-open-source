from abc import ABC, abstractmethod
from typing import Any

class Runnable(ABC):
    """
    Abstract Base Class for all runnable components in the IVR system.
    This class defines the interface for components that can be invoked
    as part of a processing chain, similar to LangChain runnables.
    """

    @abstractmethod
    async def invoke(self, *args, **kwargs) -> Any:
        """
        Abstract method to be implemented by all concrete runnable classes.
        This method defines the core logic of the runnable component.

        It should be an asynchronous method, allowing for non-blocking I/O
        operations (e.g., API calls, model inference).

        Args:
            *args: Positional arguments specific to the runnable's operation.
            **kwargs: Keyword arguments specific to the runnable's operation.

        Returns:
            Any: The result of the runnable's operation.
        """
        pass

# Example of how a concrete module would inherit from Runnable (for illustration)
# This part is commented out as it's just an example and not part of the base model file.
"""
# server/modules/stt_module.py (Example)
from .runnable_basemodel import Runnable

class STTModule(Runnable):
    async def invoke(self, audio_data: bytes) -> str:
        # Implement Faster Whisper transcription logic here
        print("Transcribing audio...")
        # Placeholder for actual transcription
        await asyncio.sleep(1) # Simulate async operation
        return "This is a transcribed message."

# server/modules/llm_module.py (Example)
from .runnable_basemodel import Runnable

class LLMModule(Runnable):
    async def invoke(self, text_input: str, conversation_history: list = None) -> str:
        # Implement Llama Tiny inference logic here
        print(f"Generating response for: {text_input}")
        # Placeholder for actual LLM generation
        await asyncio.sleep(2) # Simulate async operation
        return "Hello! How can I help you today?"
"""
