import asyncio
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
import torch
from typing import Any, Dict, List
from .runnable_basemodel import Runnable
from ..utils.logger import setup_logger
from pathlib import Path

# Define Hugging Face model names for initial download
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Define the local paths where we will save the models
# Hardcode the model download directory to server/models/faster-whisper/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
LLM_LOCAL_PATH = os.path.join(project_root, 'models', 'llama-tiny')
EMBEDDING_LOCAL_PATH = os.path.join(project_root, 'models', 'miniLM')


# Initialize logger for the LLM module
llm_logger = setup_logger('LLMModule', level=logging.INFO)

class LLMModule(Runnable):
    """
    A runnable module for generating text with a local Large Language Model.
    This version saves and loads models from specific local directories.
    """

    def __init__(self):
        """
        Initializes the LLMModule. The models will be loaded lazily on the first
        invocation to save on startup time.
        """
        self.llm_model = None
        self.tokenizer = None
        self.embedding_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        llm_logger.info(f"LLMModule initialized. Using device: {self.device}")

    async def _load_models(self):
        """
        Asynchronously loads the LLM, tokenizer, and embedding models.
        It first checks for the models in the local paths. If they are not found,
        it downloads them from Hugging Face and saves them to the specified locations.
        """
        # Create model directories if they don't exist
        os.makedirs(LLM_LOCAL_PATH, exist_ok=True)
        os.makedirs(EMBEDDING_LOCAL_PATH, exist_ok=True)

        if self.llm_model is None or self.tokenizer is None:
            llm_logger.info("Checking for local LLM model and tokenizer...")
            
            # The transformers library handles downloading and caching automatically
            # when you provide a path that doesn't exist yet, it will download
            # the model to that path and save it for future use.
            llm_logger.info(f"Loading LLM model '{LLM_MODEL_NAME}' to path '{LLM_LOCAL_PATH}'...")
            try:
                # Load the tokenizer, which will download it if the directory is empty
                self.tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained,
                    LLM_MODEL_NAME,
                    cache_dir=LLM_LOCAL_PATH,
                )

                # Load the model config
                config = await asyncio.to_thread(
                    AutoConfig.from_pretrained,
                    LLM_MODEL_NAME,
                    cache_dir=LLM_LOCAL_PATH,
                )
                config.max_position_embeddings = 2048

                # Load the model itself
                self.llm_model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    LLM_MODEL_NAME,
                    config=config,
                    cache_dir=LLM_LOCAL_PATH,
                )
                self.llm_model.to(self.device)
                self.llm_model.eval()
                llm_logger.info("LLM model and tokenizer loaded successfully.")
            except Exception as e:
                llm_logger.error(f"Failed to load LLM model: {e}", exc_info=True)
                raise

        if self.embedding_model is None:
            llm_logger.info("Checking for local embedding model...")
            llm_logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' to path '{EMBEDDING_LOCAL_PATH}'...")
            try:
                # The SentenceTransformer library also handles downloading and caching
                self.embedding_model = await asyncio.to_thread(
                    SentenceTransformer,
                    EMBEDDING_MODEL_NAME,
                    device=self.device,
                    cache_folder=EMBEDDING_LOCAL_PATH
                )
                llm_logger.info("Embedding model loaded successfully.")
            except Exception as e:
                llm_logger.error(f"Failed to load embedding model: {e}", exc_info=True)
                pass

    async def invoke(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Generates a text response from the LLM based on a user prompt.
        """
        await self._load_models()
        print("-" * 30)
        print("invoking LLMModule with chat history:")
        if not chat_history:
            llm_logger.warning("Received an empty user_prompt.")
            return "Please provide some text to continue."

        llm_logger.info(f"Generating LLM response for user prompt: '{chat_history}'")
        try:
            # Prepare the prompt using a chat template with system, user, and assistant roles
            # The system message is added to the beginning of the chat history
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Provide relevant answers to the given questions in only 1 line no more than 1 line. If you don't know the answer, just say 'I don't know the answer'.",
                    "role": "user",
                    "content": "What is the capital of France?",
                    "role": "assistant",
                    "content": "The capital of France is Paris."
                }
            ]
            messages.extend(chat_history)
            
            # Apply the tokenizer's chat template to format the messages
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            output_tokens = await asyncio.to_thread(
                self.llm_model.generate,
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

            response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            response = response.split("### Response:")[-1].strip()
            ## add response to chat history but we hav to convert chat_history to a list of messages
            llm_logger.info(f"LLM response generated: '{response}'")
            return response
        except Exception as e:
            llm_logger.error(f"An error occurred during LLM text generation: {e}", exc_info=True)
            return "I apologize, but I encountered an error while processing your request. Please try again."

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generates an embedding vector for the given text.
        """
        if self.embedding_model is None:
            llm_logger.warning("Embedding model is not loaded. Cannot generate embeddings.")
            return []

        try:
            llm_logger.info(f"Generating embedding for text: '{text[:50]}...'")
            embedding_np = await asyncio.to_thread(self.embedding_model.encode, text)
            embedding = embedding_np.tolist()
            llm_logger.debug("Embedding generated successfully.")
            return embedding
        except Exception as e:
            llm_logger.error(f"An error occurred during embedding generation: {e}", exc_info=True)
            return []
