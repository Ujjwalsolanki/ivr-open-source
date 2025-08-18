# 🗣️ Voice Chatbot: An End-to-End AI Conversation Platform

This project is a complete, real-time voice chatbot that demonstrates proficiency in building production-ready conversational AI applications from the ground up. It seamlessly integrates advanced open-source models for speech-to-text, text-to-speech, and natural language understanding.

-----

### 🚀 Key Features & Technologies

  * **🎙️ Speech-to-Text (STT):** Powered by **Faster Whisper**, a highly optimized and efficient version of OpenAI's Whisper model. This enables lightning-fast and accurate transcription of spoken language.
  * **🧠 Large Language Model (LLM):** At its core, the chatbot uses **LLaMA-tiny**, a powerful transformer-based model. This showcases expertise in working with modern AI architectures to generate intelligent and context-aware responses.
  * **🗣️ Text-to-Speech (TTS):** The chatbot's voice is generated using **Coqui-TTS**, a state-of-the-art text-to-speech library known for producing natural, human-like voice synthesis.
  * **🔗 Vector Embeddings:** The system utilizes **MiniLM** to create high-quality vector embeddings. This is a critical component for building scalable retrieval-augmented generation (RAG) pipelines, ensuring the model's responses are grounded in relevant information.
  * **🌐 Full-Stack Integration:** The project features a robust **FastAPI** backend and a responsive frontend built with pure **HTML, CSS, and JavaScript**, demonstrating a comprehensive understanding of web application development.

-----

### 💻 Getting Started

Follow these steps to set up and run the AI Voice Chatbot locally.

1.  **Create a Virtual Environment:**
    Start by creating a dedicated virtual environment to manage project dependencies.

    ```bash
    python3 -m venv venv
    ```

2.  **Activate the Virtual Environment:**
    Activate the environment before installing packages.

    ```bash
    # On macOS/Linux
    source venv/bin/activate
    # On Windows
    venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install all required libraries using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    Launch the main application by running the `main.py` file. This will start the FastAPI server and host the frontend.

    ```bash
    python3 main.py
    ```

After running the command, open your web browser and navigate to `http://127.0.0.1:8000` to start interacting with the chatbot.