// public/js/main.js

import { setupRecorder } from './audio_recorder.js';

document.addEventListener('DOMContentLoaded', async () => {
    const voiceButton = document.getElementById('voice-button');
    const userInput = document.getElementById('user-input');
    const statusIndicator = document.getElementById('status-indicator');
    const chatHistory = document.getElementById('chat-history');

    let isListening = false;
    let mediaRecorder = await setupRecorder(handleServerResponse);

    // This function will handle the server's response.
    // It's passed to `audio_recorder.js` so it can be called there.
    async function handleServerResponse(audioBlob) {
        console.log('Sending audio to server...');
        statusIndicator.textContent = "Processing...";
        statusIndicator.classList.remove('text-green-400');
        statusIndicator.classList.add('text-yellow-400');

        const formData = new FormData();
        formData.append('audio', audioBlob, 'audio.webm');

        try {
            const response = await fetch('/stt', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Server response:', result);
            
            // Append the user's message to the chat history
            appendMessage(result.transcription || 'No speech detected.', 'user');

            // Append the LLM's response to the chat history
            appendMessage(result.llm_response, 'assistant');

            // Play the AI's audio response
            playBase64Audio(result.audio_data, 'audio/wav', 1.5);

        } catch (error) {
            console.error('Error sending audio to server:', error);
            // Do not use alert() as it's blocked in the Canvas environment
            console.warn('Error sending audio. Please check the console for details.');
        } finally {
            statusIndicator.textContent = "Offline";
            statusIndicator.classList.remove('text-yellow-400');
            statusIndicator.classList.add('text-red-400');
        }
    }

    // This is a helper function to display messages in the UI.
    function appendMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('flex', 'items-start', 'mb-2');

        const messageBubble = document.createElement('div');
        messageBubble.textContent = text;
        messageBubble.classList.add('p-3', 'rounded-lg', 'max-w-sm');

        if (sender === 'user') {
            messageDiv.classList.add('justify-end');
            messageBubble.classList.add('bg-[#b4b4e8]', 'text-[#23233c]');
        } else {
            messageDiv.classList.add('justify-start');
            messageBubble.classList.add('bg-gray-800', 'text-white');
        }

        messageDiv.appendChild(messageBubble);
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight; // Auto-scroll to the latest message
    }

    // This is the function to play audio from the server response
    function playBase64Audio(base64AudioData, mimeType, playbackRate = 1.0) {
        if (!base64AudioData || !mimeType) {
            console.error("No audio data or MIME type provided.");
            return;
        }
    
        try {
            const audioUrl = `data:${mimeType};base64,${base64AudioData}`;
            const audio = new Audio(audioUrl);
            audio.playbackRate = playbackRate;
            audio.play().catch(error => {
                console.error("Error playing audio:", error);
            });
        } catch (error) {
            console.error("Failed to create and play audio from base64 data:", error);
        }
    }

    voiceButton.addEventListener('click', () => {
        if (!mediaRecorder) {
            // Do not use alert() as it's blocked in the Canvas environment
            console.warn('MediaRecorder is not initialized. Please refresh the page and allow microphone access.');
            return;
        }

        isListening = !isListening;
        
        if (isListening) {
            // Start recording and update UI
            mediaRecorder.start();
            voiceButton.classList.remove('bg-[#8e8ee8]');
            voiceButton.classList.add('bg-red-500', 'animate-pulse');
            userInput.disabled = true;
            userInput.placeholder = "Listening...";
            statusIndicator.textContent = "Listening";
            statusIndicator.classList.remove('text-red-400');
            statusIndicator.classList.add('text-green-400');
            console.log('Voice input started.');
        } else {
            // Stop recording and update UI
            mediaRecorder.stop();
            voiceButton.classList.remove('bg-red-500', 'animate-pulse');
            voiceButton.classList.add('bg-[#8e8ee8]');
            userInput.disabled = true;
            userInput.placeholder = "Input is disabled. Press the microphone button to speak...";
            statusIndicator.textContent = "Processing...";
            statusIndicator.classList.remove('text-green-400');
            statusIndicator.classList.add('text-yellow-400');
            console.log('Voice input stopped.');
        }
    });
});
