// public/js/audio_recorder.js

/**
 * Requests microphone access and sets up the MediaRecorder instance.
 * @param {Function} onAudioStopCallback The callback function to execute when recording stops.
 * @returns {Promise<MediaRecorder|null>} A promise that resolves with the MediaRecorder instance or null if access is denied.
 */
export async function setupRecorder(onAudioStopCallback) {
    let mediaRecorder;
    let audioChunks = [];

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = [];
            // Call the provided callback with the audio blob
            onAudioStopCallback(audioBlob);
        };

        return mediaRecorder;
    } catch (error) {
        console.error('Microphone access denied or not available:', error);
        // Do not use alert() as it's blocked in the Canvas environment
        console.warn('Microphone access is required to use voice input.');
        return null;
    }
}
