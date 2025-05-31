document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);

    const response = await fetch('/process', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    document.getElementById('transcription').textContent = result.transcription;
    document.getElementById('translated').textContent = result.translation;
    document.getElementById('ttsAudio').src = result.tts_path;
});