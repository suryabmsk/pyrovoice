from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from transformers import MarianMTModel, MarianTokenizer
import whisper
from TTS.api import TTS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

whisper_model = whisper.load_model("base")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    audio_file = request.files['audio']
    language_code = request.form['language']

    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(audio_path)

    result = whisper_model.transcribe(audio_path)
    transcription = result['text']

    model_name = f'Helsinki-NLP/opus-mt-en-{language_code}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(transcription, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    tts_path = os.path.join(app.config['OUTPUT_FOLDER'], f'tts_{language_code}.wav')
    tts.tts_to_file(text=translation, file_path=tts_path)

    return jsonify({
        'transcription': transcription,
        'translation': translation,
        'tts_path': f'/output_audio/{os.path.basename(tts_path)}'
    })

@app.route('/output_audio/<filename>')
def output_audio(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)