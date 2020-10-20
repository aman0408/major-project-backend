from flask import Flask
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask import send_file
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys

UPLOAD_FOLDER = './mp3_files'
ALLOWED_EXTENSIONS = {'wav'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

embedded_model = None
num_generated = 0


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return 'Index'


@app.route('/add-voice-sample', methods=['POST'])
def initialize_model():
    if 'file' not in request.files:
        flash('No file part')
        return 'Invalid Params', 400
    file = request.files['file']
    if file and allowed_file(file.filename):
        print("Preparing the encoder, the synthesizer and the vocoder...")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        preprocessed_wav = encoder.preprocess_wav(filepath)
        original_wav, sampling_rate = librosa.load(str(filepath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file successfully")
        embed = encoder.embed_utterance(preprocessed_wav)
        global embedded_model
        embedded_model = embed
        print("Created the embedding")
        return 'Embedding created successfully', 200
    else:
        return 'Invalid File Type', 400


@app.route('/get-response', methods=['POST'])
def generate_response():
    text = request.form['text']
    texts = [text]
    embeds = [embedded_model]
    synthesizer_obj = Synthesizer(Path("synthesizer/saved_models/logs-pretrained/").joinpath("taco_pretrained"),
                                  low_mem=False)
    specs = synthesizer_obj.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")
    print("Synthesizing the waveform:")
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer_obj.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)
    global num_generated
    filename = "demo_output_%02d.wav" % num_generated
    print(generated_wav.dtype)
    sf.write(filename, generated_wav.astype(np.float32), synthesizer_obj.sample_rate)
    num_generated += 1
    print("\nSaved output as %s\n\n" % filename)
    return send_file(filename, mimetype='audio/mpeg')


if __name__ == '__main__':
    encoder.load_model(Path("encoder/saved_models/pretrained.pt"))

    vocoder.load_model(Path("vocoder/saved_models/pretrained/pretrained.pt"))
    app.run(debug=True)
