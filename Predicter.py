import os
import numpy as np
import pylab
import librosa
from absl import logging
# from librosa.core import istft, load, stft, magphase
import soundfile as sf
import streamlit as st
from pathlib import Path
from pydub import AudioSegment 
import time
from preprocess import sperate_magnitude_phase, combine_magnitdue_phase
from model import SVSRNN
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob

SAMPLE_RATE = 8192
WINDOW_SIZE = 1024
HOP_LENGTH = 768

# streamlit run app.py

def spectogram_librosa(_wav_file_,flag):
    (sig, rate) = librosa.load(_wav_file_, sr=None, mono=True,  dtype=np.float32)
    pylab.specgram(sig, Fs=rate)
    if flag==0:
        p=_wav_file_.replace('wav','png')
        pylab.savefig(p)
    else:
        p=_wav_file_.replace('wav','png')
        pylab.savefig(p)
    return p

mir1k_sr = 16000
n_fft = 1024
hop_length = n_fft // 4
num_rnn_layer = 3
num_hidden_units = [256, 256, 256]
model_directory = 'model'
model_filename = 'svsrnn.ckpt'
model_filepath = os.path.join(model_directory, model_filename)
model = SVSRNN(num_features = n_fft // 2 + 1, num_rnn_layer = num_rnn_layer, num_hidden_units = num_hidden_units)
model.load(filepath = model_filepath)

# songs_dir = r"E:\DSD100\*\mixture.wav"
songs_dir=r"C:\Users\Axiomatize\Desktop\Wavfile\*.wav"
songs_dir=glob.glob(songs_dir)
c=1
for file in songs_dir:
    
    ext=str(file.split('.')[-1])
    wav_mono, _ = librosa.load(file, sr = mir1k_sr, mono = True)

    stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
    stft_mono=stft_mono.transpose()

    # wav_filename=file.replace(r"C:\Users\Axiomatize\Desktop\FamousEnglishSongs\Small","database")
    wav_filename=file.replace(r"C:\Users\Axiomatize\Desktop\Wavfile","database2")
    # dir=os.path.dirname(wav_filename)
    dir=wav_filename.replace(".wav","")
    os.makedirs(dir,exist_ok=True)

    wav_mono_filepath = os.path.join(dir,f'_mono.{ext}')
    wav_src1_hat_filepath = os.path.join(dir,f'mel.{ext}')
    wav_src2_hat_filepath = os.path.join(dir,f'voc.{ext}')

    print('Processing %s ...' % wav_mono_filepath)

    stft_mono_magnitude, stft_mono_phase = sperate_magnitude_phase(data = stft_mono)
    stft_mono_magnitude = np.array([stft_mono_magnitude])

    y1_pred, y2_pred = model.test(x = stft_mono_magnitude)

    # ISTFT with the phase from mono
    y1_stft_hat = combine_magnitdue_phase(magnitudes = y1_pred[0], phases = stft_mono_phase)
    y2_stft_hat = combine_magnitdue_phase(magnitudes = y2_pred[0], phases = stft_mono_phase)

    y1_stft_hat = y1_stft_hat.transpose()
    y2_stft_hat = y2_stft_hat.transpose()

    y1_hat = librosa.istft(y1_stft_hat, hop_length = hop_length)
    y2_hat = librosa.istft(y2_stft_hat, hop_length = hop_length)

    # sf.write(wav_mono_filepath, wav_mono, mir1k_sr)
    file_var = AudioSegment.from_ogg(file) 
    wav_mix=wav_mono_filepath.replace("_mono",'mix')
    file_var.export(wav_mix, format=f'{ext}')
    sf.write(wav_src1_hat_filepath, y1_hat, mir1k_sr)
    sf.write(wav_src2_hat_filepath, y2_hat, mir1k_sr)

    specpath=wav_mix
    specpath=spectogram_librosa(specpath,0) 
    specpath=wav_src2_hat_filepath
    specpath=spectogram_librosa(specpath,1)
os.system("shutdown /s /t 1")