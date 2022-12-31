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
from PIL import Image
import time
from preprocess import sperate_magnitude_phase, combine_magnitdue_phase
from model import SVSRNN
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


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

def run(file):

    songs_dir = 'input'
    inpath=songs_dir+'/mix.wav'
    orig, _ = librosa.load(file, sr = mir1k_sr, mono = False)
    # sf.write(inpath,orig,samplerate=mir1k_sr)

    

    song_filenames = []
    for files in os.listdir(songs_dir):
        if (files.endswith('.wav') or files.endswith('.mp3')):
            song_filenames.append(os.path.join(songs_dir, files))

    # Preprocess parameters
    # mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    num_rnn_layer = 3
    num_hidden_units = [256, 256, 256]
    model_directory = 'model'
    model_filename = 'svsrnn.ckpt'
    model_filepath = os.path.join(model_directory, model_filename)
    dir="A"
    output_directory=rf'output/{dir}/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # wav_filename=song_filenames[0]
    wav_mono, _ = librosa.load(file, sr = mir1k_sr, mono = True)

    stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
    stft_mono=stft_mono.transpose()

    model = SVSRNN(num_features = n_fft // 2 + 1, num_rnn_layer = num_rnn_layer, num_hidden_units = num_hidden_units)
    model.load(filepath = model_filepath)
    wav_filename="output/A/mix.wav"

    wav_filename_base = os.path.basename(wav_filename)
    wav_mono_filename = wav_filename_base.split('.')[0] + '_mono.wav'
    wav_src1_hat_filename = wav_filename_base.split('.')[0] + 'mel.wav'
    wav_src2_hat_filename = wav_filename_base.split('.')[0] + 'voc.wav'
    wav_mono_filepath = os.path.join(output_directory, wav_mono_filename)
    wav_src1_hat_filepath = os.path.join(output_directory, wav_src1_hat_filename)
    wav_src2_hat_filepath = os.path.join(output_directory, wav_src2_hat_filename)

    print('Processing %s ...' % wav_filename_base)

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
    wav_mix=wav_mono_filepath.replace("_mono",'')
    file_var.export(wav_mix, format='wav')
    sf.write(wav_src1_hat_filepath, y1_hat, mir1k_sr)
    sf.write(wav_src2_hat_filepath, y2_hat, mir1k_sr)


    col1, col2 = st.columns(2)
    with col1:
        #Original Mix
        # st.title("Original Mix")
        st.markdown("<h1 style='text-align: center; color: black;'>Original Mix</h1>", unsafe_allow_html=True)

        # sf.write(outpath, 
        #             istft(mix_wav_mag_orig * mix_wav_phase,win_length=WINDOW_SIZE,hop_length=HOP_LENGTH),
        #             SAMPLE_RATE)
        audio_file = open(wav_mix, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)

        specpath=wav_mix
        specpath=spectogram_librosa(specpath,0)
        image = Image.open(specpath)
        st.image(image, caption='Original Mix')        

    with col2:
        #Predicted Vocals
        # st.title("Predicted Vocals")
        st.markdown("<h1 style='text-align: center; color: black;'>Vocals</h1>", unsafe_allow_html=True)

        audio_file = open(wav_src2_hat_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)

        specpath=wav_src2_hat_filepath
        specpath=spectogram_librosa(specpath,1)
        image = Image.open(specpath)
        st.image(image, caption='Vocals')

    st.markdown("<h1 style='text-align: center; color: black;'>Melody </h1>", unsafe_allow_html=True)
    # st.markdown("<h1 style='text-align: center; color: black;'>(We're still working on the accompaniment model) </h1>", unsafe_allow_html=True)
    audio_file = open(wav_src1_hat_filepath, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)

# st.markdown("<h1 style='text-align: center; color: black;'>Vocario</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: black;'>Audio Source Separation</h1>", unsafe_allow_html=True)


flag=0
with st.sidebar:
    r = st.radio(
        "Choose wisely",
        ("Upload your own song", "Listen to separated songs"),index=0,
    )
    if(r=="Upload your own song"):
        flag=0
        
    else:
        flag=1
        

if flag==0:
    file = st.file_uploader("Choose a file",type=".wav")
    if file is not None:
        wav=AudioSegment.from_wav(file=file)
        wav.export("extracted.wav",format='wav')
        run("extracted.wav")
else:
    ch = st.selectbox('Want to experience more generated music?',
                ('1', '2', '3','4', '5', '6','7', '8', '9','10'),index=0)
    p4=Path(r"database").iterdir()
    dirsaved=[]
    for i in p4:
        dirsaved.append(str(i))

    if ch:
        folder=dirsaved[(int(ch)-1)]+'\\'
        col11, col22 = st.columns(2)

        with col11:
            st.markdown("<h1 style='text-align: center; color: black;'>Original Mix</h1>", unsafe_allow_html=True)

            outpath=folder+'mix.wav'
            audio_file = open(outpath, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)

            specpath=outpath.replace('wav', 'png')
            image = Image.open(specpath)
            st.image(image, caption='Original Mix')       

        with col22:
            st.markdown("<h1 style='text-align: center; color: black;'>Vocals</h1>", unsafe_allow_html=True)

            outpath=folder+'voc.wav'
            audio_file = open(outpath, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)

            specpath=outpath.replace('wav', 'png')
            image = Image.open(specpath)
            st.image(image, caption='Vocals')