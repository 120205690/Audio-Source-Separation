import os
import numpy as np
import pylab
import librosa
import soundfile as sf
import streamlit as st
from pathlib import Path
from pydub import AudioSegment 
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def sperate_magnitude_phase(data):

    return np.abs(data), np.angle(data)

def combine_magnitdue_phase(magnitudes, phases):

    return magnitudes * np.exp(1.j * phases)

class SVSRNN(object):

    def __init__(self, num_features, num_rnn_layer = 3, num_hidden_units = [256, 256, 256]):

        assert len(num_hidden_units) == num_rnn_layer

        self.num_features = num_features
        self.num_rnn_layer = num_rnn_layer
        self.num_hidden_units = num_hidden_units

        self.gstep = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')
        self.learning_rate = tf.placeholder(tf.float32, shape = [], name = 'learning_rate')
        # The shape of x_mixed, y_src1, y_src2 are [batch_size, n_frames (time), n_frequencies]
        self.x_mixed = tf.placeholder(tf.float32, shape = [None, None, num_features], name = 'x_mixed')
        self.y_src1 = tf.placeholder(tf.float32, shape = [None, None, num_features], name = 'y_src1')
        self.y_src2 = tf.placeholder(tf.float32, shape = [None, None, num_features], name = 'y_src2')

        self.y_pred_src1, self.y_pred_src2 = self.network_initializer()

        self.gamma = 0.001
        self.loss = self.loss_initializer()
        self.optimizer = self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def network(self):

        rnn_layer = [tf.nn.rnn_cell.GRUCell(size) for size in self.num_hidden_units]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layer)
        outputs, state = tf.nn.dynamic_rnn(cell = multi_rnn_cell, inputs = self.x_mixed, dtype = tf.float32)
        y_hat_src1 = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_hat_src1')
        y_hat_src2 = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_hat_src2')
        y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        # Mask with abs
        #y_tilde_src1 = tf.abs(y_hat_src1) / (tf.abs(y_hat_src1) + tf.abs(y_hat_src2) + np.finfo(float).eps) * self.x_mixed
        #y_tilde_src2 = tf.abs(y_hat_src2) / (tf.abs(y_hat_src1) + tf.abs(y_hat_src2) + np.finfo(float).eps) * self.x_mixed
        return y_tilde_src1, y_tilde_src2
        #return y_hat_src1, y_hat_src2
 
    def network_initializer(self):

        with tf.variable_scope('rnn_network') as scope:
            y_pred_src1, y_pred_src2 = self.network()

        return y_pred_src1, y_pred_src2


    def generalized_kl_divergence(self, y, y_hat):

        return tf.reduce_mean(y * tf.log(y / y_hat) - y + y_hat)


    def loss_initializer(self):

        with tf.variable_scope('loss') as scope:

            # Mean Squared Error Loss
            loss = tf.reduce_mean(tf.square(self.y_src1 - self.y_pred_src1) + tf.square(self.y_src2 - self.y_pred_src2), name = 'loss')

        return loss

    def optimizer_initializer(self):

        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss, global_step = self.gstep)

        return optimizer

    def train(self, x, y1, y2, learning_rate):

        #step = self.gstep.eval()

        step = self.sess.run(self.gstep)

        _, train_loss, summaries = self.sess.run([self.optimizer, self.loss], 
            feed_dict = {self.x_mixed: x, self.y_src1: y1, self.y_src2: y2, self.learning_rate: learning_rate})
        return train_loss

    def validate(self, x, y1, y2):

        y1_pred, y2_pred, validate_loss = self.sess.run([self.y_pred_src1, self.y_pred_src2, self.loss], 
            feed_dict = {self.x_mixed: x, self.y_src1: y1, self.y_src2: y2})
        return y1_pred, y2_pred, validate_loss

    def test(self, x):

        y1_pred, y2_pred = self.sess.run([self.y_pred_src1, self.y_pred_src2], feed_dict = {self.x_mixed: x})

        return y1_pred, y2_pred

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)

SAMPLE_RATE = 8192
WINDOW_SIZE = 1024
HOP_LENGTH = 768

# streamlit run app2.py

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
output_directory=rf'output/'

def run(file):

    dir=output_directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # wav_filename=song_filenames[0]
    wav_mono, _ = librosa.load(file, sr = mir1k_sr, mono = True)

    stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
    stft_mono=stft_mono.transpose()

    wav_filename="output/mix.wav"

    wav_filename_base = os.path.basename(wav_filename)
    wav_mono_filepath = os.path.join(dir,f'_mono.wav')
    wav_src1_hat_filepath = os.path.join(dir,f'mel.wav')
    wav_src2_hat_filepath = os.path.join(dir,f'voc.wav')

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
    wav_mix=wav_mono_filepath.replace("_mono",'mix')
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

    st.markdown("<h1 style='text-align: center; color: black;'>Accompaniment </h1>", unsafe_allow_html=True)
    # st.markdown("<h1 style='text-align: center; color: black;'>(We're still working on the accompaniment model) </h1>", unsafe_allow_html=True)
    audio_file = open(wav_src1_hat_filepath, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)

# st.markdown("<h1 style='text-align: center; color: black;'>Vocario</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: black;'>Audio Source Separation</h1>", unsafe_allow_html=True)

mir1k_sr = 16000
n_fft = 1024
hop_length = n_fft // 4
num_rnn_layer = 3
num_hidden_units = [256, 256, 256]
model_directory = 'model'
model_filename = 'svsrnn.ckpt'
model_filepath = os.path.join(model_directory, model_filename)


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
        model = SVSRNN(num_features = n_fft // 2 + 1, num_rnn_layer = num_rnn_layer, num_hidden_units = num_hidden_units)
        model.load(filepath = model_filepath)
        run("extracted.wav")
else:
    ch = st.selectbox('Want to experience more generated music?',
                ('1', '2', '3','4', '5', '6','7', '8', '9','10'),index=0)
    p4=Path("database").iterdir()
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
        st.markdown("<h1 style='text-align: center; color: black;'>Accompaniment </h1>", unsafe_allow_html=True)
        # st.markdown("<h1 style='text-align: center; color: black;'>(We're still working on the accompaniment model) </h1>", unsafe_allow_html=True)
        outpath=folder+'mel.wav'
        audio_file = open(outpath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)