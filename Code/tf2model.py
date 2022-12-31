import tensorflow as tf
import numpy as np
# import tensorflow.compat.v1 as tf1
from tensorflow import keras as k

n_fft=1024
EPOCHS=10
BS=64
num_hidden_units = [256, 256, 256]
num_features = n_fft // 2 + 1
#[batch_size, n_frames (time), n_frequencies]
x_mixed=np.random.rand(64,10,513)
path=r"E:\FinalDS\mix\*.npy"
modelpath=r"C:\Users\Axiomatize\Desktop\Singing_Voice_Separation_RNN-master\BossModel"

def load(mixpath):
    vocpath=tf.strings.regex_replace(mixpath,'mix','voc')
    dtype=16
    voc = tf.io.decode_raw(tf.io.read_file(vocpath), tf.float16)
    remove_len = 1024//dtype
    voc = voc[remove_len:]
    voc=tf.reshape(voc,(-1,1,513))
    voc=voc[:1000]
    voc=tf.reshape(voc,(-1,10,513))

    voc=tf.cast(voc,tf.float32)
    mix=voc;mel=voc
    # mix2=tf.cast(np.random.rand(64,10,513),tf.float32)
    # mix3=tf.cast(np.random.rand(64,10,513),tf.float32)
    return (mix,(voc,mel))

trainset=(
    tf.data.Dataset
    .list_files(path)
    .map(load)
    .flat_map(lambda mix, voc: tf.data.Dataset.zip((
      tf.data.Dataset.from_tensor_slices(mix),
      zip(tf.data.Dataset.from_tensor_slices(voc[0]),tf.data.Dataset.from_tensor_slices(voc[1])))))
    .batch(BS,drop_remainder=True,num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE))

x_mixed=k.Input(shape=(10,513))
rnn_layer = tf.keras.layers.RNN([tf.keras.layers.GRUCell(256),
                                 tf.keras.layers.GRUCell(256),
                                 tf.keras.layers.GRUCell(256)],
                                 return_sequences=True,
                                 return_state=True, 
                                 dtype = tf.float32)
outputs = rnn_layer(x_mixed)
outputs=outputs[0]

y_hat_src1 = tf.keras.layers.Dense (units = num_features,activation ='relu')(outputs)

y_hat_src2 = tf.keras.layers.Dense (units = num_features,activation ='relu')(outputs)

y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * x_mixed
y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * x_mixed

model = k.Model(inputs=x_mixed, outputs=[y_tilde_src1,y_tilde_src2])

adam=tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss=[k.losses.MeanSquaredError(),k.losses.MeanSquaredError()],optimizer=adam)
# model = tf.keras.models.load_model(modelpath)
model.summary()
# H = model.fit(
# 	x=trainset,
# 	epochs=EPOCHS)

# model.save("BossModel")
