import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from IPython.display import display, Audio
import tempfile
import joblib

BATCH_SIZE = 128
SHUFFLE_SEED = 36
SAMPLING_RATE = 16000
SCALE = 0.3
display_example = 1
labels = joblib.load('labels')
noises = joblib.load('noises')
class_names = joblib.load('class_names')

def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)
        audio = audio + noise * prop * scale

    return audio

def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

def path_to_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio

model = load_model("songDetect.keras")

path = []
st.title("Audio Signal Processing")
uploded_file = st.file_uploader("Select audio file...",type='wav',)
if uploded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploded_file.read())
        tmp_file_path = tmp_file.name
    #st.write(f"File saved to: {tmp_file_path}")
    audio_file = open(tmp_file_path,'rb')
    audio_bytes = audio_file.read()
    path.append(tmp_file_path)
    #st.write(type(uploded_file.name))
    with open("uploaded_audio.wav", "wb") as f:
       f.write(audio_bytes)
    path_ds = tf.data.Dataset.from_tensor_slices(path)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    test_ds = tf.data.Dataset.zip((audio_ds, label_ds))
    test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
    )
    test_ds = test_ds.map(lambda x, y: (add_noise(x, noises, scale=SCALE), y))
    for audios, labels in test_ds.take(1):
        ffts = audio_to_fft(audios)
        y_pred = model.predict(ffts)
        rnd = np.random.randint(0, 1, display_example)
        audios = audios.numpy()[rnd, :, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]
        
        for index in range(display_example):
            if class_names[y_pred[index]] == "BackGroundMusic":
                text = class_names[labels[index]]
            else:
                text = class_names[labels[index]]
            color = "green" if labels[index] == y_pred[index] else "red"
            #display(Audio(audios[index, :, :].squeeze(), rate=SAMPLING_RATE))'''
            st.markdown(
            f"""
        <p>
            <span style="font-size:15px;">Song Type: </span><span style="color:{color}; font-size:15px;">{text}</span>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <span style="font-size:15px;">Predicted:  </span><span style="color:{color}; font-size:15px;">{class_names[y_pred[index]]}</span>
        </p>
        """,
        unsafe_allow_html=True
        )
    st.audio(audio_bytes,format='audio/wav')