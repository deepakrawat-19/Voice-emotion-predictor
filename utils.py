import keras
import librosa
import tensorflow as tf
import numpy as np
from gtts import gTTS
import os
import playsound
import webbrowser

playlist_url={
    'neutral':'',
    'calm':'',
    'happy':'',
    'sad':'',
    'angry':'',
    'fearful':'',
    'disgust':'',
    'surprised':''
}
loaded_model = tf.keras.models.load_model('saved_models/Emotion_Voice_Detection_LSTM_Modelnew.h5')


def speak(text):
    tts=gTTS(text=text,lang="en")
    filename='speak.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

def openSongs(data):
    webbrowser.open(playlist_url[data], new=0, autoraise=True)
    return

def openMovieList():
    pass


def emotionPredict():
    data, sampling_rate = librosa.load('outputvoice.wav')
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=1)
    x = np.expand_dims(x, axis=0)
    predictions = loaded_model.predict_classes(x)
    return predictions

def predict_classes(pred):
    label_conversion = {'0': 'neutral',
                        '1': 'calm',
                        '2': 'happy',
                        '3': 'sad',
                        '4': 'angry',
                        '5': 'fearful',
                        '6': 'disgust',
                        '7': 'surprised'}
    for key, value in label_conversion.items():
        if int(key) == pred:
            label = value
    return label


