from utils import speak,emotionPredict,openSongs,openMovieList
from audio_recorder import  record

print("Welcome to the voice emotion recognition system")
speak("Welcome to the voice emotion recognition system")
speak("press 1 to start recording")
speak("press 2 to exit")

openSongs("hi")
option=int(input())
if option==1:
    record()
    speak('Predicting the emotion')
    emotion=emotionPredict()
    speak(emotion)
    print('1. Recommend songs')
    print('2. Recommend movies')
    opt = int(input())
    if opt==1:
        openSongs(emotion)
    else:
        openMovieList(emotion)
else:
    exit()
