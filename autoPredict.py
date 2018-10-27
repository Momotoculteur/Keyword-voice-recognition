#IMPORT
from keras.models import load_model
from PIL import Image
import numpy as np
import time
import pyaudio
import os
from tqdm import tqdm
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt



"""
# Classe permettant de réaliser une prédiction sur une nouvelle donnée
"""


def main():
    """
    # On definit les chemins d'acces au différentes hyper parametre
    """

    # On définit les specs de notre fichier audio de test
    format = pyaudio.paInt16
    channels = 2
    rate = 44100
    chunk = 1024
    recordTime = 2


    modelPath = '.\\trainedModel\\moModel.hdf5'
    soundPath =  '.\\testAudio\\test.wav'
    soundSize = (50,50)
    labels = ['chat', 'chien']


    record(format, channels, rate, chunk, recordTime, soundPath, soundSize, modelPath, labels)


def record(format, channels, rate, chunk, recordTime,soundPath, soundSize, modelPath, labels):
    """
    # Fonction permettant d'enregistrer notre voix pour lancer plusieurs tests à la suite
    :param format: taille de chaque sample
    :param channels: nombre de canaux
    :param rate: taux d'echantillonage
    :param chunk: nombre de frame dans le buffer
    :param recordTime: temps d'enregistrement
    :param soundPath: chemin ou on va record notre audio
    :param soundSize: taille du spectre converti en image depuis notre .wav
    :param modelPath: chemin ou est notre modèle enregistré
    :param labels: nos classes de prédictions
    """

    while True:

        # Chargement modèle, timer et biblio audio
        start = time.time()
        model = load_model(modelPath)
        audio = pyaudio.PyAudio()

        # Début enregistrement
        stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

        os.system('cls')

        # Enregistrement de la voix
        frames = []
        for i in tqdm(range(0, int(rate / chunk * recordTime)), "> Enregistrement... "):
            data = stream.read(chunk)
            frames.append(data)

        # On stop l'enregistrement
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Enregistrement OK")

        # On enregistre l'extrait audio en fichier wav
        waveFile = wave.open(soundPath, 'wb')
        waveFile.setnchannels(channels)
        waveFile.setsampwidth(audio.get_sample_size(format))
        waveFile.setframerate(rate)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        print("\n> Début...")
        start = time.time()

        # On traite le fichier audio vers un tableau
        print("\n    * Traitement des données...", end='', flush=True)
        data = []
        y, sr = librosa.load(soundPath)

        # Calcul du mel spectro selon notre audio
        temp = librosa.feature.melspectrogram(y=y, sr=sr)
        librosa.display.specshow(librosa.power_to_db(temp, ref=np.max))

        # On convertit en image pour retaille la taille de l'image
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        plt.tight_layout(pad=0)
        plt.axis('off')
        img = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        img = img.resize(size=soundSize)
        img = np.asarray(img) / 255.
        data.append(img)
        plt.close()
        data = np.asarray(data)
        print("OK")

        # On reshape notre tableau pour l'entrée de notre convNet
        dimension = data[0].shape
        data = data.astype(np.float32).reshape(data.shape[0], dimension[0], dimension[1], dimension[2])

        # On prédit
        print("\n    * Prédiction du réseau de neurones...", end='', flush=True)
        prediction = model.predict(data)
        print("OK")

        # On recupere le numero de label qui a la plus haut prediction
        maxPredict = np.argmax(prediction)

        # On recupere le mot correspondant à l'indice precedent
        word = labels[maxPredict]
        pred = prediction[0][maxPredict] * 100.
        end = time.time()

        # On affiche les prédictions
        print()
        print('----------')
        print(" Prediction :")
        for i in range(0, len(labels)):
            print('     ' + labels[i] + ' : ' + "{0:.2f}%".format(prediction[0][i] * 100.))

        print()
        print('RESULTAT : ' + word + ' : ' + "{0:.2f}%".format(pred))
        print('temps prediction : ' + "{0:.2f}secs".format(end - start))

        print('----------')
        again = input("\nRecommencer ? (O/N) ")
        if again == 'n':
            return


if __name__ == "__main__":
    """
    # MAIN
    """
    main()