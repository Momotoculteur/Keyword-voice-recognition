# IMPORT
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import matplotlib.pyplot as plt

"""
# Classe permettant de convertir notre dataset de fichiers audios en tableaux Numpy
"""


def launchConversion(pathData, pathNumpy, resizeImg, imgSize):
    """
    # Permet de lancer la conversion des images en tableau numpy
    :param pathData: chemin ou sont le jeu de données
    :param pathNumpy: chemin de destination ou on va sauvegarder nos tableaux
    :param resizeImg: booleen pour permettre de recouper les images ou non
    :param imgSize: on taille les images à une même taille commune
    """

    #Pour chaque classe
    for soundClasse in os.listdir(pathData):
        pathSound = pathData + '\\' + soundClasse
        imgs = []

        #Pour chaque image d'une classe, on la charge, resize et transforme en tableau
        for soundFile in tqdm(os.listdir(pathSound), "Conversion de la classe : '{}'".format(soundClasse)):
            imgSoundPath = pathSound + '\\' + soundFile
            # Chargement de l'image
            y, sr = librosa.load(imgSoundPath)
            # Calcul du mel spectro selon notre audio
            temp = librosa.feature.melspectrogram(y=y, sr=sr)
            librosa.display.specshow(librosa.power_to_db(temp, ref=np.max))
            # On convertit en image pour retaille la taille de l'image
            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            img = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

            if resizeImg == True:
                img = img.resize(size=imgSize)

            # On enleve les axes pour n'avoir que l'image
            plt.tight_layout(pad=0)
            plt.axis('off')
            # Decommenter la ligne si vous voulez un aperçu du spectro en image
            #plt.show()
            data = np.asarray(img, dtype=np.float32)
            imgs.append(data)
            plt.close()

        #Converti les gradients de pixels (allant de 0 à 255) vers des gradients compris entre 0 et 1
        imgs = np.asarray(imgs) / 255.

        #Enregistre une classe entiere en un fichier numpy
        np.save(pathNumpy + '\\ ' + soundClasse + '.npy', imgs)


def main():
    """
    # Fonction main
    """

    pathNumpy = '.\\numpyFiles'
    pathData = '.\\dataset'
    resizeImg = True
    imgSize = (50, 50)
    launchConversion(pathData, pathNumpy, resizeImg, imgSize)


if __name__ == '__main__':
    """
    # MAIN
    """
    main()

