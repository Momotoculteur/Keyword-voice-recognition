# Infos
Code source du tutoriel de https://deeplylearning.fr/cours-pratiques-deep-learning/reconnaissance-vocale-de-mots-cles/

Permet de creer avec Tensorflow et Keras une reconnaissance vocale de mots clés entre 2 types de mots différentes, avec des algorithmes de deep learning.

## Installer les pré-requis
Permet d'installer les différentes bibliothèques essentiel pour réaliser des algorithmes de deep learning. Optionnel si vous avez déjà votre environnement pré-configuré.

`$ install.bat`

## Generer les tableaux numpy
Changer les chemins si nécessaire :

| Attribut | Description                    |
| ------------- | ------------------------------ |
| `pathNumpy`      | Destination ou seront sauvegarder les tableaux      |
| `pathData`   | Chemin ou sont les images en format png    |
| `imgResize`   | On met l'ensemble du dataset à la même taille    | 

Lancer la commande suivante :

`$ python generateNumpyFiles.py`

## Entrainer le model
Changer les chemins si nécessaire :

| Attribut | Description                    |
| ------------- | ------------------------------ |
| `csv_logger`      | Chemin du callbak permettant l'enregistrement des metriques      |
| `check`   | Chemin du callback permettant d'enregistrer le modèle sous format hdf5    |
| `pathData`   | Chemin des tableaux numpy   |
| `trainRatio`   | Ratio définissant la taille du jeu d'entrainemnt et de validation   |
| `batch_size`   | Nombre d'item que on envoi sur une phase de feedforward/backpropagation   |
| `earlyStopPatience`   | Permet de définir l'arrêt de l'entrainement, lorsque les données de précision sur le jeu de validation n'évolu plus  |

Lancer la commande suivante :

`$ python trainModel.py`

## Generer les graphiques de suivi de métriques
Changer les chemins si nécessaire :

| Attribut | Description                    |
| ------------- | ------------------------------ |
| `pathLogs`      | Destination ou est notre fichier de métriques au format CSV      |
| `pathSaveGraph`   | Chemin ou on va sauvegarder    |

Lancer la commande suivante :

`$ python generateMetrics.py`

## Generer la matrice de confusion
Changer les chemins si nécessaire :

| Attribut | Description                    |
| ------------- | ------------------------------ |
| `modelPath`      | Destination ou est stocké notre modele pré entrainé      |
| `datasetTestPath`   | Chemin ou sont les audios de test    |
| `destinationMatrix`   | Destination ou on va sauvegarder notre matrice     |
| `imageSize`   | On met l'ensemble du dataset à la même taille    |

Lancer la commande suivante :

`$ python generateConfusionMatrix.py`

## Realiser une prédiction sur une nouvelle donnée
Changer les chemins si nécessaire :

| Attribut | Description                    |
| ------------- | ------------------------------ |
| `modelPath`      | Chemin du model au format hdf5 pour le charger en memoire      |
| `soundPath`   | Chemin de l audio que l'on doit predire    |
| `soundSize`   | Doit être identique aux tailles d audio d'entrainement    |
| `labels`   | Libéllé des classes à prédire    |
| `format`   | Doit être identique aux tailles d image d entrainement    |
| `channels`   | Nombre de canaux de l audio    |
| `chunk`   | Taille du buffer de l audio    |
| `rate`   | Taux d échantillonage de l audio   |
| `recordTime`   | Temps d enregistrement de l audio   |



Puis lancer la commande suivante :

`$ python autoPredict.py`