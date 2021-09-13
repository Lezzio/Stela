# -*- coding: utf-8 -*-
# Déclaration des différentes importations
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing import sequence
from keras.layers.recurrent import GRU
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import keras
import re
import string
import os

# Obtention de la destination de travail
dir_path = os.path.dirname(os.path.realpath(__file__))

# Chargement du modèle cbox word2vec (de Google) pré-entrainé par le Docteur Jean-Phillipe Fauconnier (ingénieur chez Apple)
model_w2v = gensim.models.KeyedVectors.load_word2vec_format(dir_path + "/word2vec/frWac_no_postag_phrase_500_cbow_cut100.bin", encoding='utf-8',binary=True)

# Déclaration du callback à utiliser pour l'affichage de l'entraînement sur tensorboard avec la commande
# Visualisation avec la commande cmd tensorboard --logdir 'C:\Users\pauls\PycharmProjects\kerastest\tensorboard'
tbCallBack = keras.callbacks.TensorBoard(log_dir='./tensorboard', histogram_freq=0,
          write_graph=True, write_images=True)

# Préparation des données (tokenisation, passage en minuscules, et suppression de la ponctuation)
def preparedata(sent):
    sent = sent.lower()
    replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    sent = sent.translate(replace_punctuation)
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

# Vectorisation des phrases avec le modèle word2vec en cinq-cents dimensions
def vectorize(sentence):
    question = []
    for word in sentence:
        word = model_w2v[word]
        question.append(word)
    return question

def removeWords(sent):
    for word in sent:
        i = sent.index(word)
        if not model_w2v.__contains__(word):
            sent.remove(word)
    return sent


# Obtention des données préparées (si l'on exclu le padding)
def getprepareddata(filepath):
    questions = []
    labels = []
    originals = []
    with open(filepath) as inf:
        for line in inf:
            original, label = line.strip().split(";")
            question = preparedata(original)
            for word in question:
                i = question.index(word)
                if not model_w2v.__contains__(word):
                    question.remove(word)
                    if i == len(question):
                        question = vectorize(question)
                        questions.append(question)
                        labels.append(label)
                        originals.append(original + label)
                elif i + 1 == len(question):
                    question = vectorize(question)
                    questions.append(question)
                    labels.append(label)
                    originals.append(original + label)

    return np.array(questions), labels, originals

# Définition de la valeur du padding (le padding détermine la taille maximum et minimum de la phrase,
# si une phrase est plus grande elle sera tronquer à la fin et si elle est plus petite elle sera complétée par
# des vecteurs de 500 dimensions remplis de 0
padding_value = 12

# Déclaration du fichier contenant les données
filepath = dir_path + "/test.csv"

# Récupération des données préparées (si l'on exclu le padding)
X_train, y_train, init = getprepareddata(filepath)

# Transformation des cibles (y_train) en une liste contenant une seule fois chaque label
labels = list(set(y_train))
labels.sort()

# Padding des données avec des float32, un padding de valeurs zéros à la fin de la phrase et une troncature à la fin de la phrase
X_train = sequence.pad_sequences(X_train, maxlen=padding_value, dtype='float32',
    padding='post', truncating='post', value=0.)

# Conversion des labels en un encodage à catégories uniques
one_hot_labels = keras.utils.to_categorical(np.array(labels), num_classes=len(labels))

# Création du modèle séquentiel
model = Sequential()

# Ajout d'une couche de cinq-cents GRUCells (Gated Recurrent Units) similaires à celle d'un LSTM (Long-Short-Term-Memory)
# Avec pour InputShape une listes d'éléments (12,500)
# Retournant une séquence de vecteurs de dimensions 256
model.add(GRU(256, return_sequences=True, input_shape=(None, 500)))

# Ajout d'une autre couche de 248 GRUCells retournant une sequence de vecteurs de dimension 128
model.add(GRU(128, return_sequences=True))

# Ajout d'une autre couche de 64 GRUCells retournant un seul vecteur de dimension 64
model.add(GRU(64))

# Ajout d'une couche normale densément connectée avec pour fonction d'activation un softmax et autant de catégorie que de labels
model.add(Dense(len(labels), activation='softmax'))

# Configuration du processus d'apprentissage avec une fonction de coût categorical_crossentropy (que le modèle va essayer de minimiser),
# un optimiseur RMSProp (Root Mean Square Propagation) et avec pour métrique la précision
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Entraînement du modèle par lot de 64 questions/fonctions sur 50 époques et avec pour callback le tbCallBack permettant la visualisation sur Tensorboard
model.fit(X_train, np.array(one_hot_labels), batch_size=64, epochs=50, callbacks=[tbCallBack])

# Prédiction d'un certain nombre de fonctions à partir de questions pour vérifier le modèle en dehors de la validation effectuée en interne dans l'entraînement
category1 = model.predict_classes(np.array(X_train[1]).reshape(1, padding_value, 500), verbose = 1)
categories = model.predict_classes(np.array(X_train).reshape(len(X_train), padding_value, 500), verbose = 1)
print(categories, labels)

def predict(sentence):
    sentence = np.array([vectorize(removeWords(preparedata(sentence)))])
    sentence = sequence.pad_sequences(sentence, maxlen=padding_value, dtype='float32',
    padding='post', truncating='post', value=0.)
    return model.predict_classes(np.array(sentence).reshape(1, padding_value, 500))


print(predict('Salut comment vas-tu ?'))
print(predict('C\'est quoi ton activité préféré ?'))
print(predict('Mais qui es-tu ?'))
print(predict('Je n\'est pas pas compris qui tu étais')) #This one is amazing