# -*- coding: utf-8 -*-
# Déclaration des différentes importations
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing import sequence
from keras.layers.recurrent import GRU
import keras.backend as K
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import keras
import re
import string
import os
import tensorflow as tf

#
# Chargement et préparation des données
#

# Chargement du modèle cbox word2vec (de Google) pré-entrainé par le Docteur Jean-Phillipe Fauconnier (ingénieur chez Apple)
model_w2v = gensim.models.KeyedVectors.load_word2vec_format("frWac_no_postag_phrase_500_cbow_cut100.bin", encoding='utf-8',binary=True)

# Déclaration du callback à utiliser pour l'affichage de l'entraînement sur tensorboard avec la commande
# Visualisation avec la commande cmd tensorboard --logdir 'C:\Users\pauls\PycharmProjects\kerastest\tensorboard'
tbCallBack = keras.callbacks.TensorBoard(log_dir='./tensorboard', histogram_freq=0,
          write_graph=True, write_images=True)

# Préparation des données (tokenisation, passage en minuscules, et suppression de la ponctuation)
def preparedata(sent):
    if '-' in sent:
        sent = sent.replace('-', ' ');
    sent = sent.lower()
    translator = str.maketrans('', '', string.punctuation)
    sent = sent.translate(translator)
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

# Vectorisation des phrases avec le modèle word2vec en cinq-cents dimensions
def vectorize(sentence):
    question = []
    for word in sentence:
        word = model_w2v[word]
        question.append(word)
    return question

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
                    print(word, question)
                    question.remove(word)
                    print(question)
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

    print(questions)
    return np.array(questions), labels, originals

# Définition de la valeur du padding (le padding détermine la taille maximum et minimum de la phrase,
# si une phrase est plus grande elle sera tronquer à la fin et si elle est plus petite elle sera complétée par
# des vecteurs de 500 dimensions remplis de 0
padding_value = 12

# Obtention de la destination de travail
dir_path = os.path.dirname(os.path.realpath(__file__))

# Déclaration du fichier contenant les données
filepath = dir_path + "/test.csv"

# Récupération des données préparées (si l'on exclu le padding)
X_train, y_train, init = getprepareddata(filepath)

# Transformation des cibles (y_train) en une liste contenant une seule fois chaque label
labels = list(set(y_train))

# Padding des données avec des float32, un padding de valeurs zéros à la fin de la phrase et une troncature à la fin de la phrase
X_train = sequence.pad_sequences(X_train, maxlen=padding_value, dtype='float32',
    padding='post', truncating='post', value=0.)

# Conversion des labels en un encodage à catégories uniques
one_hot_labels = keras.utils.to_categorical(y_train, num_classes=len(labels))

#
# Création du modèle
#

# Création du modèle séquentiel
model = Sequential()

# Ajout d'une couche de cinq-cents GRUCells (Gated Recurrent Units) similaires à celle d'un LSTM (Long-Short-Term-Memory)
# Avec pour InputShape une listes d'éléments (12,500)
# Retournant une séquence de vecteurs de dimensions 500
model.add(GRU(500, return_sequences=True, input_shape=(padding_value, 500)))

# Ajout d'une autre couche de 248 GRUCells retournant une sequence de vecteurs de dimension 248
model.add(GRU(248, return_sequences=True))

# Ajout d'une autre couche de 64 GRUCells retournant un seul vecteur de dimension 64
model.add(GRU(64))

# Ajout d'une couche normale densément connectée avec pour fonction d'activation un softmax et autant de catégorie que de labels
model.add(Dense(len(labels), activation='softmax'))

# Configuration du processus d'apprentissage avec une fonction de coût categorical_crossentropy (que le modèle va essayer de minimiser),
# un optimiseur RMSProp (Root Mean Square Propagation) et avec pour métrique la précision
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#
# Entraînement du modèle du modèle
#

# Entraînement du modèle par lot de 64 questions/fonctions sur 50 époques et avec pour callback le tbCallBack permettant la visualisation sur Tensorboard
model.fit(X_train, np.array(one_hot_labels).reshape(len(X_train),len(labels)), batch_size=64, epochs=50, callbacks=[tbCallBack])

# Sauvegarde du modèle dans C:\Users\pauls\PycharmProjects\kerastest\models
model.save(dir_path + '\models\model_test.h5')

# Prédiction d'un certain nombre de fonctions à partir de questions pour vérifier le modèle en dehors de la validation effectuée en interne dans l'entraînement
category = model.predict_classes(np.array(X_train[0]).reshape(1, padding_value, 500), verbose = 1)
category1 = model.predict_classes(np.array(X_train[1]).reshape(1, padding_value, 500), verbose = 1)
category2 = model.predict_classes(np.array(X_train[2]).reshape(1, padding_value, 500), verbose = 1)
categories = model.predict_classes(np.array(X_train).reshape(len(X_train), padding_value, 500), verbose = 1)

print(init[0],category)
print(init[1],category1)
print(init[2],category2)

print(categories, y_train)

if category == [[0]] :
    print("Phrase non-pertinente", category)
else :
    print("Phrase pertinente", category)