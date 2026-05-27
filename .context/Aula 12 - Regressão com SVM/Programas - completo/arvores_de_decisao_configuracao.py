# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:19:58 2019

@author: marco
"""

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
training_accuracy = []
test_accuracy = []

# tentando diferentes valores de K: de 1 to 20
neighbors_settings = range(1, 20)
for k in neighbors_settings:
    # Construindo o modelo
    classificador = DecisionTreeRegressor(max_depth=k, random_state=0)
    # Treinamento do modelo
    classificador.fit(previsores_treinamento, objetivo_treinamento)    
    # Gravando o resultado para os dados de treinamento
    training_accuracy.append(classificador.score(previsores_treinamento, objetivo_treinamento))
    # Gravando o resultado para os dados de teste (generalização)
    test_accuracy.append(classificador.score(previsores_teste, objetivo_teste))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Altura da árvore")
plt.legend()


