# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:18:09 2020

@author: marco
"""

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
training_score = []
test_score = []

#  testando diferentes valores
altura = range(1, 20)
for h in altura:
    # Construindo o modelo
    regressor = DecisionTreeRegressor(max_depth = h, random_state = 0)
    #  Treinando o modelo
    regressor.fit(previsores_treinamento, objetivo_treinamento)
    #  Gravando o resultado para os dados de treinamento
    training_score.append(regressor.score(previsores_treinamento, objetivo_treinamento))
    #  Gravando o resultado para os dados de teste
    test_score.append(regressor.score(previsores_teste, objetivo_teste))
    
plt.plot(altura, training_score, label = "training_score")
plt.plot(altura, test_score, label = "training_score")
plt.ylabel("Score")
plt.xlabel("Altura da árvore")
plt.legend()