# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:42:10 2020

@author: marco
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  

################## Preprocessamento ################## 

base = pd.read_csv('plano_saude.csv')

# Separando dados em previsores e classes
cols_previsores = ['idade']

cols_objetivo = ['custo']
previsores = base[cols_previsores]
objetivo = base[cols_objetivo]

# Separando em base de testes e treinamento (usando 25% para teste)
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(previsores, objetivo, test_size=0.25, random_state=0)

# Visualização dos dados
plt.scatter(previsores, objetivo)
plt.title('Regressão com SVM (dados completos)')
plt.xlabel('idade')
plt.ylabel('custo')


################## Regressão com Redes Neurais ################## 

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(activation='relu',
                         max_iter=100,
                         verbose=True,
                         hidden_layer_sizes = (100),
                         random_state=0)

#  Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

################## Avaliação dos resultados ################## 

# Visualização dos dados de treinamento
plt.scatter(previsores_treinamento, objetivo_treinamento)
plt.plot(previsores_treinamento, regressor.predict(previsores_treinamento), color = 'red')
plt.title('Regressão com SVM')
plt.xlabel('Idade')
plt.ylabel('Custo')

# Visualização dos dados de teste
plt.scatter(previsores_teste, objetivo_teste)
plt.plot(previsores_teste.values, previsoes, color = 'red')
plt.title('Regressão com SVM')
plt.xlabel('Idade')
plt.ylabel('Custo')

score = metrics.r2_score(objetivo_teste, previsoes)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', score)  
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)

