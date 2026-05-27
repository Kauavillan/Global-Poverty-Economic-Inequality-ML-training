# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:55:50 2020

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
previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(previsores, 
                                                                                            objetivo, 
                                                                                            test_size=0.25, 
                                                                                            random_state=0)

# Visualização dos dados
plt.scatter(previsores_treinamento, objetivo_treinamento)
plt.title('Plano de Saúde')
plt.xlabel('idade')
plt.ylabel('custo')

# Visualização dos dados
plt.scatter(previsores_teste, objetivo_teste)
plt.title('Plano de Saúde')
plt.xlabel('idade')
plt.ylabel('custo')

################## Regressão Linear ################## 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#  Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

################## Avaliação dos resultados ################## 

# Visualização dos dados
plt.scatter(previsores, objetivo)
plt.plot(previsores, regressor.predict(previsores), color = 'red')
plt.title('Regressão linear')
plt.xlabel('idade')
plt.ylabel('custo')


score = regressor.score(previsores_teste, objetivo_teste)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', score)  
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)

