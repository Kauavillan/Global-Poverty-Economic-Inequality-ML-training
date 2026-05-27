# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:12:58 2020

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
plt.scatter(previsores_treinamento, objetivo_treinamento)
plt.title('Plano de Saúde')
plt.xlabel('idade')
plt.ylabel('custo')

# Visualização dos dados
plt.scatter(previsores_teste, objetivo_teste)
plt.title('Plano de Saúde')
plt.xlabel('idade')
plt.ylabel('custo')


################## Regressão com Árvores de Decisão ################## 

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=4,
                                  random_state=0)

#  Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

################## Avaliação dos resultados ################## 

# Visualização dos dados
import numpy as np
X_plot = np.arange(previsores.min().values[0], previsores.max().values[0], 0.1)
X_plot = X_plot.reshape(-1, 1)
Y_plot = regressor.predict(X_plot)

plt.scatter(previsores_treinamento, objetivo_treinamento)
plt.plot(X_plot, Y_plot, color = 'red')
plt.title('Regressão com árvores')
plt.xlabel('Idade')
plt.ylabel('Custo')

plt.scatter(previsores_teste, objetivo_teste)
plt.plot(X_plot, Y_plot, color = 'red')
plt.title('Regressão com árvores')
plt.xlabel('Idade')
plt.ylabel('Custo')


score = regressor.score(previsores_teste, objetivo_teste)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', score)  
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)

