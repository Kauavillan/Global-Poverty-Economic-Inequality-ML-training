# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:21:45 2020

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

# Padronização dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
scaler_y = StandardScaler()
objetivo_treinamento = scaler_y.fit_transform(objetivo_treinamento)

# Visualização dos dados (sem padronização)
plt.scatter(previsores, objetivo)
plt.title('Regressão com SVM (dados completos)')
plt.xlabel('idade')
plt.ylabel('custo')

# Visualização dos dados (com padronização)
plt.scatter(previsores_treinamento, objetivo_treinamento)
plt.title('Regressão com SVM (dados completos)')
plt.xlabel('idade')
plt.ylabel('custo')

################## Regressão com SVM ################## 

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', 
                 C = 1, 
                 gamma = 'scale', 
                 epsilon = 0.1)

#  Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)


# Teste (fazendo despadronização da variável objetivo)
previsoes = regressor.predict(scaler.transform(previsores_teste))
previsoes = scaler_y.inverse_transform(previsoes.reshape(-1, 1))

################## Avaliação dos resultados ################## 

# Visualização dos dados (sem padronização)
import numpy as np
X_plot = np.arange(previsores.min().values[0], previsores.max().values[0], 0.1)
X_plot = X_plot.reshape(-1, 1)
Y_plot = regressor.predict(X_plot)
plt.scatter(previsores_treinamento, objetivo_treinamento)
plt.plot(X_plot, Y_plot, color = 'red')
plt.title('Regressão com SVM')
plt.xlabel('Idade')
plt.ylabel('Custo')

# Com padronização
# Visualização dos dados
import numpy as np
X_plot = np.arange(previsores.min().values[0], previsores.max().values[0], 0.1)
X_plot = X_plot.reshape(-1,1)
Y_plot = regressor.predict(scaler.transform(X_plot))
plt.scatter(scaler.inverse_transform(previsores_treinamento.reshape(-1,1)), scaler_y.inverse_transform(objetivo_treinamento).reshape(-1,1))
plt.plot(X_plot, scaler_y.inverse_transform(Y_plot.reshape(-1,1)), color = 'red')
plt.title('Regressão com SVR')
plt.xlabel('idade')
plt.ylabel('custo')

score = metrics.r2_score(objetivo_teste, previsoes)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', score)  
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)

