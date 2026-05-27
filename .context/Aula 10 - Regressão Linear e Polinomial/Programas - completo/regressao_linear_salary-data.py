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

base = pd.read_csv('salary_data.csv')

# Separando dados em previsores e classes
cols_previsores = ['YearsExperience']

cols_objetivo = ['Salary']
previsores = base[cols_previsores]
objetivo = base[cols_objetivo]

# Separando em base de testes e treinamento (usando 25% para teste)
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(previsores, 
                                                                                                  objetivo, 
                                                                                                  test_size=0.25, 
                                                                                                  random_state=0)

# Visualização dos dados
plt.scatter(previsores, objetivo)
plt.title('Regressão linear (dados completos)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')

# Visualização dos dados
plt.scatter(previsores_treinamento, objetivo_treinamento)
plt.title('Regressão linear (dados de treinamento)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')

# Visualização dos dados
plt.scatter(previsores_teste, objetivo_teste)
plt.title('Regressão linear (dados de teste)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')



################## Regressão Linear ################## 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#  Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

################## Avaliação dos resultados ################## 

# Visualização dos dados de treinamento
plt.scatter(previsores_treinamento, objetivo_treinamento)
plt.plot(previsores_treinamento, regressor.predict(previsores_treinamento), color = 'red')
plt.title('Regressão linear (treinamento)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')

# Visualização dos dados de teste
plt.scatter(previsores_teste, objetivo_teste)
plt.plot(previsores_teste, previsoes, color = 'red')
plt.title('Regressão linear (teste)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')

score = regressor.score(previsores_teste, objetivo_teste)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', score)  
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)

coef_0 = regressor.intercept_
coeficientes = regressor.coef_


