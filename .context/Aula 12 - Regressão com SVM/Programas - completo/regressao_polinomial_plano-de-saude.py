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
previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(previsores, objetivo, test_size=0.25, random_state=0)

# Visualização dos dados
plt.scatter(previsores, objetivo)
plt.title('Regressão linear (dados completos)')
plt.xlabel('idade')
plt.ylabel('custo')

################## Regressão Polinomial ################## 

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3)
previsores_treinamento_poly = poly.fit_transform(previsores_treinamento)
previsores_teste_poly = poly.transform(previsores_teste)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#  Treinamento
regressor.fit(previsores_treinamento_poly, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste_poly)

################## Avaliação dos resultados ################## 

# Visualização dos dados de treinamento
plt.scatter(previsores_treinamento, objetivo_treinamento)
plt.plot(previsores, regressor.predict(poly.fit_transform(previsores)), color = 'red')
plt.title('Regressão polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')

# Visualização dos dados de teste
plt.scatter(previsores_teste, objetivo_teste)
plt.plot(previsores, regressor.predict(poly.fit_transform(previsores)), color = 'red')
plt.title('Regressão polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')

score = regressor.score(previsores_teste_poly, objetivo_teste)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', score)  
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)

# Parâmetros estimados para o modelo
coef_0 = regressor.intercept_
coeficientes = regressor.coef_

