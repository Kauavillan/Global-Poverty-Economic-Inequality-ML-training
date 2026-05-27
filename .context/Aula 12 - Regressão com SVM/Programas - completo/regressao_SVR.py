# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:01:55 2020

@author: marco
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

################# Regressão com SVR ##############

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf',
                C = 1,
                gamma = 'scale',
                epsilon = 0.1)

# Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

previsoes_escala_original = scaler_objetivo.inverse_transform(previsoes.reshape(-1, 1))
objetivo_escala_original = scaler_objetivo.inverse_transform(objetivo_teste.reshape(-1, 1))

########### Avaliação dos resultados ###############

score = regressor.score(previsores_teste, objetivo_teste)

mae = metrics.mean_absolute_error(objetivo_escala_original, previsoes_escala_original)
mse = metrics.mean_squared_error(objetivo_escala_original, previsoes_escala_original)
rmse = np.sqrt(mse)

print('Score:',score)
print('Mean Absolute Error:',mae)
print('Mean Squared Error:',mse)
print('Root Mean Squared Error:',rmse)