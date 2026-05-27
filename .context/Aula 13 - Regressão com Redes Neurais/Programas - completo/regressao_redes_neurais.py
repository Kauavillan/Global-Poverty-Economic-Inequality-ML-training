# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:39:03 2020

@author: marco
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  

################## Preprocessamento ################## 

# scaler_y = StandardScaler()
# objetivo_treinamento = scaler_y.fit_transform(objetivo_treinamento)

################## Regressão com Redes Neurais ################## 

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(activation='relu',
                         max_iter=300,
                         verbose=True,
                         hidden_layer_sizes = (10,9),
                         random_state=0)

#  Treinamento
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


# Plotagem do "Mapa de calor" de uma rede neural
import matplotlib.pyplot as plt
plt.imshow(regressor.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(len(cols_previsores)), cols_previsores)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
