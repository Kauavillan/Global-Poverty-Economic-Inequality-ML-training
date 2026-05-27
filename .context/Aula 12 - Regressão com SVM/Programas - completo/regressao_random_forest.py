# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:13:33 2020

@author: marco
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  

################## Preprocessamento ################## 

# Arquivo separado

################## Regressão com Random Forest ################## 

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100,     # número de árvores
                                  max_features=10, # qtd de caracteristicas
                                  max_depth=15,
                                  random_state=0)

#  Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

################## Avaliação dos resultados ################## 

score = regressor.score(previsores_teste, objetivo_teste)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', score)  
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)

# Visualizando a importância das características
import matplotlib.pyplot as plt
import numpy as np
n_features = previsores.columns.size
plt.barh(range(n_features), regressor.feature_importances_, align='center')
plt.yticks(np.arange(n_features), previsores.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")


