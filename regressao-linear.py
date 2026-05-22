
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  

################## Regressão Linear Múltipla ################## 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

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

# Parâmetros estimados para o modelo
coef_0 = regressor.intercept_
coeficientes = regressor.coef_