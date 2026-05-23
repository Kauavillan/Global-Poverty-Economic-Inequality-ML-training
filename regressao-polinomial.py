################## Regressão Polinomial ################## 

from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

poly = PolynomialFeatures(degree = 3)
previsores_treinamento_poly = poly.fit_transform(previsores_treinamento)
previsores_teste_poly = poly.transform(previsores_teste)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#  Treinamento
regressor.fit(previsores_treinamento_poly, objetivo_treinamento)

score = regressor.score(previsores_treinamento_poly, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste_poly)

################## Avaliação dos resultados ################## 

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
