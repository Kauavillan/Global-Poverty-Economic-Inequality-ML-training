# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:15:38 2020

@author: marco
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

######################## Funcões úteis #######################################

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(((y_true - y_pred) / y_true)) * 100)

##############################################################################

################## Preprocessamento ################## 

# Leitura dos dados
base = pd.read_csv('house_prices.csv')

# Separando dados em previsores e classes
cols_previsores = ['bedrooms','bathrooms','sqft_living', 'sqft_lot', 
                   'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                   'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']

cols_objetivo = ['price']

# Extraindo apenas os valores (numpy arrays) para facilitar a indexação no KFold
previsores = base[cols_previsores].values
objetivo = base[cols_objetivo].values

#####################################################################
####################### Validação cruzada ###########################
#####################################################################

# Divisão dos dados para validação cruzada (KFold ideal para regressão)
kfold = KFold(n_splits=5, shuffle=True, random_state=3)

scores = []
maes = []
mses = []
rmses = []
mapes = []

print("Iniciando treinamento...")

for indice_treinamento, indice_teste in kfold.split(previsores):
    
    # 1. Separar os dados PRIMEIRO (na escala original)
    X_treino = previsores[indice_treinamento]
    X_teste = previsores[indice_teste]
    
    y_treino = objetivo[indice_treinamento]
    y_teste = objetivo[indice_teste]
    
    # 2. Padronização DENTRO do loop (evita Data Leakage)
    scaler_x = StandardScaler()
    X_treino_scaled = scaler_x.fit_transform(X_treino)
    X_teste_scaled = scaler_x.transform(X_teste)
    
    scaler_y = StandardScaler()
    y_treino_scaled = scaler_y.fit_transform(y_treino)
    
    # 3. Construção e Treinamento do Modelo
    regressor = MLPRegressor(activation='relu',
                             max_iter=300,
                             verbose=True,
                             hidden_layer_sizes=(10,9),
                             random_state=0)
    
    # Treinamento
    regressor.fit(X_treino_scaled, y_treino_scaled.ravel())
    
    # 4. Previsões (feitas com os dados de teste padronizados)
    previsoes_scaled = regressor.predict(X_teste_scaled)
    
    # 5. Voltar as previsões para a escala original para calcular os erros reais
    previsoes = scaler_y.inverse_transform(previsoes_scaled.reshape(-1, 1))
    
    # 6. Avaliação
    score = metrics.r2_score(y_teste, previsoes)
    mae = metrics.mean_absolute_error(y_teste, previsoes)
    mse = metrics.mean_squared_error(y_teste, previsoes)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_teste, previsoes)

    scores.append(score)
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
    mapes.append(mape)


######################## Resultado final ########################
# Métricas médias
scores = np.asarray(scores)
score_final_medio = scores.mean()
score_final_desvio_padrao = scores.std()

maes = np.asarray(maes)
mae_final_medio = maes.mean()
mae_final_desvio_padrao = maes.std()

mses = np.asarray(mses)
mse_final_medio = mses.mean()
mse_final_desvio_padrao = mses.std()

rmses = np.asarray(rmses)
rmse_final_medio = rmses.mean()
rmse_final_desvio_padrao = rmses.std()

mapes = np.asarray(mapes)
mape_final_medio = mapes.mean()
mape_final_desvio_padrao = mapes.std()

print("\n--- Resultados Finais ---")
print(f"R² Médio: {score_final_medio:.4f}")
print(f"MAE Médio: {mae_final_medio:.2f}")
print(f"RMSE Médio: {rmse_final_medio:.2f}")
print(f"MAPE Médio: {mape_final_medio:.2f}%\n")

################## Gráficos de Avaliação #######################################

sns.set_style("whitegrid")
sns.despine(top=True, right=False, left=False, bottom=False, offset=None, trim=False)

# Usando o modelo da última iteração (último fold) para plotar os gráficos
previsoes_treinamento_scaled = regressor.predict(X_treino_scaled)
previsoes_treinamento = scaler_y.inverse_transform(previsoes_treinamento_scaled.reshape(-1, 1))

# Cálculo dos erros (desvio relativo)
erros_treinamento = (y_treino - previsoes_treinamento) / y_treino
erros_teste = (y_teste - previsoes) / y_teste

# 1. Gráfico de Resíduos (Residplot)
plt.figure(figsize=(8, 5))
# .ravel() é usado para transformar a matriz 2D em 1D e evitar warnings do seaborn
ax1 = sns.residplot(x=y_treino.ravel(), y=previsoes_treinamento.ravel(), lowess=False, color="blue", label='Treinamento')
ax1 = sns.residplot(x=y_teste.ravel(), y=previsoes.ravel(), lowess=False, color="orange", label='Teste')
ax1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax1.set_xlabel("Valor Real (Imóvel)", fontsize=12)
ax1.set_ylabel("Resíduos", fontsize=12)
ax1.set_title("Gráfico de Resíduos")

# 2. Gráfico de Previsão vs Real
plt.figure(figsize=(8, 5))
plt.scatter(x=y_treino, y=previsoes_treinamento, alpha=0.5, label='Treinamento', color="blue")
plt.scatter(x=y_teste, y=previsoes, alpha=0.5, label='Teste', color="orange")
# Adicionando uma reta de referência (onde Previsão = Valor Real)
min_val = min(y_treino.min(), y_teste.min())
max_val = max(y_treino.max(), y_teste.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Previsão Perfeita')
plt.xlabel("Valor Real")
plt.ylabel("Previsão")
plt.title("Previsões vs Valores Reais")
plt.legend()

# 3. Histograma dos resíduos (Desvio Relativo)
plt.figure(figsize=(8, 5))
# Atualizado de sns.distplot para sns.histplot 
ax2 = sns.histplot(erros_treinamento.ravel(), kde=True, stat="density", color="blue", label="Treinamento", alpha=0.4)
ax2 = sns.histplot(erros_teste.ravel(), kde=True, stat="density", color="orange", label="Teste", alpha=0.4)
ax2.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax2.set_xlabel("Desvio Relativo", fontsize=12)
ax2.set_ylabel("Densidade", fontsize=12)
ax2.set_title("Distribuição do Desvio Relativo")

plt.show()