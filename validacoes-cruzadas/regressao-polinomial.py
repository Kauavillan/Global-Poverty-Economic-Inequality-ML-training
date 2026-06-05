import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import os

# ######################## Funções úteis #######################################

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(((y_true - y_pred) / y_true)) * 100)


# ################## Preprocessamento ################## 

# Leitura dos dados
base = pd.read_csv('datasets/dataset.csv')

# Descartando o identificador
base = base.drop(columns=['record_id'])

# Variável-alvo
cols_objetivo = ['female_labor_participation_pct']
cols_previsores = [col for col in base.columns if col not in cols_objetivo]

# Identificando colunas numéricas (contínuas) e categóricas
colunas_nominais = ['country', 'region']
colunas_numericas = [col for col in cols_previsores if col not in colunas_nominais and col != 'income_group']

previsores = base[cols_previsores].copy()
objetivo = base[cols_objetivo].copy()

# Transformar as variáveis categóricas (ordinais) em valores numéricos
coluna_ordinal = 'income_group'
labelencoder_income_group = LabelEncoder()
labelencoder_income_group.classes_ = np.array([
    'Low Income',
    'Lower-Middle Income',
    'Upper-Middle Income',
    'High Income',
])
previsores.loc[:, coluna_ordinal] = labelencoder_income_group.transform(
    previsores.loc[:, coluna_ordinal]
).astype('int64')

# Transformar as variáveis categóricas (nominais) em variáveis dummy
previsores = pd.get_dummies(
    previsores,
    columns=colunas_nominais,
    dtype='int64',
)

# Identificando as colunas no DataFrame final para separação no loop
cols_num = list(colunas_numericas)
cols_cat = [col for col in previsores.columns if col not in cols_num]

# Índices das colunas numéricas e categóricas no array numpy
idx_num = [previsores.columns.get_loc(c) for c in cols_num]
idx_cat = [previsores.columns.get_loc(c) for c in cols_cat]

# Extraindo apenas os valores (numpy arrays) para facilitar a indexação no KFold
previsores_arr = previsores.values
objetivo_arr = objetivo.values

# #####################################################################
# ####################### Validação cruzada ###########################
# #####################################################################

# Divisão dos dados para validação cruzada (KFold com 5 splits)
kfold = KFold(n_splits=5, shuffle=True, random_state=3)

scores = []
maes = []
mses = []
rmses = []
mapes = []

print("Iniciando treinamento (Regressão Polinomial)...")

for indice_treinamento, indice_teste in kfold.split(previsores_arr):
    
    # 1. Separar os dados PRIMEIRO (na escala original)
    X_treino = previsores_arr[indice_treinamento]
    X_teste = previsores_arr[indice_teste]
    
    y_treino = objetivo_arr[indice_treinamento]
    y_teste = objetivo_arr[indice_teste]
    
    # Separando numéricas de categóricas
    X_treino_num = X_treino[:, idx_num]
    X_treino_cat = X_treino[:, idx_cat]
    X_teste_num = X_teste[:, idx_num]
    X_teste_cat = X_teste[:, idx_cat]
    
    # 2. Padronização DENTRO do loop (evita Data Leakage)
    scaler_x_num = StandardScaler()
    X_treino_num_scaled = scaler_x_num.fit_transform(X_treino_num)
    X_teste_num_scaled = scaler_x_num.transform(X_teste_num)
    
    # Padronização das categóricas
    scaler_x_cat = StandardScaler()
    X_treino_cat_scaled = scaler_x_cat.fit_transform(X_treino_cat)
    X_teste_cat_scaled = scaler_x_cat.transform(X_teste_cat)
    
    # 3. Transformação Polinomial (apenas nas numéricas)
    # NOTA: O modelo original no script "regressao-polinomial.py" usa degree=5 na base inteira. 
    # Para evitar explosão de features e problemas de convergência por colinearidade perfeita
    # de variáveis dummy (onde x_i * x_j = 0 ou x_i^2 = x_i), aplicamos grau 2 apenas nas variáveis numéricas.
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_treino_num_poly = poly.fit_transform(X_treino_num_scaled)
    X_teste_num_poly = poly.transform(X_teste_num_scaled)
    
    # Concatenando as numéricas polinomiais e as categóricas escaladas
    X_treino_poly_final = np.hstack((X_treino_num_poly, X_treino_cat_scaled))
    X_teste_poly_final = np.hstack((X_teste_num_poly, X_teste_cat_scaled))
    
    scaler_y = StandardScaler()
    y_treino_scaled = scaler_y.fit_transform(y_treino)
    
    # 4. Construção e Treinamento do Modelo
    regressor = LinearRegression()
    
    # Treinamento
    regressor.fit(X_treino_poly_final, y_treino_scaled.ravel())
    
    # 5. Previsões (feitas com os dados de teste correspondentes)
    previsoes_scaled = regressor.predict(X_teste_poly_final)
    
    # 6. Voltar as previsões para a escala original para calcular os erros reais
    previsoes = scaler_y.inverse_transform(previsoes_scaled.reshape(-1, 1))
    
    # 7. Avaliação
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


# ######################## Resultado final ########################
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

print("\n--- Resultados Finais (Regressão Polinomial) ---")

# Formatar resultados com vírgula como separador decimal
score_str = f"{score_final_medio:.4f}".replace('.', ',')
score_sd_str = f"{score_final_desvio_padrao:.4f}".replace('.', ',')
print(f"Score (R²): {score_str} (± {score_sd_str})")

mape_str = f"{mape_final_medio:.4f}".replace('.', ',')
mape_sd_str = f"{mape_final_desvio_padrao:.4f}".replace('.', ',')
print(f"MAPE Médio: {mape_str}% (± {mape_sd_str})")

mae_str = f"{mae_final_medio:.4f}".replace('.', ',')
mae_sd_str = f"{mae_final_desvio_padrao:.4f}".replace('.', ',')
print(f"MAE Médio: {mae_str} (± {mae_sd_str})")

rmse_str = f"{rmse_final_medio:.4f}".replace('.', ',')
rmse_sd_str = f"{rmse_final_desvio_padrao:.4f}".replace('.', ',')
print(f"RMSE Médio: {rmse_str} (± {rmse_sd_str})")

mse_str = f"{mse_final_medio:.4f}".replace('.', ',')
mse_sd_str = f"{mse_final_desvio_padrao:.4f}".replace('.', ',')
print(f"MSE Médio: {mse_str} (± {mse_sd_str})")

# ################## Gráficos de Avaliação #######################################

sns.set_style("whitegrid")

# Usando o modelo da última iteração (último fold) para plotar os gráficos
previsoes_treinamento_scaled = regressor.predict(X_treino_poly_final)
previsoes_treinamento = scaler_y.inverse_transform(previsoes_treinamento_scaled.reshape(-1, 1))

# Cálculo dos erros (desvio relativo)
erros_treinamento = (y_treino - previsoes_treinamento) / y_treino
erros_teste = (y_teste - previsoes) / y_teste

# 1. Gráfico de Resíduos (Residplot)
plt.figure(figsize=(8, 5))
ax1 = sns.residplot(x=y_treino.ravel(), y=previsoes_treinamento.ravel(), lowess=False, color="blue", label='Treinamento')
ax1 = sns.residplot(x=y_teste.ravel(), y=previsoes.ravel(), lowess=False, color="orange", label='Teste')
ax1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax1.set_xlabel("Valor Real (female_labor_participation_pct)", fontsize=12)
ax1.set_ylabel("Resíduos", fontsize=12)
ax1.set_title("Gráfico de Resíduos (Regressão Polinomial)")

# 2. Gráfico de Previsão vs Real
plt.figure(figsize=(8, 5))
plt.scatter(x=y_treino, y=previsoes_treinamento, alpha=0.5, label='Treinamento', color="blue")
plt.scatter(x=y_teste, y=previsoes, alpha=0.5, label='Teste', color="orange")
min_val = min(y_treino.min(), y_teste.min())
max_val = max(y_treino.max(), y_teste.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Previsão Perfeita')
plt.xlabel("Valor Real")
plt.ylabel("Previsão")
plt.title("Previsões vs Valores Reais (Regressão Polinomial)")
plt.legend()

# 3. Histograma dos resíduos (Desvio Relativo)
plt.figure(figsize=(8, 5))
ax2 = sns.histplot(erros_treinamento.ravel(), kde=True, stat="density", color="blue", label="Treinamento", alpha=0.4)
ax2 = sns.histplot(erros_teste.ravel(), kde=True, stat="density", color="orange", label="Teste", alpha=0.4)
ax2.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax2.set_xlabel("Desvio Relativo", fontsize=12)
ax2.set_ylabel("Densidade", fontsize=12)
ax2.set_title("Distribuição do Desvio Relativo (Regressão Polinomial)")

plt.show()
