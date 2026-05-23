################## # Pré-processamento dos dados ##################

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =============================================================================
# Leitura dos dados
# =============================================================================

base = pd.read_csv('datasets/dataset.csv')


# =============================================================================
# Tratando valores inválidos / inconsistentes
# =============================================================================

# O dataset não possui valores faltantes, mas mantemos uma etapa de imputação
# para deixar o pré-processamento robusto caso novos dados tragam ausências.
pd.isnull(base).any()


# =============================================================================
# Separando dados em previsores e objetivo
# =============================================================================

# Descartando o identificador, que não deve entrar como atributo preditivo.
base = base.drop(columns=['record_id'])

# Variável-alvo: participação feminina na força de trabalho.
cols_objetivo = ['female_labor_participation_pct']

# Todas as demais colunas entram como previsores.
cols_previsores = [col for col in base.columns if col not in cols_objetivo]

previsores = base[cols_previsores]
objetivo = base[cols_objetivo]


# =============================================================================
# Separando em base de testes e treinamento
# =============================================================================

# Usando 25% para teste.
previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(
	previsores,
	objetivo,
	test_size=0.25,
	random_state=0,
)


# =============================================================================
#      Transformar as variáveis categóricas (ordinais) em valores numéricos
# =============================================================================

coluna_ordinal = 'income_group'

# Ordem ordinal: Low < Lower-Middle < Upper-Middle < High.
labelencoder_income_group = LabelEncoder()
labelencoder_income_group.classes_ = np.array([
	'Low Income',
	'Lower-Middle Income',
	'Upper-Middle Income',
	'High Income',
])

previsores_treinamento.loc[:, coluna_ordinal] = labelencoder_income_group.transform(
	previsores_treinamento.loc[:, coluna_ordinal]
).astype('int64')
previsores_teste.loc[:, coluna_ordinal] = labelencoder_income_group.transform(
	previsores_teste.loc[:, coluna_ordinal]
).astype('int64')


# =============================================================================
#      Transformar as variáveis categóricas (nominais) em variáveis dummy
# =============================================================================

colunas_nominais = ['country', 'region']

previsores_treinamento = pd.get_dummies(
	previsores_treinamento,
	columns=colunas_nominais,
	dtype='int64',
)
previsores_teste = pd.get_dummies(
	previsores_teste,
	columns=colunas_nominais,
	dtype='int64',
)

# Garante as mesmas colunas entre treino e teste apos o one-hot encoding.
previsores_teste = previsores_teste.reindex(columns=previsores_treinamento.columns, fill_value=0)

cols_previsores = previsores_treinamento.columns


# =============================================================================
#                     Padronização dos dados
# =============================================================================

scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
previsores_teste = scaler.transform(previsores_teste)


scaler_objetivo = StandardScaler()
objetivo_treinamento = scaler_objetivo.fit_transform(objetivo_treinamento)
objetivo_teste = scaler_objetivo.transform(objetivo_teste)
