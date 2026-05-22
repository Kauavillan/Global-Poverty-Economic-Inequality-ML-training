################## # Pré-processamento dos dados ##################

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
# Transformar variáveis categóricas em valores numéricos
# =============================================================================

colunas_categoricas = ['country', 'region', 'income_group']
colunas_numericas = [col for col in cols_previsores if col not in colunas_categoricas]

preprocessador = ColumnTransformer(
	transformers=[
		(
			'numericas',
			Pipeline(steps=[
				('imputer', SimpleImputer(strategy='median')),
				('scaler', StandardScaler()),
			]),
			colunas_numericas,
		),
		(
			'categoricas',
			Pipeline(steps=[
				('imputer', SimpleImputer(strategy='most_frequent')),
				('onehot', OneHotEncoder(handle_unknown='ignore')),
			]),
			colunas_categoricas,
		),
	]
)


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
# Pré-processamento dos dados
# =============================================================================

previsores_treinamento = preprocessador.fit_transform(previsores_treinamento)
previsores_teste = preprocessador.transform(previsores_teste)
