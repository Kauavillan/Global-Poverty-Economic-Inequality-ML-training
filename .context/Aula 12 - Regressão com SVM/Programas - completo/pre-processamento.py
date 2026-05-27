# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:53:31 2019

@author: marco
"""

import pandas as pd

################## #Pré-processamento dos dados ################## 

#Leitura dos dados
base = pd.read_csv('house_prices.csv')
base.describe()

# =============================================================================
#                     Tratando valores inválidos
# =============================================================================
# Não há valores inconsistentes

# Procurando as colunas que possuem algum valor faltante
pd.isnull(base).any()

# =============================================================================
#                     Separando dados em previsores e objetivo
# =============================================================================

cols_previsores = ['bedrooms','bathrooms','sqft_living', 'sqft_lot', 
                   'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                   'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']
# Não usarei: date, sqft_living15 e sqft_lot15


cols_objetivo = ['price']
previsores = base[cols_previsores]
objetivo = base[cols_objetivo]

# =============================================================================
#      Transformar as variáveis categóricas em valores numéricos
# =============================================================================
#  Todas as variáveis são numéricas

# =============================================================================
#                 Separando em base de testes e treinamento
# =============================================================================

#  usando 25% para teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(previsores, 
                                                                                                  objetivo, 
                                                                                                  test_size=0.25, 
                                                                                                  random_state=0)

# =============================================================================
#                     Padronização dos dados
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
previsores_teste = scaler.transform(previsores_teste)

scaler_objetivo = StandardScaler()
objetivo_treinamento = scaler_objetivo.fit_transform(objetivo_treinamento)
objetivo_teste = scaler_objetivo.transform(objetivo_teste)

