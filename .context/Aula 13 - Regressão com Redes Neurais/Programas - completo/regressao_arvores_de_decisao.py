# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:04:13 2020

@author: marco
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


################# Regressão com Árvores de Decisão ##############

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth = 9,
                                  random_state=0)

# Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

########### Avaliação dos resultados ###############

score = regressor.score(previsores_teste, objetivo_teste)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(mse)

print('Score:',score)
print('Mean Absolute Error:',mae)
print('Mean Squared Error:',mse)
print('Root Mean Squared Error:',rmse)


#######################################
# Gerando árvore
from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file="tree.dot", feature_names=cols_previsores, impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

# Visualizar a importância de cada caracteristica
import matplotlib.pyplot as plt
import numpy as np
n_features = previsores.columns.size
plt.barh(range(n_features), regressor.feature_importances_, align='center')
plt.yticks(np.arange(n_features), previsores.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
