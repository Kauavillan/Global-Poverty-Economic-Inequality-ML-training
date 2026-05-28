################################################################################
# Regressão com Árvores de Decisão
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.inspection import permutation_importance
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

TESTAR_POSSIBILIDADES = False

# =============================================================================
# 1. Importação do Pré-Processamento
# =============================================================================
if 'objetivo_treinamento' not in locals():
    print("Variáveis não encontradas. Executando pre-processamento.py...")
    import os
    os.makedirs('resultados', exist_ok=True)
    with open('pre-processamento.py', 'r', encoding='utf-8') as f:
        exec(f.read())

# Ajuste para formato exigido por alguns modelos (vetor 1D)
y_treino = objetivo_treinamento.ravel()
y_teste = objetivo_teste.ravel()

# =============================================================================
# 2. Funções de Teste de Regressão
# =============================================================================
def executar_teste(X_train, y_train, X_test, y_test, nomes_colunas, nome_teste, testar_possibilidades=True):
    print(f"\n\n{'='*80}")
    print(f"=== TESTE: {nome_teste} ===")
    print(f"{'='*80}\n")

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    if testar_possibilidades:
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        }

        print(f"[{nome_teste}] 1. Treinando e otimizando parâmetros (GridSearch)...")
        grid_search = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv_strategy, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        melhor_modelo = grid_search.best_estimator_
        print(f"[{nome_teste}] Melhor configuração: {grid_search.best_params_}")
    else:
        print(f"[{nome_teste}] 1. Treinando modelo com configuração predefinida...")
        # Configuração baseada nos exemplos de aula
        melhor_modelo = DecisionTreeRegressor(max_depth=9, random_state=42)

    print(f"[{nome_teste}] 2. Rodando Validação Cruzada KFold...")
    scores = cross_validate(melhor_modelo, X_train, y_train, cv=cv_strategy, scoring=('r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'))

    print(f"\n--- MÉTRICAS NA VALIDAÇÃO CRUZADA DE TREINO ({nome_teste}) ---")
    print(f"Score Médio CV (R²): {np.mean(scores['test_r2']):.4f}")

    # 3. Teste final na base de teste
    melhor_modelo.fit(X_train, y_train)
    previsoes = melhor_modelo.predict(X_test)
    
    print(f"\n--- MÉTRICAS FINAIS NA BASE DE TESTE ({nome_teste}) ---")
    print(f"Score (R²): {melhor_modelo.score(X_test, y_test):.4f}")
    print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, previsoes):.4f}")
    print(f"Mean Squared Error: {metrics.mean_squared_error(y_test, previsoes):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, previsoes)):.4f}")

    top_features = []
    print(f"\n[{nome_teste}] 3. Analisando a importância das variáveis...")
    importances = melhor_modelo.feature_importances_
    indices = importances.argsort()[::-1]
    
    top_n = min(10, len(nomes_colunas))
    top_features = np.array(nomes_colunas)[indices[:top_n]]
    top_importances = importances[indices[:top_n]]

    for i in range(len(top_features)):
        print(f"    {i+1}º - {top_features[i]}: {top_importances[i]:.4f}")

    # Geração de Gráfico de Importância
    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_importances[::-1], color='forestgreen')
    plt.xlabel("Importância da Feature")
    plt.title(f"Importância de Features (Árvore de Decisão) - {nome_teste}")
    plt.tight_layout()
    
    os.makedirs('resultados', exist_ok=True)
    nome_arquivo = f"resultados/plot_Arvore_{nome_teste.replace(' ', '_').replace(':', '')}.png"
    plt.savefig(nome_arquivo)
    print(f"[{nome_teste}] 4. Gráfico gerado com sucesso: '{nome_arquivo}'")
    plt.close()

    return melhor_modelo, list(top_features)


# =============================================================================
# 3. Execução Sequencial dos Experimentos (5 Configurações)
# =============================================================================
if __name__ == '__main__':
    configuracoes = [
        ("Config 1 - Default", DecisionTreeRegressor(random_state=42)),
        ("Config 2 - Shallow", DecisionTreeRegressor(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)),
        ("Config 3 - Medium", DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42)),
        ("Config 4 - Deep", DecisionTreeRegressor(max_depth=15, min_samples_split=2, min_samples_leaf=1, random_state=42)),
        ("Config 5 - Constrained", DecisionTreeRegressor(max_depth=20, min_samples_split=20, min_samples_leaf=10, random_state=42)),
    ]

    for nome_config, modelo in configuracoes:
        print(f"\n\n{'='*80}")
        print(f"=== {nome_config} ===")
        print(f"Parâmetros: {modelo.get_params()}")
        print(f"{'='*80}\n")

        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        
        print(f"[{nome_config}] 1. Rodando Validação Cruzada KFold (5 folds)...")
        scores = cross_validate(modelo, previsores_treinamento, y_treino, cv=cv_strategy, 
                                scoring=('r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'))

        print(f"\n--- MÉTRICAS NA VALIDAÇÃO CRUZADA DE TREINO ({nome_config}) ---")
        print(f"Score Médio CV (R²): {np.mean(scores['test_r2']):.4f}")

        # Teste final
        print(f"[{nome_config}] 2. Treinando modelo na base completa de treino e testando na base de teste...")
        modelo.fit(previsores_treinamento, y_treino)
        previsoes = modelo.predict(previsores_teste)
        
        score_teste = modelo.score(previsores_teste, y_teste)
        mae = metrics.mean_absolute_error(y_teste, previsoes)
        mse = metrics.mean_squared_error(y_teste, previsoes)
        rmse = np.sqrt(mse)

        print(f"\n--- MÉTRICAS FINAIS NA BASE DE TESTE ({nome_config}) ---")
        print(f"Score (R²): {score_teste:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")

        print(f"\n[{nome_config}] 3. Gerando gráfico de importância...")
        importances = modelo.feature_importances_
        indices = importances.argsort()[::-1]
        
        top_n = min(10, len(cols_previsores))
        top_features = np.array(cols_previsores)[indices[:top_n]]
        top_importances = importances[indices[:top_n]]

        plt.figure(figsize=(10, 6))
        plt.barh(top_features[::-1], top_importances[::-1], color='forestgreen')
        plt.xlabel("Importância da Feature")
        plt.title(f"Importância - Árvore de Decisão - {nome_config}")
        plt.tight_layout()
        
        os.makedirs('resultados', exist_ok=True)
        nome_arquivo = f"resultados/plot_Arvore_{nome_config.replace(' ', '_').replace(':', '')}.png"
        plt.savefig(nome_arquivo)
        plt.close()

    print(f"\n\n{'='*80}")
    print("Processamento de todas as 5 configurações concluído!")
    print(f"{'='*80}\n")
