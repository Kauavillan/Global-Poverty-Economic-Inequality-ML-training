################################################################################
# Trabalho 2 - Regressão com Redes Neurais (MLPRegressor)
# Testes Múltiplos, Validação Cruzada, Gráficos Diagnósticos e Importância
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# Toggle para testar múltiplas possibilidades (GridSearch, Seleção de Variáveis) ou apenas rodar a melhor configuração do modelo.
TESTAR_POSSIBILIDADES = True

# =============================================================================
# 1. Importação do Pré-Processamento do Professor
# =============================================================================
#print("Executando pre-processamento.py...")
#os.makedirs('resultados', exist_ok=True)

#with open('pre-processamento.py', 'r', encoding='utf-8') as f:
#    exec(f.read())

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
            'hidden_layer_sizes': [(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10)],
            'activation': ['relu', 'tanh'],
            'max_iter': [1000],
            'random_state': [42]
        }

        print(f"[{nome_teste}] 1. Treinando e otimizando parâmetros em 75% da base...")
        grid_search = GridSearchCV(estimator=MLPRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv_strategy, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        melhor_modelo = grid_search.best_estimator_
        print(f"[{nome_teste}] Melhor configuração: {grid_search.best_params_}")
    else:
        print(f"[{nome_teste}] 1. Treinando modelo com a melhor configuração predefinida...")
        melhor_modelo = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', max_iter=500, random_state=42)

    print(f"[{nome_teste}] 2. Rodando Validação Cruzada KFold...")
    scores = cross_validate(melhor_modelo, X_train, y_train, cv=cv_strategy, scoring=('r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'))

    print(f"\n--- MÉTRICAS NA VALIDAÇÃO CRUZADA DE TREINO ({nome_teste}) ---")
    print(f"Score Médio CV: {np.mean(scores['test_r2']):.4f}")

    # 3. Teste final na base de teste
    melhor_modelo.fit(X_train, y_train)
    previsoes = melhor_modelo.predict(X_test)
    
    print(f"\n--- MÉTRICAS FINAIS NA BASE DE TESTE ({nome_teste}) ---")
    print(f"Score: {melhor_modelo.score(X_test, y_test):.4f}")
    print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, previsoes):.4f}")
    print(f"Mean Squared Error: {metrics.mean_squared_error(y_test, previsoes):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, previsoes)):.4f}")

    top_features = []
    if testar_possibilidades:
        print(f"\n[{nome_teste}] 3. Analisando as variáveis mais cruciais (Permutation Importance)...")
        resultado_imp = permutation_importance(melhor_modelo, X_train, y_train, n_repeats=5, random_state=42, n_jobs=-1)
        indices = resultado_imp.importances_mean.argsort()[::-1]
        
        top_n = min(10, len(nomes_colunas))
        top_features = np.array(nomes_colunas)[indices[:top_n]]
        top_importances = resultado_imp.importances_mean[indices[:top_n]]

        for i in range(len(top_features)):
            print(f"    {i+1}º - {top_features[i]}: {top_importances[i]:.4f}")

        # Geração Automática de Gráfico de Importância
        plt.figure(figsize=(10, 6))
        plt.barh(top_features[::-1], top_importances[::-1], color='darkorange')
        plt.xlabel("Permutation Importance")
        plt.title(f"Importância de Features (Redes Neurais) - {nome_teste}")
        plt.tight_layout()
        
        nome_arquivo = f"resultados/plot_NN_{nome_teste.replace(' ', '_').replace(':', '')}.png"
        plt.savefig(nome_arquivo)
        print(f"[{nome_teste}] 4. Gráfico gerado com sucesso: '{nome_arquivo}'")
        plt.close()

    # Gráficos Diagnósticos são gerados independentemente de ser ou não modo de teste
    print(f"[{nome_teste}] 5. Gerando gráficos de diagnóstico (Predito vs Real e Resíduos)...")
    y_pred_cv = cross_val_predict(melhor_modelo, X_train, y_train, cv=cv_strategy)
    residuos = y_train - y_pred_cv

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_cv, alpha=0.6, color='darkorange', edgecolors='k')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Predição Perfeita')
    plt.xlabel('Valores Reais (Alvo)')
    plt.ylabel('Valores Preditos pelo Modelo')
    plt.title(f'Predito vs Real - {nome_teste}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(residuos, bins=25, color='darkorange', edgecolor='k', alpha=0.7)
    plt.axvline(0, color='r', linestyle='dashed', linewidth=2, label='Zero Erro')
    plt.xlabel('Erro (Resíduo)')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Erros (Resíduos)')
    plt.legend()

    plt.tight_layout()
    diag_arquivo = f"resultados/diag_NN_{nome_teste.replace(' ', '_').replace(':', '')}.png"
    plt.savefig(diag_arquivo)
    print(f"    -> Diagnósticos salvos em '{diag_arquivo}'")
    plt.close()

    return melhor_modelo, list(top_features)

# =============================================================================
# 3. Execução Sequencial dos Experimentos
# =============================================================================
if __name__ == '__main__':
    # --- TESTE 1 ---
    _, top_features_t1 = executar_teste(
        previsores_treinamento, y_treino, 
        previsores_teste, y_teste, 
        cols_previsores, "Teste 1 - Todas as Variaveis",
        testar_possibilidades=TESTAR_POSSIBILIDADES
    )

    # --- TESTE 2 ---
    if TESTAR_POSSIBILIDADES and top_features_t1:
        qtd_top = min(5, len(top_features_t1))
        indices_top = [list(cols_previsores).index(col) for col in top_features_t1[:qtd_top]]
        
        X_treino_top = previsores_treinamento[:, indices_top]
        X_teste_top = previsores_teste[:, indices_top]
        
        executar_teste(
            X_treino_top, y_treino, 
            X_teste_top, y_teste, 
            top_features_t1[:qtd_top], 
            f"Teste 2 - Apenas as TOP {qtd_top} Variaveis",
            testar_possibilidades=TESTAR_POSSIBILIDADES
        )

    print(f"\n\n{'='*80}")
    print("Processamento concluído com sucesso!")
    print(f"{'='*80}\n")
