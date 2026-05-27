################################################################################
# Trabalho 2 - Regressão com Support Vector Machines (SVR)
# Testes Múltiplos, Análise de Correlação e Exportação de Gráficos
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
import warnings

# Suprimir avisos para manter o console limpo durante as múltiplas execuções
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Funções de Auxílio (Encapsulamento de Padrões)
# =============================================================================

def carregar_e_preparar_dados(caminho_arquivo):
    print(f"Lendo e processando a base de dados ({caminho_arquivo})...")
    base = pd.read_csv(caminho_arquivo)
    base = base.drop(columns=['record_id'], errors='ignore')

    col_alvo = 'female_labor_participation_pct'
    X = base.drop(columns=[col_alvo]).copy()
    y = base[col_alvo].copy()

    # Codificação de Variáveis
    le_income = LabelEncoder()
    le_income.classes_ = np.array(['Low Income', 'Lower-Middle Income', 'Upper-Middle Income', 'High Income'])
    X['income_group'] = le_income.transform(X['income_group']).astype('int64')
    X = pd.get_dummies(X, columns=['country', 'region'], dtype='int64')
    
    return X, y

def verificar_correlacoes(X, limite=0.8):
    print("\nVerificando possíveis problemas no pré-processamento (Variáveis Altamente Correlacionadas)...")
    numeric_X = X.select_dtypes(include=[np.number])
    matriz_corr = numeric_X.corr().abs()
    
    # Retira a diagonal principal para não contar a correlação da variável com ela mesma
    matriz_superior = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
    cols_para_remover = [coluna for coluna in matriz_superior.columns if any(matriz_superior[coluna] > limite)]
    
    if cols_para_remover:
        print(f"[!] CUIDADO: Variáveis encontradas com correlação de Pearson > {limite}: {cols_para_remover}")
        print("    -> Podem gerar distorções (multicolinearidade).")
    else:
        print(f"[i] Nenhum par de variáveis excede o limite de {limite}.")
        
    return cols_para_remover

def executar_teste(X, y, nome_teste):
    """
    Agrupa a rotina inteira: pipeline, tuning, validação, relatório e plotagem de gráfico.
    """
    print(f"\n\n{'='*80}")
    print(f"=== TESTE: {nome_teste} ===")
    print(f"{'='*80}\n")

    # Pipeline Robusto de Escalonamento e Modelo
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    modelo_final = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    # Configurações do modelo
    param_grid = {
        'regressor__svr__kernel': ['rbf', 'linear'],
        'regressor__svr__C': [0.1, 1.0, 10.0],
        'regressor__svr__gamma': ['scale'],
        'regressor__svr__epsilon': [0.01, 0.1, 1.0]
    }

    print(f"[{nome_teste}] 1. Treinando e otimizando parâmetros (GridSearch)...")
    grid_search = GridSearchCV(estimator=modelo_final, param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv_strategy, n_jobs=-1)
    grid_search.fit(X, y)
    melhor_modelo = grid_search.best_estimator_
    
    print(f"[{nome_teste}] Melhor configuração: {grid_search.best_params_}")

    print(f"[{nome_teste}] 2. Rodando Validação Cruzada KFold...")
    scores = cross_validate(melhor_modelo, X, y, cv=cv_strategy, scoring=('r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'))

    print(f"\n--- MÉTRICAS FINAIS ({nome_teste}) ---")
    print(f"R² Score:  {np.mean(scores['test_r2']):.4f} (+/- {np.std(scores['test_r2']):.4f})")
    print(f"MAE:       {-np.mean(scores['test_neg_mean_absolute_error']):.4f}")
    print(f"MSE:       {-np.mean(scores['test_neg_mean_squared_error']):.4f}")
    print(f"RMSE:      {-np.mean(scores['test_neg_root_mean_squared_error']):.4f}")

    print(f"\n[{nome_teste}] 3. Analisando as variáveis mais cruciais para este teste...")
    resultado_imp = permutation_importance(melhor_modelo, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    indices = resultado_imp.importances_mean.argsort()[::-1]
    
    top_n = min(10, len(X.columns))
    top_features = X.columns[indices[:top_n]]
    top_importances = resultado_imp.importances_mean[indices[:top_n]]

    for i in range(len(top_features)):
        print(f"    {i+1}º - {top_features[i]}: {top_importances[i]:.4f}")

    # 4. Geração Automática de Gráfico
    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_importances[::-1], color='royalblue')
    plt.xlabel("Permutation Importance")
    plt.title(f"Importância de Features - {nome_teste}")
    plt.tight_layout()
    
    nome_arquivo = f"plot_SVR_{nome_teste.replace(' ', '_').replace(':', '')}.png"
    plt.savefig(nome_arquivo)
    print(f"[{nome_teste}] 4. Gráfico gerado com sucesso: '{nome_arquivo}'")
    plt.close()

    return melhor_modelo, list(top_features)


# =============================================================================
# 2. Execução Sequencial dos Experimentos
# =============================================================================
if __name__ == '__main__':
    # Carga Inicial
    X_completo, y = carregar_e_preparar_dados('datasets/dataset.csv')

    # Análise Exploratória Rápida no script
    cols_para_remover = verificar_correlacoes(X_completo, limite=0.8)

    # --- TESTE 1 ---
    # Baseline usando toda a complexidade dos dados (todas as features)
    _, top_features_t1 = executar_teste(X_completo, y, "Teste 1 - Todas as Variaveis")

    # --- TESTE 2 ---
    # Retira ruído de pré-processamento (variáveis fortemente colineares)
    if cols_para_remover:
        X_sem_ruido = X_completo.drop(columns=cols_para_remover)
        executar_teste(X_sem_ruido, y, "Teste 2 - Sem Variaveis Muito Correlacionadas")
    else:
        print("\n\n[i] Pulando o 'Teste 2', pois não há dependência grave apontada entre as features.")

    # --- TESTE 3 ---
    # Tentar forçar um modelo mais simples (menos recursos gastos e maior explicabilidade)
    qtd_top = min(5, len(top_features_t1))
    X_top = X_completo[top_features_t1[:qtd_top]]
    executar_teste(X_top, y, f"Teste 3 - Apenas as TOP {qtd_top} Variaveis")

    print(f"\n\n{'='*80}")
    print("Processamento total e geração de gráficos concluídos com sucesso!")
    print(f"{'='*80}\n")