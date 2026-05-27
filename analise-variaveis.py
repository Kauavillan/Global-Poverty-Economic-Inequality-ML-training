import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('resultados', exist_ok=True)

print("=== TAREFA 1: Análise Exploratória de Variáveis ===\n")

# print("1. Executando pre-processamento.py para carregar a base e variáveis processadas...")
# with open('pre-processamento.py', 'r', encoding='utf-8') as f:
#     exec(f.read())

# ---------------------------------------------------------
#? Tabela descrevendo cada atributo (usando a base original do script)
# ---------------------------------------------------------
print("2. Gerando tabela de descrição das variáveis (Tarefa 1)...")
info_variaveis = []

for coluna in base.columns:
    tipo = base[coluna].dtype
    if pd.api.types.is_numeric_dtype(tipo):
        if base[coluna].nunique() > 20:
            tipo_desc = "Numérico Contínuo"
        else:
            tipo_desc = "Numérico Discreto"
        valores = f"Min: {base[coluna].min()} | Max: {base[coluna].max()}"
    else:
        tipo_desc = "Categórico"
        valores = f"Categorias: {base[coluna].nunique()}"
        if base[coluna].nunique() <= 10:
            valores += f" ({', '.join(base[coluna].dropna().unique().astype(str))})"
            
    info_variaveis.append({
        'Atributo': coluna,
        'Tipo': tipo_desc,
        'Valores Possíveis / Range': valores
    })

df_info = pd.DataFrame(info_variaveis)
df_info.to_csv('resultados/tabela_variaveis.csv', index=False, encoding='utf-8-sig')
print("   -> Tabela descritiva salva em 'resultados/tabela_variaveis.csv'")

# ---------------------------------------------------------
#? Matriz de Correlação e Mapa de Calor
# ---------------------------------------------------------
print("\n3. Gerando Matriz de Correlação e Heatmap...")

# Dataframe
df_previsores = pd.DataFrame(previsores_treinamento, columns=cols_previsores)
df_objetivo = pd.DataFrame(objetivo_treinamento, columns=cols_objetivo)
df_completo = pd.concat([df_previsores, df_objetivo], axis=1)

plt.figure(figsize=(20, 16)) # Tamanho maior por conta das variáveis dummy criadas
matriz_corr = df_completo.corr()

# Mapa de Calor
sns.heatmap(matriz_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.1)
plt.title("Heatmap de Correlação - Variáveis Numéricas", fontsize=16)
plt.tight_layout()

caminho_heatmap = 'resultados/heatmap_correlacao.png'
plt.savefig(caminho_heatmap, dpi=300)
print(f"   -> Gráfico de calor salvo em '{caminho_heatmap}'")
plt.close()

# ---------------------------------------------------------
#? Identificando Fortes Correlações
# ---------------------------------------------------------
limite = 0.75
print(f"\n4. Filtrando variáveis com forte correlação (acima de {limite} ou abaixo de {-limite})...")

correlacoes_fortes = []
for i in range(len(matriz_corr.columns)):
    for j in range(i):
        valor_corr = matriz_corr.iloc[i, j]
        if abs(valor_corr) > limite:
            colname_i = matriz_corr.columns[i]
            colname_j = matriz_corr.columns[j]
            correlacoes_fortes.append((colname_i, colname_j, valor_corr))

# Salva um relatório de correlações na pasta de resultados
with open('resultados/relatorio_correlacoes.txt', 'w', encoding='utf-8') as f:
    f.write("====================================================\n")
    f.write(" RELATÓRIO DE FORTES CORRELAÇÕES ENTRE VARIÁVEIS\n")
    f.write(f" (Limite absoluto considerado: {limite})\n")
    f.write("====================================================\n\n")
    
    if correlacoes_fortes:
        for c1, c2, valor in sorted(correlacoes_fortes, key=lambda x: abs(x[2]), reverse=True):
            mensagem = f"[{c1}]  <--->  [{c2}]  (Correlação: {valor:.4f})"
            print(f"   ! {mensagem}")
            f.write(mensagem + "\n")
    else:
        mensagem = f"Nenhuma correlação forte ({limite} / -{limite}) foi encontrada."
        print(f"   i {mensagem}")
        f.write(mensagem + "\n")

print("   -> Relatório de variáveis salvas em 'resultados/relatorio_correlacoes.txt'")
print(f"\n{'='*80}")
print("Análise de Variáveis finalizada e dados armazenados na pasta 'resultados/'")
print(f"{'='*80}\n")