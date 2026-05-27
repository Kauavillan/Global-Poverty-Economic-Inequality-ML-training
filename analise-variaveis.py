################################################################################
# Trabalho 2 - Tarefa 1: Análise de Variáveis
# Geração automatizada da tabela de dicionário de dados
################################################################################

import pandas as pd

def gerar_tabela_variaveis(caminho_arquivo):
    print(f"Lendo base de dados em: {caminho_arquivo}\n")
    base = pd.read_csv(caminho_arquivo)
    
    tabela_resultado = []
    
    for coluna in base.columns:
        # Ignorando o record_id conforme feito nos outros scripts
        if coluna == 'record_id':
            continue
            
        tipo_pandas = str(base[coluna].dtype)
        
        # Classificação baseada no tipo de dado
        if 'int' in tipo_pandas:
            classificacao = 'Numérico (Discreto)'
        elif 'float' in tipo_pandas:
            classificacao = 'Numérico (Contínuo)'
        else:
            # Identificando se é ordinal pelo conhecimento prévio do negócio
            if coluna == 'income_group':
                classificacao = 'Categórico (Ordinal)'
            else:
                classificacao = 'Categórico (Nominal)'
                
        # Extraindo os valores que podem assumir (Range ou Classes)
        num_valores_unicos = base[coluna].nunique()
        if classificacao.startswith('Categórico') or num_valores_unicos < 10:
            valores_assumidos = ", ".join(map(str, base[coluna].dropna().unique()))
        else:
            valores_assumidos = f"Min: {base[coluna].min()} | Max: {base[coluna].max()}"
            
        tabela_resultado.append({
            'Atributo (Feature)': coluna,
            'Classificação': classificacao,
            'Valores que pode assumir': valores_assumidos
        })
        
    df_tabela = pd.DataFrame(tabela_resultado)
    
    # Exportando os resultados
    df_tabela.to_csv('tabela_variaveis.csv', index=False, encoding='utf-8')
    df_tabela.to_markdown('tabela_variaveis.md', index=False)
    print("[!] Sucesso! Arquivos 'tabela_variaveis.csv' e 'tabela_variaveis.md' gerados na raiz do projeto.")

if __name__ == "__main__":
    gerar_tabela_variaveis('datasets/dataset.csv')