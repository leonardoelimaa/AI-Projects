# -*- coding: utf-8 -*-

import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

"""## Configurar Chaves de API"""

GOOGLE_API_KEY = 'Minha chave de API'
genai.configure(api_key=GOOGLE_API_KEY)

"""# Case Prático: Correção de Dataset Desbalanceado Usando Gen AI

Somos a equipe de dados de uma fintech e nosso desafio é construir um modelo para detectar transações fraudulentas. O problema é que nosso dataset é extremamente desbalanceado: mais de 99% das transações são legítimas. Modelos de ML clássicos sofrem para aprender com tão poucos exemplos de fraude.

__Nossa Estratégia:__

1. Treinar um modelo baseline para provar que ele é ruim em detectar fraudes.
2. Usar o Gemini para gerar novos dados sintéticos de fraude.
3. Retreinar o modelo com os dados aumentados e comprovar a melhora.

## Carregando o Dataset de Fraude
"""

#Carregando o Dataset
url_fraud = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
df_fraud = pd.read_csv(url_fraud)

#Shape
df_fraud.shape

#Sample
df_fraud.sample(5)

"""## Análise Exploratória (EDA)"""

# Value Counts da variavel Class
df_fraud['Class'].value_counts()

# Normalizacao
df_fraud['Class'].value_counts(normalize=True) * 100

"""## Modelo Baseline: Treinando nos Dados Originais"""

# Regressao logistica
X = df_fraud.drop('Class', axis=1) # Variavel preditora: informações usadas para prever algo
y = df_fraud['Class'] # Saida: o que queremos que o modelo aprenda a prever
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 1. Selecionar modelo
model = LogisticRegression()

# 2. Treinar modelo
model.fit(X_train, y_train)

# 3. Fazer as predicoes
y_pred_baseline = model.predict(X_test)

# Report de classificacao
print(classification_report(y_test, y_pred_baseline))

# Matriz de confusao
sns.heatmap(confusion_matrix(y_test, y_pred_baseline), annot=True, fmt='d', cmap='Reds')

"""1. Precisão (Precision) A precisão responde à pergunta: "De todas as vezes que o modelo previu a classe 1, quantas ele acertou?"

É uma métrica de "qualidade" da previsão positiva. Uma alta precisão significa que, quando o modelo diz que é 1, ele tem uma alta probabilidade de estar certo.

Interpretação: De todas as previsões "positivas" (classe 1) que o modelo fez, 83.6% estavam corretas.

2. Recall (Revocação ou Sensibilidade) O recall responde à pergunta: "De todos os exemplos que eram realmente da classe 1, quantos o modelo conseguiu encontrar?"

É uma métrica de "quantidade" ou "abrangência". Um recall alto significa que o modelo é bom em encontrar todos os exemplos positivos existentes nos dados.

Interpretação: O modelo foi capaz de identificar 65.5% de todos os casos que realmente pertenciam à classe 1. Os outros 34.5% (os 51 Falsos Negativos) não foram detectados.

## Usando IA Generativa para Criar Dados Sintéticos
"""

# Pegando 5 exemplos de fraude do nosso dataset para mostrar ao LLM
df_fraudes_reais = X_train[y_train == 1].sample(5)

#Formatando os exemplos para o prompt (Few-Shot Prompting)
exemplos_texto = ''
#TODO Gerar exemplos
for i, row in df_fraudes_reais.iterrows():
  exemplos_texto += f'Exemplo de transação fraudulenta {i+1}:\n'
  exemplos_texto += str(row.to_dict()) + '\n\n'

prompt_geracao = f"""
Você é um especialista em ciência de dados simulando dados para um modelo de detecção de fraude.
Com base nos exemplos de transações fraudulentas abaixo, gere 10 novos exemplos de transações fictícias, mas realistas, que sigam um padrão similar.
Retorne apenas os dicionários de dados, um por linha, sem texto adicional.

{exemplos_texto}

Gere 10 novos exemplos aqui:
"""

model_gen = genai.GenerativeModel('gemini-2.5-flash')
response = model_gen.generate_content(prompt_geracao)

# Processando a resposta do LLM para transformá-la em um DataFrame
novas_fraudes = []
for line in response.text.strip().split('\n'):
    try:
        novas_fraudes.append(ast.literal_eval(line))
    except:
        continue # Ignora linhas mal formatadas

# Criando novo DataFrame
df_novas_fraudes = pd.DataFrame(novas_fraudes)

df_novas_fraudes['Class'] = 1

df_novas_fraudes

"""## Modelo Aprimorado: Treinando com os Dados Aumentados"""

print("\n--- 🚀 Treinando nosso Modelo Aprimorado com Dados Sintéticos ---")
X_train_aumentado = pd.concat([X_train, df_novas_fraudes.drop('Class', axis=1)], ignore_index=True)
y_train_aumentado = pd.concat([y_train, df_novas_fraudes['Class']], ignore_index=True)

model_melhorado = LogisticRegression()
model_melhorado.fit(X_train_aumentado, y_train_aumentado)
y_pred_melhorado = model_melhorado.predict(X_test)

"""## Comparando resultados"""

print("\n--- Resultado do Modelo Baseline ---")
print(classification_report(y_test, y_pred_baseline))

print("\n--- Resultado do Modelo Aprimorado ---")
print(classification_report(y_test, y_pred_melhorado))

sns.heatmap(confusion_matrix(y_test, y_pred_baseline), annot=True, fmt='d', cmap='Reds')
plt.title('Matriz de Confusão - Modelo Baseline')

sns.heatmap(confusion_matrix(y_test, y_pred_melhorado), annot=True, fmt='d', cmap='Greens')
plt.title('Matriz de Confusão - Modelo Aprimorado com GenAI')