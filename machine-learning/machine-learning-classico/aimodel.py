import pandas as pd # Para manipular os dados em formato de tabela
from sklearn import datasets # Para baixar o conjunto de dados pronto.
from sklearn.model_selection import train_test_split # Para dividir os dados em treino e teste.
from sklearn.preprocessing import StandardScaler # Para padronizar a escala dos números.

from sklearn.linear_model import LinearRegression # O algoritmo matemático que será usado.

housing = datasets.fetch_california_housing()
housing

df = pd.DataFrame(housing.data, columns=housing.feature_names) # Features (características) será o X

dfTarget = pd.DataFrame(housing.target, columns=housing.target_names) # Target (o alvo) será o y

# O computador tentará aprender como as características em X influenciam o valor em y
X = df
y = dfTarget

"""# Preparando os dados"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) # Separa 20% dos dados para o "exame final" (teste) e usa 80% para ensinar o modelo (treino)
X_train

# A Regressão Linear funciona melhor quando os números estão na mesma "escala"
scaler = StandardScaler()

X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test)
# Usa-se fit_transform no treino (para aprender a média/desvio dos dados de treino) e apenas transform no teste. Isso evita que informações do teste "vazem" para o treino.

"""# Aplicando o modelo"""

reg = LinearRegression().fit(X_train_Scaled, y_train)

"""O modelo analisa os dados de treino (X_train_Scaled) e as respostas corretas (y_train) para encontrar uma equação matemática (uma linha/hiperplano) que melhor se ajusta aos dados."""

results = reg.predict(X_test_Scaled)

results

"""Agora que o modelo "estudou", ele recebe os dados de teste (que ele nunca viu antes) e tenta adivinhar os valores das casas."""

reg.score(X_test_Scaled, y_test)