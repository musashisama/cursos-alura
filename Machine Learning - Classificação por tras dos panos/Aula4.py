import pandas as pd
from collections import Counter
from pathlib import Path


pathStatic = Path('D:/Git/cursos/Machine Learning - Classificação por tras dos panos/static')
print(pathStatic)
df = pd.read_csv(pathStatic/'busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.9

tamanho_de_treino = int(porcentagem_treino * len(Y))
tamanho_de_teste = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
acertos = (resultado == teste_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
print("<<<Multinomial Naive-Bayes>>>")
print("Taxa de acerto do algoritmo: %f" % taxa_de_acerto)
print(total_de_elementos)

# a eficácia do algoritmo que chuta tudo um único valor
acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

from sklearn.ensemble import AdaBoostClassifier

modelo = AdaBoostClassifier()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
acertos = (resultado == teste_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
print("<<<AdaBoost>>")
print("Taxa de acerto do algoritmo: %f" % taxa_de_acerto)
print(total_de_elementos)

# a eficácia do algoritmo que chuta tudo um único valor
acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

def fit_and_predict(nome, modelo, treino_dados,treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados,treino_marcacoes)
    
    resultado = modelo.predict(teste_dados)
    acertos = (resultado == teste_marcacoes)
    
    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)

from sklearn.ensemble import AdaBoostClassifier
modelo = AdaBoostClassifier()
fit_and_predict("AdaBoostClassifier",modelo,treino_dados,treino_marcacoes,teste_dados,teste_marcacoes)

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
fit_and_predict("MultinomialNB",modelo,treino_dados,treino_marcacoes,teste_dados,teste_marcacoes)
