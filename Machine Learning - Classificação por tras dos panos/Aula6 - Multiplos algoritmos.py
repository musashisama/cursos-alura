from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from collections import Counter
from pathlib import Path


pathStatic = Path('D:/Git/cursos/Machine Learning - Classificação por tras dos panos/static')
print(pathStatic)
df = pd.read_csv(pathStatic/'busca.csv')

def fit_and_predict(nome, modelo, treino_dados,treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados,treino_marcacoes)
    
    resultado = modelo.predict(teste_dados)
    acertos = (resultado == teste_marcacoes)
    
    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    
    msg = "Taxa de acerto do {0} (teste): {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.8
porcentagem_teste = 0.1

tamanho_de_treino = int(porcentagem_treino * len(Y))
tamanho_de_teste = int(porcentagem_teste * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

fim_de_teste = tamanho_de_treino + tamanho_de_teste

teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]

modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier",modeloAdaBoost,treino_dados,treino_marcacoes,teste_dados,teste_marcacoes)

modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB",modeloMultinomial,treino_dados,treino_marcacoes,teste_dados,teste_marcacoes)

if resultadoMultinomial > resultadoAdaBoost:
    vencedor = modeloMultinomial
else:
    vencedor = modeloAdaBoost

resultado = vencedor.predict(validacao_dados)
acertos = (resultado == teste_marcacoes)    
total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real (validação): {0}".format(taxa_de_acerto)
print(msg)
