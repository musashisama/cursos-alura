from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

pathStatic = Path('D:\\Git\\cursos\\Machine Learning 2 - Mais Classificacoes\\static')

df = pd.read_csv(pathStatic/'situacao_do_cliente.csv')

X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values 

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
#tamanho_de_validacao = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]



def fit_and_predict(nome, modelo,treino_dados,treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo,treino_dados, treino_marcacoes,cv=k)
    taxa_de_acerto  = np.mean(scores)
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

resultados = {}

modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0, max_iter=5000))
resultadoOneVsRest = fit_and_predict("OneVsRest",modeloOneVsRest,treino_dados,treino_marcacoes,)
resultados[resultadoOneVsRest] = modeloOneVsRest

modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0,max_iter=5000))
resultadoOneVsOne = fit_and_predict("OneVsOne",modeloOneVsOne,treino_dados,treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne


modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados,treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial


modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados,treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

print(resultados)

print(resultados)
maximo = max(resultados)
vencedor = resultados[maximo]
print("Vencedor: ", vencedor)

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    vencedor.fit(treino_dados,treino_marcacoes)
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
    print(msg)

teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)
