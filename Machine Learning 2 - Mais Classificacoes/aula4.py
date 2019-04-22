#!-*- coding: utf-8 -*-
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# texto1 = "Se eu comprar cinco anos antecipados, eu ganho algum desconto?"
# texto2 = "O exercício 15 do curso de Java 1 está com a resposta errada. Pode conferir pf?"
# texto3 = "Existe algum curso para cuidar do marketing da minha empresa?"
pathStatic = Path('D:\\Git\\cursos\\Machine Learning 2 - Mais Classificacoes\\static')
classificacoes = pd.read_csv(pathStatic/'emails.csv')

textosPuros = classificacoes['email']
textosQuebrados = textosPuros.str.lower().str.split(' ')
dicionario = set()

for lista in textosQuebrados:
    dicionario.update(lista)    

totalDePalavras = len(dicionario)
tuplas = zip(dicionario,range(totalDePalavras))
tradutor = {palavra:indice for palavra,indice in tuplas}

def vetorizar_texto(texto,tradutor):
    vetor= [0]*len(tradutor)
    for palavra in texto:
        if palavra in tradutor:
            posicao = tradutor[palavra]
            vetor[posicao] += 1
    return vetor

vetoresDeTexto = [vetorizar_texto(texto,tradutor) for texto in textosQuebrados]
marcas = classificacoes['classificacao']

X = np.array(vetoresDeTexto)
Y = np.array(marcas.tolist())

porcentagem_de_treino = 0.8
tamanho_do_treino = int(porcentagem_de_treino*len(Y))
tamanho_de_validacao = len(Y) - tamanho_do_treino
treino_dados = X[0:tamanho_do_treino]
treino_marcacoes = Y[0:tamanho_do_treino]
validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]

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

modeloAdaBoost = AdaBoostClassifier(random_state=0)
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados,treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

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