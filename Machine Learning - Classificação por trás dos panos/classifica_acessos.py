from lesson1_carrega_csv import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

X,Y = carregar_acessos()

# Separar 90% para treino e 10% para teste. Resultado 88.89% de acerto.

treino_dados = X[:90]
treino_marcacoes = Y[:90]

teste_dados = X[-9:]
teste_marcacoes = Y[-9:]



modelo = MultinomialNB()
modelo.fit(treino_dados,treino_marcacoes)

print(modelo.predict([[1,0,1]]))


resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes
acertos = [d for d in diferencas if d == 0]
print(acertos)
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0* total_de_acertos/total_de_elementos
print(taxa_de_acerto)
print(total_de_acertos)
print(total_de_elementos)