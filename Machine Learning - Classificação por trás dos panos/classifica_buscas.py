from carrega_csv import carregar_buscas
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals.joblib import dump,load

df = pd.read_csv('O:\\Projetos Python\\Cursos Alura\\Machine Learning - Classificação por trás dos panos\\csv\\buscas.csv')
X_df = df[['home','busca','logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

percentual_treino = 0.9

tamanho_de_treino = int(percentual_treino*len(Y))
tamanho_de_teste = int(len(Y)-tamanho_de_treino)

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]

modelo = MultinomialNB()
modelo.fit(treino_dados,treino_marcacoes)

resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0* total_de_acertos/total_de_elementos
print(taxa_de_acerto)
print(total_de_acertos)
print(total_de_elementos)

#salva o modelo em disco
dump(modelo,'O:\\Projetos Python\\Cursos Alura\\Machine Learning - Classificação por trás dos panos\\modelo.joblib')
#Carrega o modelo salvo
modelo2 = load('O:\\Projetos Python\\Cursos Alura\\Machine Learning - Classificação por trás dos panos\\modelo.joblib')
#Testa o mesmo set com o modelo carregado
resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0* total_de_acertos/total_de_elementos
print('Modelo Carregado:')
print(taxa_de_acerto)
print(total_de_acertos)
print(total_de_elementos)

#Testa a qualidade do modelo com algoritmo burro

acerto_de_um = sum(Y)
acerto_de_um = len(Y[Y==1])
acerto_de_zero = len(Y) - acerto_de_um
acerto_de_zero = len(Y[Y==0])
taxa_de_acerto_base = 100.0 * max(acerto_de_um,acerto_de_zero)/len(Y)
print("Taxa de acerto base: %f"% taxa_de_acerto_base)
print("Taxa de acerto do modelo: %f"% taxa_de_acerto)
#832. Se apenas chutássemos que Y seria 1 para todas as vezes, acertaríamos 83,2% das vezes. Melhor que o modelo calculado e menos complexo.

