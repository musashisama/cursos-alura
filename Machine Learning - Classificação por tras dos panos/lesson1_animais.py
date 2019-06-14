from sklearn.naive_bayes import MultinomialNB

# [É gordinho?, Tem pernas curtas?, Late?]
porco1 = [1,1,0]
porco2 = [1,1,0]
porco3 = [1,1,0]
cachorro1 = [1,1,1]
cachorro2 = [0,1,1]
cachorro3 = [0,1,1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

marcacoes = [1, 1, 1,-1,-1,-1]

misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [1, 0, 1]

teste = [misterioso1, misterioso2, misterioso3]

marcacoes_teste = [-1,1,1]

modelo = MultinomialNB()
modelo.fit(dados,marcacoes)

resultado = modelo.predict(teste)
print(resultado)
#Direfença entre o resulktado e as marcacoes de teste. Se zero, acertou. Caso +2 ou -2, errou.
diferencas = resultado - marcacoes_teste
print(diferencas)

acertos = [d for d in diferencas if d==0]
print (acertos)
total_de_acertos = len(acertos)
print(total_de_acertos)
total_de_elementos = len(teste)

taxa_de_acerto = 100 * total_de_acertos / total_de_elementos
print('Taxa de Acerto = '+ str(taxa_de_acerto) +'%')
