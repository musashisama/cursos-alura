#Aula 2 do curso introdução a Machine Learning
import csv

def carregar_acessos():

    X = []
    Y = []

    arquivo = open('O:\\Projetos Python\\Cursos Alura\\Machine Learning - Classificação por trás dos panos\\csv\\acesso.csv', 'r+')
    leitor = csv.reader(arquivo)
    next(leitor)   

    for acessou_home, acessou_como_funciona,acessou_contato,comprou in leitor:

        dado = [int(acessou_home),int(acessou_como_funciona),int(acessou_contato)]
        X.append(dado)
        Y.append(int(comprou))

    return X,Y

def carregar_buscas():
    X = []
    Y = []

    arquivo = open('O:\\Projetos Python\\Cursos Alura\\Machine Learning - Classificação por trás dos panos\\csv\\buscas.csv', 'r+')
    leitor = csv.reader(arquivo)
    next(leitor)

    for home, busca, logado, comprou in leitor:
        dado = [int(home), busca,int(logado)]
        X.append(dado)
        Y.append(int(comprou))
    
    return X,Y