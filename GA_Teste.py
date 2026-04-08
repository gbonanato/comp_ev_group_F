''' UNIVERSIDADE FEDERAL DE MINAS GERAIS - UFMG
Programa de Pós-Graduação em Engenharia Elétrica - PPGEE
Trabalho: Estudo e Desenvolvimento de uma Ferramenta Computacional 
          Baseada em Algoritmo Genético -AG
PRoblema: Otimização e solução com o uso de um Algoritmo Genético.
Disciplina: Otimização em Engenharia Elétrica
Professor: Rodiney Rezende Saldanha  
Aluno: Vanderley Matias da Silva - 2026673882
--------------------------------------------------------------------------- '''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#--------------------------------------------------------------------------
# Parâmetros do Algoritmo Genético conforme tabela 8
POP_SIZE = 40          # Tamanho da população entre 10 e 100
DIM = 2                # Tamanho do cromossomo
GERACOES = 40          # Número máximo de gerações entre 10 e 50
CROSSOVER_RATE = 0.7   # Taxa de cruzamento (70%) entre 60% e 80%
MUTATION_RATE = 0.03   # Taxa de mutação (3%) entre 1% e 5%
ELITISMO = 0.6         # Taxa de elitismo entre 55% e 75%
#--------------------------------------------------------------------------
# Funções objetivo

def rastrigin(x, y):
    A = 10
    return A*2 + (x**2 - A*np.cos(2*np.pi*x)) + (y**2 - A*np.cos(2*np.pi*y))

def ackley(x, y):
    return (-20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2)))
            - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))
            + 20 + np.e)

def peaks(x, y):
    return (3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)
           -10*((x/5)-(x**3)-(y**5))*np.exp(-x**2-y**2)
           -(1/3)*np.exp(-(x+1)**2-y**2))
#--------------------------------------------------------------------------
# Algoritmo Genético (retorna também a população final)
def executar_ag(func_name): # define qual função será utilizada

    if func_name == 'rastrigin':  
        func = lambda v: rastrigin(v[0], v[1]) # cálculo do valor da função na coordenada [x, y]
        LIM_INF, LIM_SUP = -5.12, 5.12 # limites do domínio de busca 
    elif func_name == 'ackley':
        func = lambda v: ackley(v[0], v[1]) # cálculo do valor da função na coordenada [x, y]
        LIM_INF, LIM_SUP = -35, 35 # limites do domínio de busca 
    else:
        func = lambda v: peaks(v[0], v[1]) # cálculo do valor da função na coordenada [x, y]
        LIM_INF, LIM_SUP = -3, 3 # limites do domínio de busca 
#--------------------------------------------------------------------------
    # Inicialização da população
    pop = np.random.uniform(LIM_INF, LIM_SUP, (POP_SIZE, DIM))
    historico = []       # Armazena a trajetória (x,y) do melhor indivíduo
    fitness_hist = []    # Armazena melhor fitness por geração
#--------------------------------------------------------------------------
    # Seleção por torneio
    def selecao(pop, fitness):
        i, j = np.random.randint(0, len(pop), 2)
        return pop[i] if fitness[i] < fitness[j] else pop[j]
#--------------------------------------------------------------------------
    # Cruzamento
    def crossover(p1, p2):
        if np.random.rand() < CROSSOVER_RATE:
            alpha = np.random.rand()
            return alpha*p1 + (1-alpha)*p2
        return p1.copy()
#--------------------------------------------------------------------------
    # Mutação - 
    def mutacao(ind):
        for i in range(len(ind)):
            if np.random.rand() < MUTATION_RATE:
                ind[i] += np.random.normal(0, 0.5) # distribuição gaussiana - (média, desvio padrão)
                                                   # cria pertubação - ruído - necessário para 
                                                   # introduzir diversidade. Evita manter a população
                                                   # em mínimos locais.
        return np.clip(ind, LIM_INF, LIM_SUP) # garante que cada coordenada do indivíduo não saia 
                                              # do limite de domínio da função objetivo  
#--------------------------------------------------------------------------
    # Loop evolutivo
    for g in range(GERACOES):
        fitness = np.array([func(ind) for ind in pop])

        # Ordenação (melhor primeiro)
        idx = np.argsort(fitness)
        pop = pop[idx]
        fitness = fitness[idx]

        melhor = pop[0]
        melhor_fit = fitness[0]

        historico.append(melhor[:2])
        fitness_hist.append(melhor_fit)

        print(f"[{func_name.upper()}] Geração {g+1:02d} | Melhor Fitness: {melhor_fit:.6f} | x={melhor[0]:.4f}, y={melhor[1]:.4f}")
#--------------------------------------------------------------------------
        # Elitismo
        n_elite = int(ELITISMO * POP_SIZE)
        nova_pop = list(pop[:n_elite])
#--------------------------------------------------------------------------
        # Reprodução
        while len(nova_pop) < POP_SIZE:
            p1 = selecao(pop, fitness)
            p2 = selecao(pop, fitness)
            filho = crossover(p1, p2)
            filho = mutacao(filho)
            nova_pop.append(filho)
        pop = np.array(nova_pop)

    return np.array(historico), np.array(fitness_hist), func, LIM_INF, LIM_SUP, pop
#--------------------------------------------------------------------------
# Análise de estabilidade - Avalia a consistência e robustez
# Contexto: Se o algoritmo for rodado por diversas vezes resultados semelhantes podem 
#           ser esperados?
N_EXEC = 30
resultados = []
for i in range(N_EXEC):
    _, fitness_hist, _, _, _, _ = executar_ag('rastrigin')
    resultados.append(fitness_hist)

resultados = np.array(resultados)          # shape (N_EXEC, GERACOES)
media = np.mean(resultados, axis=0)
desvio_padrao = np.std(resultados, axis=0)
#--------------------------------------------------------------------------
# Curva média com faixa de ±1 desvio padrão / Desvio padrão baixo = algoritmo confiável
plt.figure()
plt.plot(media, label='Média do fitness')
plt.fill_between(range(len(media)),
                 media - desvio_padrao,
                 media + desvio_padrao,
                 alpha=0.3, label='±1 desvio padrão')
plt.title("Convergência média e variabilidade (Rastrigin)")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.legend()
plt.show()
#--------------------------------------------------------------------------
# Boxplot do fitness final (última geração) das 30 execuções
plt.figure()
plt.boxplot(resultados[:, -1])
plt.title("Distribuição do fitness final (30 execuções)")
plt.ylabel("Fitness")
plt.show()
#--------------------------------------------------------------------------
# Para os gráficos de dispersão e KDE, executamos mais uma vez e obtemos a população final
# KDE - para áreas mais escuras = maior densidade de indivíduos
_, _, _, _, _, pop_final = executar_ag('rastrigin')
# Visualização da densidade da população final (KDE)
sns.kdeplot(x=pop_final[:,0], y=pop_final[:,1], fill=True)
plt.title("Densidade da população final")
plt.show()

# Exibição de todos os indivíduos da população final
plt.scatter(pop_final[:,0], pop_final[:,1])
plt.title("População final")
plt.show()

#--------------------------------------------------------------------------
# Plots 2D E 3D
def plot_2d(func_name):
    hist, _, func, lim_inf, lim_sup, _ = executar_ag(func_name)

    x = np.linspace(lim_inf, lim_sup, 100)
    y = np.linspace(lim_inf, lim_sup, 100)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])

    plt.figure()
    plt.contour(X, Y, Z, levels=50)
    plt.plot(hist[:,0], hist[:,1], 'r-', linewidth=1, label='Trajetória do melhor')
    plt.title(f"Trajetória do AG - {func_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def plot_3d(func_name):
    hist, _, func, lim_inf, lim_sup, _ = executar_ag(func_name)

    x = np.linspace(lim_inf, lim_sup, 100)
    y = np.linspace(lim_inf, lim_sup, 100)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
    traj_z = [func([p[0], p[1]]) for p in hist]
    ax.plot(hist[:,0], hist[:,1], traj_z, 'r-', linewidth=2, label='Trajetória')
    ax.set_title(f"Superfície + Trajetória - {func_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()

def comparar_desempenho():
    resultados = {}
    for nome in ['rastrigin','ackley','peaks']:
        print(f"\nExecutando comparação para {nome}...")
        _, fitness_hist, _, _, _, _ = executar_ag(nome)
        resultados[nome] = fitness_hist

    plt.figure()
    for nome, valores in resultados.items():
        plt.plot(valores, label=nome.capitalize())
    plt.title("Comparação de Convergência")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

# Execução completa
for funcao in ['rastrigin','ackley','peaks']:
    print("\n" + "-"*30)
    print(f"EXECUTANDO: {funcao.upper()}")
    print("-"*30 + "\n")
    plot_2d(funcao)
    plot_3d(funcao)

comparar_desempenho()