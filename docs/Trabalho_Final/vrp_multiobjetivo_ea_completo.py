'''
UNIVERSIDADE FEDERAL DE MINAS GERAIS - UFMG
Programa de Pós-Graduação em Engenharia Elétrica - PPGEE
Trabalho: Estudo e Desenvolvimento de uma Ferramenta Computacional
Problema: VRP multiobjetivo com equidade de carga via algoritmos evolucionários.
Descrição: Problema Multiobjetivo de Roteamento de Veículos - Equidade de carga entre veículos via algoritmos evolucionários
Disciplina: Computação Evolucionária - Maio/2026
Professor: Michel Bessani  
Alunos: Gabriel Bonanato Lopes - 
        Stephanie Ferreira Lemos - 2026692372
        Vanderley Matias da Silva - 2026673882 
        Vinícius F. Cerqueira -  

#------------------------------------------------------------------------------------------------------------------

Este script implementa três algoritmos multiobjetivo:  
1) NSGA-II;  
2) SPEA2;  
3) NSGA-III.

O problema tratado é uma versão multiobjetivo do CVRP, com três objetivos:  
F1 = custo total das rotas;
F2 = variância dos custos das rotas, como medida de equidade global;  
F3 = makespan, isto é, custo da rota mais longa.

O código salva figuras e tabelas no mesmo diretório do arquivo-fonte.  
Caso nenhum arquivo CVRPLIB seja informado, uma instância didática interna é usada. '''

#------------------------------------------------------------------------------------------------------------------

from __future__ import annotations  # Permite anotações de tipo mais flexíveis em versões modernas do Python.

import math  # Importa funções matemáticas elementares.
import random  # Importa geração de números pseudoaleatórios.
import re  # Importa expressões regulares para leitura opcional de arquivos CVRPLIB.
import time  # Importa medição de tempo computacional.
from dataclasses import dataclass, field  # Importa recurso para criar classes simples de dados.
from pathlib import Path  # Importa manipulação robusta de caminhos de arquivos.
from typing import Dict, Iterable, List, Optional, Sequence, Tuple  # Importa tipos auxiliares para legibilidade.

import matplotlib.pyplot as plt  # Importa biblioteca para geração de gráficos.
import numpy as np  # Importa biblioteca numérica para vetores, matrizes e estatísticas.
import pandas as pd  # Importa biblioteca para tabelas e salvamento em CSV.

# 1. CONFIGURAÇÃO GERAL DO EXPERIMENTO

MODO_COMPLETO_PROPOSTA = False  # False executa rápido; True usa configuração próxima à proposta do trabalho.

POPULACAO_RAPIDA = 28  # Define tamanho de população para execução didática rápida.
GERACOES_RAPIDAS = 25  # Define número de gerações para execução didática rápida.
EXECUCOES_RAPIDAS = 1  # Define número de execuções independentes para execução didática rápida.

POPULACAO_PROPOSTA = 100  # Define população sugerida na apresentação.
GERACOES_PROPOSTA = 500  # Define gerações sugeridas na apresentação.
EXECUCOES_PROPOSTA = 30  # Define execuções independentes sugeridas na apresentação.

TAXA_CROSSOVER = 0.90  # Define probabilidade de aplicar crossover.
SEMENTE_BASE = 20260524  # Define semente base para reprodutibilidade.

ARQUIVO_CVRPLIB: Optional[str] = None  # Informe aqui um caminho .vrp da CVRPLIB, se desejar usar dados reais.

USAR_MAX_MIN_COMO_EQUIDADE = False  # False usa variância; True substitui F2 por max-min para estudo secundário.

MOSTRAR_GRAFICOS = False  # True chama plt.show(); False apenas salva as figuras.

# 2. MODELOS DE DADOS

@dataclass  # Transforma a classe em estrutura de dados com construtor automático.
class VRPInstance:  # Define a classe que representa uma instância de CVRP.
    name: str  # Armazena o nome da instância.
    coords: np.ndarray  # Armazena coordenadas dos nós, incluindo depósito na posição zero.
    demands: np.ndarray  # Armazena demandas dos nós, com demanda zero no depósito.
    capacity: float  # Armazena capacidade máxima de cada veículo.
    vehicles: int  # Armazena número de veículos disponíveis.
    distance_matrix: np.ndarray  # Armazena matriz de distâncias entre todos os pares de nós.

@dataclass  # Transforma a classe em estrutura de dados com construtor automático.
class Individual:  # Define a classe que representa uma solução candidata.
    perm: List[int]  # Armazena a permutação dos clientes, sem incluir o depósito.
    routes: List[List[int]] = field(default_factory=list)  # Armazena as rotas decodificadas.
    route_costs: np.ndarray = field(default_factory=lambda: np.array([]))  # Armazena custo de cada rota.
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))  # Armazena vetor [F1, F2, F3].
    rank: int = 10**9  # Armazena ranking de Pareto usado no NSGA-II/NSGA-III.
    crowding: float = 0.0  # Armazena distância de aglomeração usada no NSGA-II.
    spea_fitness: float = 10**9  # Armazena aptidão SPEA2; menor é melhor.

    def copy(self) -> "Individual":  # Define método para copiar o indivíduo.
        return Individual(  # Retorna uma nova instância independente.
            perm=list(self.perm),  # Copia a permutação.
            routes=[list(r) for r in self.routes],  # Copia cada rota.
            route_costs=np.array(self.route_costs, dtype=float),  # Copia custos das rotas.
            objectives=np.array(self.objectives, dtype=float),  # Copia objetivos.
            rank=int(self.rank),  # Copia rank.
            crowding=float(self.crowding),  # Copia crowding.
            spea_fitness=float(self.spea_fitness),  # Copia fitness SPEA2.
        )  # Fecha a construção do indivíduo copiado.

# 3. DIRETÓRIO DE SAÍDA

def obter_diretorio_saida() -> Path:  # Define função para encontrar onde salvar resultados.
    try:  # Inicia tentativa de usar o diretório do arquivo-fonte.
        return Path(__file__).resolve().parent  # Retorna a pasta onde este script está salvo.
    except NameError:  # Captura caso __file__ não exista, como em notebooks.
        return Path.cwd()  # Retorna o diretório atual como alternativa.

SAIDA = obter_diretorio_saida()  # Define o diretório global de saída.

# 4. INSTÂNCIA CVRP - Seção de instância.

def calcular_matriz_distancias(coords: np.ndarray) -> np.ndarray:  # Define função para calcular distâncias euclidianas.
    n = len(coords)  # Obtém o número total de nós.
    matriz = np.zeros((n, n), dtype=float)  # Cria matriz quadrada inicialmente nula.
    for i in range(n):  # Percorre cada nó de origem.
        for j in range(n):  # Percorre cada nó de destino.
            dx = coords[i, 0] - coords[j, 0]  # Calcula diferença em x.
            dy = coords[i, 1] - coords[j, 1]  # Calcula diferença em y.
            matriz[i, j] = round(math.hypot(dx, dy))  # Calcula distância euclidiana arredondada, padrão comum em CVRP.
    return matriz  # Retorna a matriz de distâncias.

def criar_instancia_didatica() -> VRPInstance:  # Define instância interna para execução sem dados externos.
    coords = np.array(  # Cria matriz de coordenadas dos nós.
        [  # Inicia lista de coordenadas.
            [50, 50],  # Nó 0: depósito.
            [20, 70],  # Cliente 1.
            [25, 85],  # Cliente 2.
            [35, 80],  # Cliente 3.
            [60, 85],  # Cliente 4.
            [75, 75],  # Cliente 5.
            [85, 60],  # Cliente 6.
            [80, 40],  # Cliente 7.
            [65, 25],  # Cliente 8.
            [45, 20],  # Cliente 9.
            [25, 30],  # Cliente 10.
            [15, 45],  # Cliente 11.
            [35, 55],  # Cliente 12.
            [55, 65],  # Cliente 13.
            [70, 55],  # Cliente 14.
            [55, 35],  # Cliente 15.
            [40, 40],  # Cliente 16.
            [30, 60],  # Cliente 17.
            [62, 72],  # Cliente 18.
            [73, 28],  # Cliente 19.
            [18, 25],  # Cliente 20.
        ],  # Encerra lista de coordenadas.
        dtype=float,  # Define tipo numérico float.
    )  # Fecha criação da matriz de coordenadas.
    demands = np.array(  # Cria vetor de demandas.
        [0, 9, 7, 12, 10, 8, 11, 13, 6, 9, 10, 7, 8, 12, 6, 11, 9, 5, 7, 10, 8],  # Define demandas.
        dtype=float,  # Define tipo numérico float.
    )  # Fecha criação do vetor de demandas.
    capacity = 35.0  # Define capacidade máxima do veículo.
    vehicles = 6  # Define número de veículos disponíveis.
    dist = calcular_matriz_distancias(coords)  # Calcula matriz de distâncias.
    return VRPInstance("Instancia_Didatica_20_clientes", coords, demands, capacity, vehicles, dist)  # Retorna instância pronta.

def ler_cvrplib(caminho: str) -> VRPInstance:  # Define função para ler arquivo .vrp em formato CVRPLIB clássico.
    path = Path(caminho)  # Converte caminho para objeto Path.
    texto = path.read_text(encoding="utf-8", errors="ignore").splitlines()  # Lê o arquivo tolerando caracteres especiais.
    name = path.stem  # Usa nome do arquivo como nome padrão da instância.
    capacity = None  # Inicializa capacidade como desconhecida.
    dimension = None  # Inicializa dimensão como desconhecida.
    coords: Dict[int, Tuple[float, float]] = {}  # Cria dicionário para coordenadas por índice.
    demands: Dict[int, float] = {}  # Cria dicionário para demandas por índice.
    section = None  # Controla seção atual do arquivo.
    for raw in texto:  # Percorre cada linha do arquivo.
        line = raw.strip()  # Remove espaços laterais.
        if not line:  # Verifica linha vazia.
            continue  # Ignora linha vazia.
        upper = line.upper()  # Cria versão em maiúsculas para comparação.
        if upper.startswith("NAME"):  # Detecta linha de nome.
            name = line.split(":")[-1].strip()  # Extrai nome após dois-pontos.
        elif upper.startswith("DIMENSION"):  # Detecta dimensão.
            dimension = int(re.findall(r"\d+", line)[0])  # Extrai o primeiro número inteiro da linha.
        elif upper.startswith("CAPACITY"):  # Detecta capacidade.
            capacity = float(re.findall(r"\d+", line)[0])  # Extrai capacidade como float.
        elif upper.startswith("NODE_COORD_SECTION"):  # Detecta início das coordenadas.
            section = "coords"  # Marca seção de coordenadas.
        elif upper.startswith("DEMAND_SECTION"):  # Detecta início das demandas.
            section = "demands"  # Marca seção de demandas.
        elif upper.startswith("DEPOT_SECTION") or upper.startswith("EOF"):  # Detecta seção que não será processada.
            section = None  # Encerra leitura de dados relevantes.
        elif section == "coords":  # Processa linha de coordenadas.
            parts = line.split()  # Divide a linha em campos.
            if len(parts) >= 3:  # Confirma que há índice, x e y.
                idx = int(parts[0]) - 1  # Converte índice 1-based da CVRPLIB para 0-based do Python.
                coords[idx] = (float(parts[1]), float(parts[2]))  # Armazena coordenada.
        elif section == "demands":  # Processa linha de demanda.
            parts = line.split()  # Divide a linha em campos.
            if len(parts) >= 2:  # Confirma que há índice e demanda.
                idx = int(parts[0]) - 1  # Converte índice 1-based para 0-based.
                demands[idx] = float(parts[1])  # Armazena demanda.
    if dimension is None or capacity is None:  # Verifica metadados obrigatórios.
        raise ValueError("Arquivo CVRPLIB sem DIMENSION ou CAPACITY reconhecíveis.")  # Lança erro explicativo.
    coords_array = np.array([coords[i] for i in range(dimension)], dtype=float)  # Ordena coordenadas por índice.
    demands_array = np.array([demands.get(i, 0.0) for i in range(dimension)], dtype=float)  # Ordena demandas por índice.
    match = re.search(r"k(\d+)", name.lower())  # Procura padrão kN no nome para inferir veículos.
    vehicles = int(match.group(1)) if match else max(1, math.ceil(sum(demands_array) / capacity))  # Infere frota se possível.
    dist = calcular_matriz_distancias(coords_array)  # Calcula matriz de distância.
    return VRPInstance(name, coords_array, demands_array, capacity, vehicles, dist)  # Retorna instância lida.

def carregar_instancia() -> VRPInstance:  # Define função central de carregamento da instância.
    if ARQUIVO_CVRPLIB is None:  # Verifica se não foi informado arquivo externo.
        return criar_instancia_didatica()  # Retorna instância didática embutida.
    return ler_cvrplib(ARQUIVO_CVRPLIB)  # Lê instância externa quando informada.

# 5. DECODIFICAÇÃO, REPARO E AVALIAÇÃO - Seção de avaliação.

def custo_rota(route: Sequence[int], inst: VRPInstance) -> float:  # Define função para calcular custo de uma rota.
    if len(route) == 0:  # Verifica rota vazia.
        return 0.0  # Rota vazia tem custo zero.
    total = inst.distance_matrix[0, route[0]]  # Soma distância do depósito até o primeiro cliente.
    for a, b in zip(route[:-1], route[1:]):  # Percorre pares consecutivos de clientes.
        total += inst.distance_matrix[a, b]  # Soma distância entre clientes consecutivos.
    total += inst.distance_matrix[route[-1], 0]  # Soma retorno do último cliente ao depósito.
    return float(total)  # Retorna custo como float.

def demanda_rota(route: Sequence[int], inst: VRPInstance) -> float:  # Define função para calcular demanda total de uma rota.
    return float(np.sum(inst.demands[list(route)])) if route else 0.0  # Soma demandas dos clientes da rota.

def decodificar_permutacao(perm: Sequence[int], inst: VRPInstance) -> List[List[int]]:  # Define função que converte permutação em rotas factíveis.
    routes: List[List[int]] = []  # Inicializa lista de rotas.
    atual: List[int] = []  # Inicializa rota em construção.
    carga_atual = 0.0  # Inicializa carga acumulada da rota atual.
    for cliente in perm:  # Percorre clientes na ordem da permutação.
        demanda = float(inst.demands[cliente])  # Obtém demanda do cliente.
        if atual and carga_atual + demanda > inst.capacity:  # Verifica se adicionar cliente violaria capacidade.
            routes.append(atual)  # Fecha rota atual.
            atual = [cliente]  # Inicia nova rota com o cliente atual.
            carga_atual = demanda  # Reinicia carga com demanda do cliente atual.
        else:  # Caso não haja violação de capacidade.
            atual.append(cliente)  # Adiciona cliente à rota atual.
            carga_atual += demanda  # Atualiza carga acumulada.
    if atual:  # Verifica se restou rota aberta.
        routes.append(atual)  # Adiciona última rota construída.
    while len(routes) < inst.vehicles:  # Completa rotas vazias se frota disponível for maior que rotas usadas.
        routes.append([])  # Adiciona rota vazia para manter dimensão operacional.
    if len(routes) > inst.vehicles:  # Verifica se número de rotas excedeu frota.
        routes = reparar_numero_de_rotas(routes, inst)  # Tenta reparar excesso de rotas.
    return routes  # Retorna lista de rotas.

def reparar_numero_de_rotas(routes: List[List[int]], inst: VRPInstance) -> List[List[int]]:  # Define reparo quando a divisão gera veículos demais.
    todas = [c for route in routes for c in route]  # Achata todos os clientes em uma lista.
    todas.sort(key=lambda c: inst.demands[c], reverse=True)  # Ordena clientes por demanda decrescente para empacotamento guloso.
    novas: List[List[int]] = [[] for _ in range(inst.vehicles)]  # Cria exatamente K rotas vazias.
    cargas = [0.0 for _ in range(inst.vehicles)]  # Cria carga acumulada para cada rota.
    for cliente in todas:  # Percorre clientes em ordem decrescente de demanda.
        candidatos = [i for i in range(inst.vehicles) if cargas[i] + inst.demands[cliente] <= inst.capacity]  # Lista rotas factíveis.
        if candidatos:  # Verifica se existe rota factível.
            idx = min(candidatos, key=lambda i: cargas[i])  # Escolhe rota factível menos carregada.
        else:  # Caso nenhuma rota comporte o cliente.
            idx = int(np.argmin(cargas))  # Escolhe rota menos carregada mesmo com penalização implícita.
        novas[idx].append(cliente)  # Insere cliente na rota escolhida.
        cargas[idx] += float(inst.demands[cliente])  # Atualiza carga da rota.
    return novas  # Retorna rotas reparadas.

def avaliar_individuo(ind: Individual, inst: VRPInstance) -> Individual:  # Define função de avaliação multiobjetivo.
    ind.routes = decodificar_permutacao(ind.perm, inst)  # Decodifica a permutação em rotas.
    custos = np.array([custo_rota(r, inst) for r in ind.routes], dtype=float)  # Calcula custo de cada rota.
    cargas = np.array([demanda_rota(r, inst) for r in ind.routes], dtype=float)  # Calcula carga de cada rota.
    excesso = np.maximum(0.0, cargas - inst.capacity)  # Calcula excesso de capacidade, se houver.
    penalidade = 10000.0 * np.sum(excesso)  # Define penalidade alta para violação residual.
    f1 = float(np.sum(custos) + penalidade)  # Calcula custo total penalizado.
    if USAR_MAX_MIN_COMO_EQUIDADE:  # Verifica se o estudo secundário usa max-min.
        f2 = float(np.max(custos) - np.min(custos)) if len(custos) else 0.0  # Calcula amplitude de custos.
    else:  # Caso padrão usa variância.
        f2 = float(np.var(custos) + penalidade)  # Calcula variância penalizada.
    f3 = float(np.max(custos) + penalidade) if len(custos) else 0.0  # Calcula makespan penalizado.
    ind.route_costs = custos  # Armazena custos das rotas.
    ind.objectives = np.array([f1, f2, f3], dtype=float)  # Armazena objetivos.
    return ind  # Retorna indivíduo avaliado.

# 6. DOMINÂNCIA, FRENTES E DIVERSIDADE - Seção de Pareto.

def domina(a: Individual, b: Individual) -> bool:  # Define relação de dominância para minimização.
    return bool(np.all(a.objectives <= b.objectives) and np.any(a.objectives < b.objectives))  # Retorna True se a domina b.

def nondominated_sort(pop: List[Individual]) -> List[List[int]]:  # Define ordenação não dominada.
    n = len(pop)  # Obtém tamanho da população.
    S = [[] for _ in range(n)]  # S[p] conterá soluções dominadas por p.
    n_dom = [0 for _ in range(n)]  # n_dom[p] contará quantas soluções dominam p.
    fronts: List[List[int]] = [[]]  # Inicializa primeira frente.
    for p in range(n):  # Percorre cada candidato p.
        for q in range(n):  # Compara p com cada candidato q.
            if p == q:  # Ignora comparação consigo mesmo.
                continue  # Continua para próximo q.
            if domina(pop[p], pop[q]):  # Verifica se p domina q.
                S[p].append(q)  # Registra q como dominado por p.
            elif domina(pop[q], pop[p]):  # Verifica se q domina p.
                n_dom[p] += 1  # Incrementa contagem de dominadores de p.
        if n_dom[p] == 0:  # Se p não é dominado por ninguém.
            pop[p].rank = 0  # Atribui rank zero.
            fronts[0].append(p)  # Insere p na primeira frente.
    i = 0  # Inicializa índice da frente atual.
    while fronts[i]:  # Continua enquanto a frente atual não estiver vazia.
        next_front: List[int] = []  # Cria próxima frente.
        for p in fronts[i]:  # Percorre soluções da frente atual.
            for q in S[p]:  # Percorre soluções dominadas por p.
                n_dom[q] -= 1  # Remove uma dominância ao avançar de frente.
                if n_dom[q] == 0:  # Verifica se q ficou sem dominadores restantes.
                    pop[q].rank = i + 1  # Atribui rank da próxima frente.
                    next_front.append(q)  # Adiciona q à próxima frente.
        i += 1  # Avança índice de frente.
        fronts.append(next_front)  # Adiciona próxima frente à lista.
    return fronts[:-1]  # Retorna todas as frentes não vazias.

def calcular_crowding(pop: List[Individual], front: List[int]) -> None:  # Define cálculo de distância de aglomeração.
    if not front:  # Verifica frente vazia.
        return  # Encerra sem ação.
    m = len(pop[front[0]].objectives)  # Obtém número de objetivos.
    for idx in front:  # Percorre indivíduos da frente.
        pop[idx].crowding = 0.0  # Zera distância de aglomeração.
    for obj in range(m):  # Percorre cada objetivo.
        front_sorted = sorted(front, key=lambda i: pop[i].objectives[obj])  # Ordena frente por objetivo atual.
        pop[front_sorted[0]].crowding = float("inf")  # Marca extremo inferior com crowding infinito.
        pop[front_sorted[-1]].crowding = float("inf")  # Marca extremo superior com crowding infinito.
        f_min = pop[front_sorted[0]].objectives[obj]  # Obtém mínimo do objetivo.
        f_max = pop[front_sorted[-1]].objectives[obj]  # Obtém máximo do objetivo.
        if f_max == f_min:  # Verifica objetivo constante.
            continue  # Evita divisão por zero.
        for k in range(1, len(front_sorted) - 1):  # Percorre pontos internos.
            prev_f = pop[front_sorted[k - 1]].objectives[obj]  # Obtém valor anterior.
            next_f = pop[front_sorted[k + 1]].objectives[obj]  # Obtém valor posterior.
            pop[front_sorted[k]].crowding += float((next_f - prev_f) / (f_max - f_min))  # Soma distância normalizada.

def obter_nao_dominados(pop: List[Individual]) -> List[Individual]:  # Define função para extrair frente não dominada.
    copias = [p.copy() for p in pop]  # Copia população para não alterar ranks originais.
    fronts = nondominated_sort(copias)  # Executa ordenação não dominada.
    if not fronts:  # Verifica se não há frente.
        return []  # Retorna lista vazia.
    return [copias[i] for i in fronts[0]]  # Retorna indivíduos da primeira frente.

# 7. OPERADORES EVOLUCIONÁRIOS

def criar_individuo_aleatorio(inst: VRPInstance, rng: random.Random) -> Individual:  # Define criação de indivíduo aleatório.
    clientes = list(range(1, len(inst.coords)))  # Cria lista de clientes excluindo depósito.
    rng.shuffle(clientes)  # Embaralha clientes.
    return avaliar_individuo(Individual(clientes), inst)  # Avalia e retorna indivíduo.

def bcrc_crossover(p1: Individual, p2: Individual, inst: VRPInstance, rng: random.Random) -> Tuple[Individual, Individual]:  # Define crossover baseado em rota.
    def filho(a: Individual, b: Individual) -> Individual:  # Define função interna para gerar um filho.
        rotas_validas = [r for r in a.routes if r]  # Mantém apenas rotas não vazias do pai A.
        if not rotas_validas:  # Verifica ausência de rota válida.
            return b.copy()  # Retorna cópia do pai B como contingência.
        rota_preservada = list(rng.choice(rotas_validas))  # Escolhe uma rota inteira do pai A.
        preservados = set(rota_preservada)  # Cria conjunto dos clientes preservados.
        resto = [c for c in b.perm if c not in preservados]  # Mantém ordem do pai B sem clientes repetidos.
        perm = rota_preservada + resto  # Concatena rota preservada com restante dos clientes.
        return avaliar_individuo(Individual(perm), inst)  # Avalia e retorna filho.
    if rng.random() > TAXA_CROSSOVER:  # Verifica se crossover não será aplicado.
        return p1.copy(), p2.copy()  # Retorna cópias dos pais.
    return filho(p1, p2), filho(p2, p1)  # Retorna dois filhos simétricos.

def mutacao_swap(perm: List[int], rng: random.Random) -> None:  # Define mutação por troca.
    if len(perm) < 2:  # Verifica tamanho mínimo.
        return  # Encerra se não há dois genes.
    i, j = rng.sample(range(len(perm)), 2)  # Sorteia duas posições distintas.
    perm[i], perm[j] = perm[j], perm[i]  # Troca os clientes.

def mutacao_inversao(perm: List[int], rng: random.Random) -> None:  # Define mutação por inversão de segmento.
    if len(perm) < 3:  # Verifica tamanho mínimo.
        return  # Encerra se segmento não é útil.
    i, j = sorted(rng.sample(range(len(perm)), 2))  # Sorteia extremidades do segmento.
    perm[i:j + 1] = reversed(perm[i:j + 1])  # Inverte a subsequência.

def mutacao_inter_rota(ind: Individual, inst: VRPInstance, rng: random.Random) -> Individual:  # Define mutação por deslocamento entre rotas.
    rotas = [list(r) for r in ind.routes if r]  # Copia rotas não vazias.
    if len(rotas) < 2:  # Verifica se existem ao menos duas rotas.
        return ind  # Retorna indivíduo sem alteração.
    origem = rng.randrange(len(rotas))  # Sorteia rota de origem.
    destino = rng.randrange(len(rotas))  # Sorteia rota de destino.
    if origem == destino:  # Verifica se origem e destino coincidem.
        return ind  # Encerra sem alteração.
    pos = rng.randrange(len(rotas[origem]))  # Sorteia posição do cliente a mover.
    cliente = rotas[origem].pop(pos)  # Remove cliente da rota de origem.
    pos_dest = rng.randrange(len(rotas[destino]) + 1)  # Sorteia posição de inserção no destino.
    rotas[destino].insert(pos_dest, cliente)  # Insere cliente na rota destino.
    perm = [c for r in rotas for c in r]  # Achata rotas em nova permutação.
    return avaliar_individuo(Individual(perm), inst)  # Avalia e retorna indivíduo mutado.

def aplicar_mutacao(ind: Individual, inst: VRPInstance, rng: random.Random) -> Individual:  # Define mutação composta.
    prob = 1.0 / max(1, len(ind.perm))  # Calcula taxa de mutação 1/n.
    perm = list(ind.perm)  # Copia permutação.
    if rng.random() < prob * len(perm):  # Aplica swap com chance escalada para ocorrer em nível de indivíduo.
        mutacao_swap(perm, rng)  # Executa troca de clientes.
    if rng.random() < prob * len(perm):  # Aplica inversão com chance escalada.
        mutacao_inversao(perm, rng)  # Executa inversão de segmento.
    mutado = avaliar_individuo(Individual(perm), inst)  # Avalia mutação por permutação.
    if rng.random() < 0.30:  # Aplica deslocamento inter-rota com probabilidade moderada.
        mutado = mutacao_inter_rota(mutado, inst, rng)  # Executa mutação entre rotas.
    return mutado  # Retorna indivíduo mutado.

# 8. NSGA-II

def torneio_nsga2(pop: List[Individual], rng: random.Random) -> Individual:  # Define seleção por torneio do NSGA-II.
    a, b = rng.sample(pop, 2)  # Sorteia dois competidores.
    if a.rank < b.rank:  # Verifica se A está em frente melhor.
        return a  # Retorna A.
    if b.rank < a.rank:  # Verifica se B está em frente melhor.
        return b  # Retorna B.
    if a.crowding > b.crowding:  # Desempata por maior crowding.
        return a  # Retorna A.
    return b  # Retorna B.

def selecionar_nsga2(pop: List[Individual], tamanho: int) -> List[Individual]:  # Define seleção ambiental do NSGA-II.
    fronts = nondominated_sort(pop)  # Ordena população por frentes de Pareto.
    selecionados: List[Individual] = []  # Inicializa próxima geração.
    for front in fronts:  # Percorre frentes em ordem de qualidade.
        calcular_crowding(pop, front)  # Calcula crowding da frente.
        if len(selecionados) + len(front) <= tamanho:  # Verifica se toda a frente cabe.
            selecionados.extend(pop[i] for i in front)  # Adiciona frente completa.
        else:  # Caso a frente não caiba inteira.
            front_ordenada = sorted(front, key=lambda i: pop[i].crowding, reverse=True)  # Ordena por diversidade.
            falta = tamanho - len(selecionados)  # Calcula quantos indivíduos faltam.
            selecionados.extend(pop[i] for i in front_ordenada[:falta])  # Adiciona os mais diversos.
            break  # Encerra seleção.
    return [s.copy() for s in selecionados]  # Retorna cópias selecionadas.

def executar_nsga2(inst: VRPInstance, tamanho_pop: int, geracoes: int, seed: int) -> Tuple[List[Individual], List[Dict[str, float]]]:  # Executa NSGA-II.
    rng = random.Random(seed)  # Cria gerador aleatório local.
    pop = [criar_individuo_aleatorio(inst, rng) for _ in range(tamanho_pop)]  # Cria população inicial.
    historico: List[Dict[str, float]] = []  # Inicializa histórico de convergência.
    for gen in range(geracoes):  # Percorre gerações.
        fronts = nondominated_sort(pop)  # Atualiza rankings.
        for front in fronts:  # Percorre frentes.
            calcular_crowding(pop, front)  # Atualiza crowding.
        filhos: List[Individual] = []  # Inicializa população de descendentes.
        while len(filhos) < tamanho_pop:  # Gera filhos até completar tamanho.
            p1 = torneio_nsga2(pop, rng)  # Seleciona primeiro pai.
            p2 = torneio_nsga2(pop, rng)  # Seleciona segundo pai.
            c1, c2 = bcrc_crossover(p1, p2, inst, rng)  # Aplica crossover.
            filhos.append(aplicar_mutacao(c1, inst, rng))  # Muta e adiciona primeiro filho.
            if len(filhos) < tamanho_pop:  # Verifica espaço para segundo filho.
                filhos.append(aplicar_mutacao(c2, inst, rng))  # Muta e adiciona segundo filho.
        pop = selecionar_nsga2(pop + filhos, tamanho_pop)  # Executa seleção ambiental elitista.
        if gen % max(1, geracoes // 20) == 0 or gen == geracoes - 1:  # Registra histórico periodicamente.
            frente = obter_nao_dominados(pop)  # Obtém frente não dominada atual.
            historico.append({"geracao": gen, "f1_min": min(i.objectives[0] for i in frente), "card": len(frente)})  # Armazena métricas simples.
    return pop, historico  # Retorna população final e histórico.

# 9. SPEA2

def atribuir_fitness_spea2(pop: List[Individual]) -> None:  # Define cálculo de fitness SPEA2.
    n = len(pop)  # Obtém tamanho do conjunto.
    strength = np.zeros(n, dtype=float)  # Inicializa força de dominância.
    raw = np.zeros(n, dtype=float)  # Inicializa fitness bruto.
    F = np.array([ind.objectives for ind in pop], dtype=float)  # Monta matriz de objetivos.
    for i in range(n):  # Percorre cada indivíduo i.
        for j in range(n):  # Compara com cada indivíduo j.
            if i != j and domina(pop[i], pop[j]):  # Verifica se i domina j.
                strength[i] += 1.0  # Incrementa força de i.
    for i in range(n):  # Percorre cada indivíduo i.
        for j in range(n):  # Compara com cada indivíduo j.
            if i != j and domina(pop[j], pop[i]):  # Verifica se j domina i.
                raw[i] += strength[j]  # Soma força dos dominadores de i.
    if n > 1:  # Verifica se há ao menos dois indivíduos.
        dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)  # Calcula matriz de distâncias no espaço objetivo.
        np.fill_diagonal(dist, np.inf)  # Ignora distância de cada ponto para si mesmo.
        k = max(1, int(math.sqrt(n)))  # Define k do k-vizinho mais próximo.
        sigma_k = np.partition(dist, kth=min(k, n - 1) - 1, axis=1)[:, min(k, n - 1) - 1]  # Obtém distância ao k-ésimo vizinho.
        density = 1.0 / (sigma_k + 2.0)  # Calcula densidade SPEA2.
    else:  # Caso conjunto tenha um indivíduo.
        density = np.array([0.0])  # Define densidade zero.
    for i, ind in enumerate(pop):  # Percorre indivíduos com índice.
        ind.spea_fitness = float(raw[i] + density[i])  # Atribui fitness total SPEA2.

def truncar_arquivo_spea2(arquivo: List[Individual], tamanho: int) -> List[Individual]:  # Define truncamento de arquivo SPEA2.
    arquivo = [a.copy() for a in arquivo]  # Copia arquivo para evitar efeitos colaterais.
    while len(arquivo) > tamanho:  # Continua removendo até atingir tamanho.
        F = np.array([ind.objectives for ind in arquivo], dtype=float)  # Monta matriz de objetivos.
        dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)  # Calcula distâncias pareadas.
        np.fill_diagonal(dist, np.inf)  # Ignora diagonal.
        ordenadas = np.sort(dist, axis=1)  # Ordena distâncias de cada indivíduo.
        remover = min(range(len(arquivo)), key=lambda i: tuple(ordenadas[i]))  # Escolhe indivíduo em região mais densa.
        arquivo.pop(remover)  # Remove indivíduo escolhido.
    return arquivo  # Retorna arquivo truncado.

def atualizar_arquivo_spea2(uniao: List[Individual], tamanho: int) -> List[Individual]:  # Define atualização do arquivo SPEA2.
    atribuir_fitness_spea2(uniao)  # Calcula fitness SPEA2 no conjunto união.
    arquivo = [ind.copy() for ind in uniao if ind.spea_fitness < 1.0]  # Seleciona indivíduos não dominados segundo critério SPEA2.
    if len(arquivo) > tamanho:  # Verifica excesso no arquivo.
        return truncar_arquivo_spea2(arquivo, tamanho)  # Retorna arquivo truncado por densidade.
    if len(arquivo) < tamanho:  # Verifica se o arquivo ficou pequeno.
        restantes = sorted(uniao, key=lambda ind: ind.spea_fitness)  # Ordena candidatos por fitness.
        chaves = {tuple(ind.perm) for ind in arquivo}  # Registra permutações já no arquivo.
        for cand in restantes:  # Percorre candidatos ordenados.
            if tuple(cand.perm) not in chaves:  # Evita duplicata exata de permutação.
                arquivo.append(cand.copy())  # Adiciona candidato ao arquivo.
                chaves.add(tuple(cand.perm))  # Atualiza conjunto de chaves.
            if len(arquivo) == tamanho:  # Verifica se arquivo completou.
                break  # Encerra preenchimento.
    return arquivo  # Retorna arquivo atualizado.

def torneio_spea2(arquivo: List[Individual], rng: random.Random) -> Individual:  # Define seleção por torneio SPEA2.
    a, b = rng.sample(arquivo, 2)  # Sorteia dois indivíduos do arquivo.
    return a if a.spea_fitness <= b.spea_fitness else b  # Retorna menor fitness.

def executar_spea2(inst: VRPInstance, tamanho_pop: int, geracoes: int, seed: int) -> Tuple[List[Individual], List[Dict[str, float]]]:  # Executa SPEA2.
    rng = random.Random(seed)  # Cria gerador aleatório local.
    pop = [criar_individuo_aleatorio(inst, rng) for _ in range(tamanho_pop)]  # Cria população inicial.
    arquivo: List[Individual] = []  # Inicializa arquivo externo vazio.
    historico: List[Dict[str, float]] = []  # Inicializa histórico.
    for gen in range(geracoes):  # Percorre gerações.
        arquivo = atualizar_arquivo_spea2(pop + arquivo, tamanho_pop)  # Atualiza arquivo elitista.
        filhos: List[Individual] = []  # Inicializa descendentes.
        while len(filhos) < tamanho_pop:  # Gera descendentes até completar população.
            p1 = torneio_spea2(arquivo, rng)  # Seleciona primeiro pai.
            p2 = torneio_spea2(arquivo, rng)  # Seleciona segundo pai.
            c1, c2 = bcrc_crossover(p1, p2, inst, rng)  # Aplica crossover.
            filhos.append(aplicar_mutacao(c1, inst, rng))  # Muta e adiciona primeiro filho.
            if len(filhos) < tamanho_pop:  # Verifica espaço para segundo filho.
                filhos.append(aplicar_mutacao(c2, inst, rng))  # Muta e adiciona segundo filho.
        pop = filhos  # Substitui população por descendentes.
        if gen % max(1, geracoes // 20) == 0 or gen == geracoes - 1:  # Registra histórico periodicamente.
            frente = obter_nao_dominados(arquivo)  # Obtém frente do arquivo.
            historico.append({"geracao": gen, "f1_min": min(i.objectives[0] for i in frente), "card": len(frente)})  # Armazena métricas.
    arquivo = atualizar_arquivo_spea2(pop + arquivo, tamanho_pop)  # Atualiza arquivo final.
    return arquivo, historico  # Retorna arquivo final e histórico.

# 10. NSGA-III

def gerar_pontos_referencia_3d(divisoes: int = 12) -> np.ndarray:  # Define geração de pontos Das-Dennis em 3 objetivos.
    refs = []  # Inicializa lista de pontos.
    for i in range(divisoes + 1):  # Percorre primeira coordenada inteira.
        for j in range(divisoes + 1 - i):  # Percorre segunda coordenada inteira.
            k = divisoes - i - j  # Calcula terceira coordenada inteira.
            refs.append([i / divisoes, j / divisoes, k / divisoes])  # Normaliza ponto para simplex.
    return np.array(refs, dtype=float)  # Retorna matriz de pontos de referência.

def normalizar_objetivos(F: np.ndarray) -> np.ndarray:  # Define normalização min-max simples dos objetivos.
    z_min = np.min(F, axis=0)  # Calcula ideal aproximado.
    z_max = np.max(F, axis=0)  # Calcula nadir aproximado.
    denom = np.where(z_max - z_min == 0.0, 1.0, z_max - z_min)  # Evita divisão por zero.
    return (F - z_min) / denom  # Retorna matriz normalizada em [0,1].

def associar_referencias(F_norm: np.ndarray, refs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # Associa pontos normalizados aos pontos de referência.
    refs_norm = refs / np.linalg.norm(refs, axis=1, keepdims=True)  # Normaliza vetores de referência.
    refs_norm = np.nan_to_num(refs_norm, nan=0.0)  # Substitui eventuais NaN por zero.
    associacoes = []  # Inicializa associações.
    distancias = []  # Inicializa distâncias perpendiculares.
    for f in F_norm:  # Percorre cada solução normalizada.
        norma_f = np.linalg.norm(f)  # Calcula norma do vetor objetivo.
        if norma_f == 0.0:  # Verifica ponto ideal.
            associacoes.append(0)  # Associa ao primeiro ponto por convenção.
            distancias.append(0.0)  # Distância zero.
            continue  # Avança para próxima solução.
        proj = refs_norm @ f  # Calcula projeção escalar em cada referência.
        closest = np.argmax(proj)  # Usa maior projeção angular como aproximação do nicho mais próximo.
        perp = np.linalg.norm(f - proj[closest] * refs_norm[closest])  # Calcula distância perpendicular.
        associacoes.append(int(closest))  # Armazena índice da referência.
        distancias.append(float(perp))  # Armazena distância perpendicular.
    return np.array(associacoes, dtype=int), np.array(distancias, dtype=float)  # Retorna associações e distâncias.

def selecionar_nsga3(pop: List[Individual], tamanho: int, refs: np.ndarray) -> List[Individual]:  # Define seleção ambiental do NSGA-III.
    fronts = nondominated_sort(pop)  # Ordena população por dominância.
    selecionados_idx: List[int] = []  # Inicializa índices selecionados.
    ultimo_front: List[int] = []  # Inicializa última frente parcial.
    for front in fronts:  # Percorre frentes.
        if len(selecionados_idx) + len(front) <= tamanho:  # Verifica se frente cabe inteira.
            selecionados_idx.extend(front)  # Adiciona frente completa.
        else:  # Caso frente não caiba inteira.
            ultimo_front = front  # Guarda frente parcial.
            break  # Interrompe inclusão de frentes completas.
    if len(selecionados_idx) == tamanho:  # Verifica se já completou seleção.
        return [pop[i].copy() for i in selecionados_idx]  # Retorna indivíduos selecionados.
    candidatos_idx = selecionados_idx + ultimo_front  # Combina selecionados completos e candidatos da frente crítica.
    F = np.array([pop[i].objectives for i in candidatos_idx], dtype=float)  # Extrai objetivos dos candidatos relevantes.
    F_norm = normalizar_objetivos(F)  # Normaliza objetivos.
    assoc, dist = associar_referencias(F_norm, refs)  # Associa candidatos a referências.
    ocupacao = {r: 0 for r in range(len(refs))}  # Inicializa ocupação dos nichos.
    for local_pos in range(len(selecionados_idx)):  # Percorre indivíduos já aceitos.
        ocupacao[int(assoc[local_pos])] += 1  # Incrementa ocupação do respectivo nicho.
    faltam = tamanho - len(selecionados_idx)  # Calcula quantos indivíduos precisam ser escolhidos.
    fronteira_locais = list(range(len(selecionados_idx), len(candidatos_idx)))  # Cria índices locais da frente parcial.
    escolhidos_locais: List[int] = []  # Inicializa lista de escolhidos da frente parcial.
    while faltam > 0 and fronteira_locais:  # Continua niching até completar ou esgotar candidatos.
        refs_com_candidatos = sorted(set(int(assoc[i]) for i in fronteira_locais), key=lambda r: ocupacao[r])  # Ordena referências pela menor ocupação.
        ref = refs_com_candidatos[0]  # Escolhe referência menos ocupada com candidatos.
        candidatos_ref = [i for i in fronteira_locais if int(assoc[i]) == ref]  # Filtra candidatos associados à referência.
        if ocupacao[ref] == 0:  # Verifica se nicho ainda está vazio.
            escolhido = min(candidatos_ref, key=lambda i: dist[i])  # Escolhe candidato mais próximo da referência.
        else:  # Caso nicho já tenha ocupantes.
            escolhido = random.choice(candidatos_ref)  # Escolhe aleatoriamente para manter diversidade.
        escolhidos_locais.append(escolhido)  # Adiciona candidato escolhido.
        fronteira_locais.remove(escolhido)  # Remove candidato da lista disponível.
        ocupacao[ref] += 1  # Atualiza ocupação do nicho.
        faltam -= 1  # Reduz quantidade faltante.
    final_idx = selecionados_idx + [candidatos_idx[i] for i in escolhidos_locais]  # Converte índices locais para globais.
    return [pop[i].copy() for i in final_idx]  # Retorna população selecionada.

def torneio_nsga3(pop: List[Individual], rng: random.Random) -> Individual:  # Define torneio simples para NSGA-III.
    a, b = rng.sample(pop, 2)  # Sorteia dois competidores.
    if a.rank < b.rank:  # Verifica melhor rank de A.
        return a  # Retorna A.
    if b.rank < a.rank:  # Verifica melhor rank de B.
        return b  # Retorna B.
    return a if rng.random() < 0.5 else b  # Desempata aleatoriamente.

def executar_nsga3(inst: VRPInstance, tamanho_pop: int, geracoes: int, seed: int) -> Tuple[List[Individual], List[Dict[str, float]]]:  # Executa NSGA-III.
    rng = random.Random(seed)  # Cria gerador aleatório local.
    refs = gerar_pontos_referencia_3d(12)  # Gera 92 pontos de referência para três objetivos.
    tamanho = min(tamanho_pop, len(refs)) if MODO_COMPLETO_PROPOSTA else tamanho_pop  # Ajusta tamanho se desejado.
    pop = [criar_individuo_aleatorio(inst, rng) for _ in range(tamanho)]  # Cria população inicial.
    historico: List[Dict[str, float]] = []  # Inicializa histórico.
    for gen in range(geracoes):  # Percorre gerações.
        nondominated_sort(pop)  # Atualiza ranks.
        filhos: List[Individual] = []  # Inicializa descendentes.
        while len(filhos) < tamanho:  # Gera filhos até completar tamanho.
            p1 = torneio_nsga3(pop, rng)  # Seleciona primeiro pai.
            p2 = torneio_nsga3(pop, rng)  # Seleciona segundo pai.
            c1, c2 = bcrc_crossover(p1, p2, inst, rng)  # Aplica crossover.
            filhos.append(aplicar_mutacao(c1, inst, rng))  # Muta primeiro filho.
            if len(filhos) < tamanho:  # Verifica espaço para segundo filho.
                filhos.append(aplicar_mutacao(c2, inst, rng))  # Muta segundo filho.
        pop = selecionar_nsga3(pop + filhos, tamanho, refs)  # Executa seleção ambiental NSGA-III.
        if gen % max(1, geracoes // 20) == 0 or gen == geracoes - 1:  # Registra histórico periodicamente.
            frente = obter_nao_dominados(pop)  # Obtém frente não dominada.
            historico.append({"geracao": gen, "f1_min": min(i.objectives[0] for i in frente), "card": len(frente)})  # Armazena métricas.
    return pop, historico  # Retorna população final e histórico.

# 11. MÉTRICAS DE AVALIAÇÃO

def matriz_objetivos(pop: List[Individual]) -> np.ndarray:  # Define função para extrair matriz de objetivos.
    return np.array([ind.objectives for ind in pop], dtype=float)  # Retorna matriz N x M.

def hypervolume_monte_carlo(F_norm: np.ndarray, ref: np.ndarray, amostras: int = 60000, seed: int = 123) -> float:  # Aproxima HV normalizado.
    rng = np.random.default_rng(seed)  # Cria gerador NumPy reprodutível.
    pontos = rng.random((amostras, F_norm.shape[1])) * ref  # Amostra pontos uniformes no hipercubo de referência.
    dominado = np.zeros(amostras, dtype=bool)  # Inicializa máscara de pontos dominados.
    for f in F_norm:  # Percorre cada solução da frente.
        dominado |= np.all(f <= pontos, axis=1)  # Marca amostras dominadas por f em minimização.
    volume_caixa = float(np.prod(ref))  # Calcula volume do hipercubo de referência.
    return float(np.mean(dominado) * volume_caixa)  # Retorna estimativa do hypervolume.

def igd(F_norm: np.ndarray, ref_front_norm: np.ndarray) -> float:  # Calcula Inverted Generational Distance.
    dist = np.linalg.norm(ref_front_norm[:, None, :] - F_norm[None, :, :], axis=2)  # Calcula distâncias referência-solução.
    return float(np.mean(np.min(dist, axis=1)))  # Retorna média da menor distância.

def spread_nn(F_norm: np.ndarray) -> float:  # Calcula diversidade por coeficiente de variação do vizinho mais próximo.
    if len(F_norm) < 3:  # Verifica quantidade mínima.
        return float("nan")  # Retorna NaN quando não há base suficiente.
    dist = np.linalg.norm(F_norm[:, None, :] - F_norm[None, :, :], axis=2)  # Calcula distâncias pareadas.
    np.fill_diagonal(dist, np.inf)  # Ignora distância do ponto a si mesmo.
    nn = np.min(dist, axis=1)  # Obtém distância ao vizinho mais próximo.
    media = np.mean(nn)  # Calcula média das distâncias.
    if media == 0.0:  # Verifica divisão por zero.
        return 0.0  # Retorna zero se todos os pontos coincidem.
    return float(np.std(nn) / media)  # Retorna coeficiente de variação; menor tende a indicar distribuição mais uniforme.

def normalizar_por_referencia(F: np.ndarray, minimo: np.ndarray, maximo: np.ndarray) -> np.ndarray:  # Normaliza objetivos por limites comuns.
    denom = np.where(maximo - minimo == 0.0, 1.0, maximo - minimo)  # Evita divisão por zero.
    return (F - minimo) / denom  # Retorna matriz normalizada.

def escolher_solucao_compromisso(frente: List[Individual], minimo: np.ndarray, maximo: np.ndarray) -> Individual:  # Escolhe solução mais próxima do ideal normalizado.
    F = matriz_objetivos(frente)  # Extrai matriz de objetivos da frente.
    F_norm = normalizar_por_referencia(F, minimo, maximo)  # Normaliza objetivos.
    dist_ideal = np.linalg.norm(F_norm, axis=1)  # Calcula distância ao ponto ideal [0,0,0].
    return frente[int(np.argmin(dist_ideal))]  # Retorna solução de menor distância ao ideal.

# 12. VISUALIZAÇÃO E SALVAMENTO DE RESULTADOS

def salvar_ou_mostrar(nome: str) -> None:  # Define função para salvar e opcionalmente exibir figura.
    caminho = SAIDA / nome  # Constrói caminho completo da figura.
    plt.tight_layout()  # Ajusta layout para reduzir cortes.
    plt.savefig(caminho, dpi=300, bbox_inches="tight")  # Salva figura em alta resolução.
    print(f"Figura salva em: {caminho}")  # Informa caminho salvo.
    if MOSTRAR_GRAFICOS:  # Verifica se deve exibir gráficos.
        plt.show()  # Exibe figura na tela.
    plt.close()  # Fecha figura para liberar memória.

def plotar_frentes_3d(frentes: Dict[str, List[Individual]]) -> None:  # Define gráfico 3D das frentes.
    fig = plt.figure(figsize=(9, 7))  # Cria figura.
    ax = fig.add_subplot(111, projection="3d")  # Cria eixo tridimensional.
    for nome, frente in frentes.items():  # Percorre frentes por algoritmo.
        F = matriz_objetivos(frente)  # Extrai objetivos.
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], label=nome, s=28)  # Plota pontos 3D.
    ax.set_xlabel("F1 - Custo total")  # Nomeia eixo x.
    ax.set_ylabel("F2 - Equidade")  # Nomeia eixo y.
    ax.set_zlabel("F3 - Makespan")  # Nomeia eixo z.
    ax.set_title("Frentes não dominadas aproximadas")  # Define título.
    ax.legend()  # Exibe legenda.
    salvar_ou_mostrar("fig_01_frentes_3d.png")  # Salva e exibe.

def plotar_pares_objetivos(frentes: Dict[str, List[Individual]]) -> None:  # Define gráfico F1-F2.
    plt.figure(figsize=(8, 6))  # Cria figura.
    for nome, frente in frentes.items():  # Percorre algoritmos.
        F = matriz_objetivos(frente)  # Extrai objetivos.
        plt.scatter(F[:, 0], F[:, 1], label=nome, s=28)  # Plota custo versus equidade.
    plt.xlabel("F1 - Custo total")  # Nomeia eixo x.
    plt.ylabel("F2 - Variância ou max-min")  # Nomeia eixo y.
    plt.title("Tradeoff entre custo total e equidade")  # Define título.
    plt.legend()  # Exibe legenda.
    plt.grid(True, alpha=0.3)  # Ativa grade discreta.
    salvar_ou_mostrar("fig_02_tradeoff_custo_equidade.png")  # Salva e exibe.

def plotar_metricas(df_metricas: pd.DataFrame) -> None:  # Define gráfico de métricas.
    metricas = ["HV", "IGD", "Spread", "Cardinalidade"]  # Lista métricas a plotar.
    for metrica in metricas:  # Percorre métricas.
        plt.figure(figsize=(8, 5))  # Cria figura.
        plt.bar(df_metricas["Algoritmo"], df_metricas[metrica])  # Plota barras.
        plt.xlabel("Algoritmo")  # Nomeia eixo x.
        plt.ylabel(metrica)  # Nomeia eixo y.
        plt.title(f"Comparação por {metrica}")  # Define título.
        plt.grid(axis="y", alpha=0.3)  # Adiciona grade horizontal discreta.
        salvar_ou_mostrar(f"fig_metrica_{metrica.lower()}.png")  # Salva figura.

def plotar_rotas(inst: VRPInstance, solucao: Individual, nome_algoritmo: str) -> None:  # Define gráfico espacial das rotas.
    plt.figure(figsize=(8, 7))  # Cria figura.
    plt.scatter(inst.coords[0, 0], inst.coords[0, 1], marker="s", s=120, label="Depósito")  # Plota depósito.
    plt.scatter(inst.coords[1:, 0], inst.coords[1:, 1], s=40, label="Clientes")  # Plota clientes.
    for idx, route in enumerate(solucao.routes):  # Percorre rotas.
        if not route:  # Ignora rota vazia.
            continue  # Avança para próxima rota.
        caminho = [0] + route + [0]  # Monta sequência com depósito na ida e volta.
        xy = inst.coords[caminho]  # Obtém coordenadas da rota.
        plt.plot(xy[:, 0], xy[:, 1], marker="o", linewidth=1.4, label=f"Rota {idx + 1}")  # Desenha rota.
    for i, (x, y) in enumerate(inst.coords):  # Percorre nós para anotação.
        plt.text(x + 0.8, y + 0.8, str(i), fontsize=8)  # Escreve número do nó.
    plt.title(f"Solução compromisso - {nome_algoritmo}")  # Define título.
    plt.xlabel("Coordenada X")  # Nomeia eixo x.
    plt.ylabel("Coordenada Y")  # Nomeia eixo y.
    plt.legend(fontsize=8, loc="best")  # Adiciona legenda.
    plt.grid(True, alpha=0.3)  # Ativa grade discreta.
    salvar_ou_mostrar(f"fig_rotas_{nome_algoritmo.lower().replace('-', '').replace(' ', '_')}.png")  # Salva figura.

def plotar_historico(historicos: Dict[str, List[Dict[str, float]]]) -> None:  # Define gráfico de convergência.
    plt.figure(figsize=(8, 5))  # Cria figura.
    for nome, hist in historicos.items():  # Percorre históricos.
        df = pd.DataFrame(hist)  # Converte histórico em DataFrame.
        plt.plot(df["geracao"], df["f1_min"], marker="o", label=nome)  # Plota evolução do melhor F1.
    plt.xlabel("Geração")  # Nomeia eixo x.
    plt.ylabel("Menor F1 na frente")  # Nomeia eixo y.
    plt.title("Evolução do custo total mínimo")  # Define título.
    plt.legend()  # Exibe legenda.
    plt.grid(True, alpha=0.3)  # Ativa grade.
    salvar_ou_mostrar("fig_03_convergencia_f1.png")  # Salva e exibe.

# 13. EXECUÇÃO DO EXPERIMENTO

def executar_algoritmos(inst: VRPInstance) -> Tuple[Dict[str, List[Individual]], Dict[str, List[Dict[str, float]]], pd.DataFrame]:  # Executa todos os algoritmos.
    if MODO_COMPLETO_PROPOSTA:  # Verifica modo completo.
        tamanho_pop = POPULACAO_PROPOSTA  # Usa população da proposta.
        geracoes = GERACOES_PROPOSTA  # Usa gerações da proposta.
        execucoes = EXECUCOES_PROPOSTA  # Usa execuções da proposta.
    else:  # Caso modo rápido.
        tamanho_pop = POPULACAO_RAPIDA  # Usa população rápida.
        geracoes = GERACOES_RAPIDAS  # Usa gerações rápidas.
        execucoes = EXECUCOES_RAPIDAS  # Usa execuções rápidas.
    algoritmos = {  # Cria dicionário de funções de execução.
        "NSGA-II": executar_nsga2,  # Associa nome ao NSGA-II.
        "SPEA2": executar_spea2,  # Associa nome ao SPEA2.
        "NSGA-III": executar_nsga3,  # Associa nome ao NSGA-III.
    }  # Fecha dicionário de algoritmos.
    todas_frentes: Dict[str, List[Individual]] = {}  # Inicializa frentes agregadas.
    historicos: Dict[str, List[Dict[str, float]]] = {}  # Inicializa históricos.
    linhas_resultados: List[Dict[str, float]] = []  # Inicializa tabela bruta.
    for nome, func in algoritmos.items():  # Percorre cada algoritmo.
        print(f"\nExecutando {nome}...")  # Informa algoritmo atual.
        pop_agregada: List[Individual] = []  # Inicializa lista agregada de soluções finais.
        hist_primeira_execucao: List[Dict[str, float]] = []  # Inicializa histórico representativo.
        for run in range(execucoes):  # Percorre execuções independentes.
            seed = SEMENTE_BASE + 1000 * run + len(nome)  # Define semente específica da execução.
            inicio = time.perf_counter()  # Marca início do tempo.
            pop_final, hist = func(inst, tamanho_pop, geracoes, seed)  # Executa algoritmo.
            tempo = time.perf_counter() - inicio  # Calcula tempo decorrido.
            frente = obter_nao_dominados(pop_final)  # Obtém frente não dominada da execução.
            pop_agregada.extend(frente)  # Agrega frente da execução.
            if run == 0:  # Verifica primeira execução.
                hist_primeira_execucao = hist  # Guarda histórico da primeira execução.
            for ind in frente:  # Percorre soluções da frente.
                linhas_resultados.append(  # Adiciona linha de resultado bruto.
                    {  # Inicia dicionário da linha.
                        "Algoritmo": nome,  # Registra algoritmo.
                        "Execucao": run + 1,  # Registra número da execução.
                        "F1_Custo_Total": ind.objectives[0],  # Registra F1.
                        "F2_Equidade": ind.objectives[1],  # Registra F2.
                        "F3_Makespan": ind.objectives[2],  # Registra F3.
                        "Tempo_s": tempo,  # Registra tempo da execução.
                    }  # Fecha dicionário.
                )  # Fecha append.
            print(f"  Execução {run + 1}/{execucoes}: frente={len(frente)}, tempo={tempo:.2f}s")  # Mostra progresso.
        todas_frentes[nome] = obter_nao_dominados(pop_agregada)  # Filtra frente agregada por algoritmo.
        historicos[nome] = hist_primeira_execucao  # Armazena histórico representativo.
    df_resultados = pd.DataFrame(linhas_resultados)  # Converte resultados brutos em DataFrame.
    return todas_frentes, historicos, df_resultados  # Retorna resultados principais.

def calcular_metricas(frentes: Dict[str, List[Individual]]) -> pd.DataFrame:  # Calcula métricas de qualidade das frentes.
    todas = [ind for frente in frentes.values() for ind in frente]  # Agrega todas as soluções.
    ref_empirica = obter_nao_dominados(todas)  # Calcula frente de referência empírica.
    F_ref = matriz_objetivos(ref_empirica)  # Extrai objetivos da referência.
    minimo = np.min(F_ref, axis=0)  # Define mínimo comum.
    maximo = np.max(F_ref, axis=0)  # Define máximo comum.
    ref_point = np.array([1.10, 1.10, 1.10], dtype=float)  # Define ponto de referência normalizado para HV.
    linhas = []  # Inicializa linhas de métricas.
    for nome, frente in frentes.items():  # Percorre frentes dos algoritmos.
        F = matriz_objetivos(frente)  # Extrai matriz de objetivos.
        F_norm = normalizar_por_referencia(F, minimo, maximo)  # Normaliza objetivos da frente.
        F_ref_norm = normalizar_por_referencia(F_ref, minimo, maximo)  # Normaliza frente de referência.
        linhas.append(  # Adiciona linha de métricas.
            {  # Inicia dicionário de métricas.
                "Algoritmo": nome,  # Registra algoritmo.
                "HV": hypervolume_monte_carlo(F_norm, ref_point, seed=SEMENTE_BASE + len(nome)),  # Calcula HV aproximado.
                "IGD": igd(F_norm, F_ref_norm),  # Calcula IGD.
                "Spread": spread_nn(F_norm),  # Calcula spread por vizinho mais próximo.
                "Cardinalidade": len(frente),  # Registra cardinalidade da frente.
            }  # Fecha dicionário.
        )  # Fecha append.
    return pd.DataFrame(linhas), minimo, maximo  # Retorna métricas e limites comuns.

def salvar_tabelas(df_resultados: pd.DataFrame, df_metricas: pd.DataFrame) -> None:  # Salva tabelas em CSV.
    resultados_path = SAIDA / "resultados_brutos_vrp_multiobjetivo.csv"  # Define caminho dos resultados brutos.
    metricas_path = SAIDA / "metricas_vrp_multiobjetivo.csv"  # Define caminho das métricas.
    df_resultados.to_csv(resultados_path, index=False, encoding="utf-8-sig")  # Salva resultados brutos.
    df_metricas.to_csv(metricas_path, index=False, encoding="utf-8-sig")  # Salva métricas.
    print(f"Resultados brutos salvos em: {resultados_path}")  # Informa caminho salvo.
    print(f"Métricas salvas em: {metricas_path}")  # Informa caminho salvo.

# 14. FUNÇÃO PRINCIPAL

def main() -> None:  # Define função principal do script.
    print("Iniciando experimento VRP multiobjetivo...")  # Informa início.
    print(f"Diretório de saída: {SAIDA}")  # Mostra diretório de saída.
    inst = carregar_instancia()  # Carrega instância escolhida.
    print(f"Instância: {inst.name}")  # Mostra nome da instância.
    print(f"Nós totais: {len(inst.coords)}")  # Mostra número de nós incluindo depósito.
    print(f"Clientes: {len(inst.coords) - 1}")  # Mostra número de clientes.
    print(f"Veículos: {inst.vehicles}")  # Mostra número de veículos.
    print(f"Capacidade: {inst.capacity}")  # Mostra capacidade.
    frentes, historicos, df_resultados = executar_algoritmos(inst)  # Executa algoritmos.
    df_metricas, minimo, maximo = calcular_metricas(frentes)  # Calcula métricas.
    print("\nMétricas finais:")  # Informa bloco de métricas.
    print(df_metricas.to_string(index=False))  # Imprime métricas no console.
    salvar_tabelas(df_resultados, df_metricas)  # Salva tabelas CSV.
    plotar_frentes_3d(frentes)  # Gera gráfico 3D das frentes.
    plotar_pares_objetivos(frentes)  # Gera gráfico de tradeoff F1-F2.
    plotar_metricas(df_metricas)  # Gera gráficos de métricas.
    plotar_historico(historicos)  # Gera gráfico de convergência.
    for nome, frente in frentes.items():  # Percorre cada algoritmo.
        compromisso = escolher_solucao_compromisso(frente, minimo, maximo)  # Escolhe solução compromisso.
        plotar_rotas(inst, compromisso, nome)  # Gera gráfico das rotas da solução compromisso.
    print("\nExecução concluída.")  # Informa fim da execução.
    print("Arquivos de figuras e tabelas foram salvos no mesmo diretório deste script.")  # Reforça local dos arquivos.

if __name__ == "__main__":  # Verifica se o arquivo está sendo executado diretamente.
    main()  # Chama a função principal.