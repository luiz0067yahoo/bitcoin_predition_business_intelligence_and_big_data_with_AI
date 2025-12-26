
import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from arch import arch_model

# Adicionar diretório pai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_and_process_data

def train_genetic_algorithm():
    print("="*50)
    print("[INI] MODELO 4: ALGORITMOS GENETICOS (OTIMIZACAO DE VOLATILIDADE)")
    print("[INFO] Objetivo: Encontrar melhores parametros (p, o, q) para modelo EGARCH")
    print("="*50)

    # 1. Carregar Dados
    # Usaremos o DataFrame original pois GARCH precisa dos Retornos (Variação %), não do preço normalizado
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'bitcoin.csv')
    
    # Carregar DF cru
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date')
    
    # Calcular Retornos Percentuais (100 * log(rt/rt-1))
    # Multiplicamos por 100 para estabilidade numérica do otimizador GARCH
    returns = 100 * df['Close'].pct_change().dropna()
    
    print(f"[INFO] Total de amostras de retorno: {len(returns)}")

    # 2. Configurar Algoritmo Genético (DEAP)
    
    # Função de Aptidão (Fitness): Minimizar o AIC (Akaike Information Criterion)
    # AIC mede a qualidade do modelo penalizando complexidade. Menor é melhor.
    def eval_egarch(individual):
        p, o, q = int(individual[0]), int(individual[1]), int(individual[2])
        # Restrições para evitar erros
        if p == 0 and q == 0: return (1e10,) # Penalidade alta
        
        try:
            # Vol='EGARCH' modela assimetria (panic selling vs panic buying)
            model = arch_model(returns, vol='EGARCH', p=p, o=o, q=q, dist='Normal')
            res = model.fit(disp='off')
            return (res.aic,)
        except:
            return (1e10,) # Penalidade se der erro de convergência

    # Definição do Problema (Minimização)
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    
    # Gene: Inteiros entre 1 e 5 para p, o, q
    toolbox.register("attr_int", random.randint, 1, 4)
    
    # Indivíduo: Lista de 3 genes [p, o, q]
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Operadores Genéticos
    toolbox.register("evaluate", eval_egarch)
    toolbox.register("mate", tools.cxTwoPoint) # Cruzamento
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=4, indpb=0.3) # Mutação
    toolbox.register("select", tools.selTournament, tournsize=3) # Seleção

    # 3. Execução do GA
    population_size = 10
    generations = 5 # Pequeno para ser rápido no teste
    
    print(f"[INFO] Iniciando Evolucao: Pop={population_size}, Geracoes={generations}")
    pop = toolbox.population(n=population_size)
    
    # Algoritmo eaSimple (Evolução básica)
    # CXPB = Probabilidade de Cruzamento, MUTPB = Probabilidade de Mutação
    final_pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=True)

    # 4. Melhor Solução
    best_ind = tools.selBest(final_pop, 1)[0]
    print("-" * 30)
    print(f"[RES] MELHORES PARAMETROS ENCONTRADOS:")
    print(f"   p (Lag Volatilidade) = {best_ind[0]}")
    print(f"   o (Assimetria)       = {best_ind[1]}")
    print(f"   q (Lag Erro)         = {best_ind[2]}")
    print(f"   AIC (Fitness)        = {best_ind.fitness.values[0]:.4f}")
    print("-" * 30)

    # 5. Visualizar Modelo Vencedor
    best_p, best_o, best_q = best_ind[0], best_ind[1], best_ind[2]
    model_opt = arch_model(returns, vol='EGARCH', p=best_p, o=best_o, q=best_q)
    res_opt = model_opt.fit(disp='off')
    
    # Plotar Volatilidade Condicional
    plt.figure(figsize=(12, 6))
    plt.plot(res_opt.conditional_volatility, color='orange', label='Volatilidade Modelada (Risco)')
    plt.plot(np.abs(returns), color='blue', alpha=0.2, label='Retornos Absolutos (Real)')
    plt.title(f'Volatilidade Bitcoin Otimizada via GA (EGARCH {best_p},{best_o},{best_q})')
    plt.legend()
    plt.grid(True)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, 'result_genetic.png')
    plt.savefig(plot_path)
    print(f"[GRAFICO] Salvo em: {plot_path}")

if __name__ == "__main__":
    train_genetic_algorithm()