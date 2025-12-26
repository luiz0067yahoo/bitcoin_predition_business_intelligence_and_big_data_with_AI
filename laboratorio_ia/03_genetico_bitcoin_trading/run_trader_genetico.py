
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from deap import base, creator, tools, algorithms

# ==========================================
# 1. CORE: SIMULADOR DE TRADING
# ==========================================
def simular_trading(df, window_short, window_long, stop_loss_pct):
    """
    Roda um Backtest simples.
    Estratégia: Cruzamento de Médias (Golden Cross).
    Se Curta > Longa: COMPRA.
    Se Curta < Longa: VENDE.
    Stop Loss: Vende se cair X%.
    """
    # Validar genes ruins (ex: Janela curta > longa é inválido)
    if window_short >= window_long:
        return -1000.0, # Penalidade grave
        
    # Calcular indicadores
    df['SMA_S'] = df['Close'].rolling(window=int(window_short)).mean()
    df['SMA_L'] = df['Close'].rolling(window=int(window_long)).mean()
    
    saldo_usd = 10000.0 # Começa com $10k
    posicao_btc = 0.0
    preco_compra = 0.0
    
    historico_patrimonio = []
    
    # Loop dia a dia (lento, mas preciso para simulação)
    for i in range(len(df)):
        preco = df['Close'].iloc[i]
        sma_s = df['SMA_S'].iloc[i]
        sma_l = df['SMA_L'].iloc[i]
        
        patrimonio_total = saldo_usd + (posicao_btc * preco)
        historico_patrimonio.append(patrimonio_total)
        
        # Lógica de Stop Loss
        if posicao_btc > 0:
            queda = (preco - preco_compra) / preco_compra
            if queda < -stop_loss_pct:
                # Stop Loss acionado
                saldo_usd = posicao_btc * preco
                posicao_btc = 0
                continue # Sai do loop do dia
        
        # Sinal de Compra
        if sma_s > sma_l and posicao_btc == 0:
            posicao_btc = saldo_usd / preco
            preco_compra = preco
            saldo_usd = 0
            
        # Sinal de Venda
        elif sma_s < sma_l and posicao_btc > 0:
            saldo_usd = posicao_btc * preco
            posicao_btc = 0
            
    # Retorna o Lucro Final ($)
    lucro = patrimonio_total - 10000.0
    return lucro, historico_patrimonio

# ==========================================
# 2. CONFIGURAÇÃO GENÉTICA (DEAP)
# ==========================================
# Definir objetivo: Maximizar Lucro
# ATENÇÃO: Verificar se Creator já foi instanciado para evitar erro em re-execução
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except:
    pass # Já criado

toolbox = base.Toolbox()

# Genes: [Janela_Curta (5-50), Janela_Longa (50-200), Stop_Loss (0.01-0.20)]
toolbox.register("attr_short", random.randint, 5, 50)
toolbox.register("attr_long", random.randint, 51, 200)
toolbox.register("attr_stop", random.uniform, 0.01, 0.20)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_short, toolbox.attr_long, toolbox.attr_stop), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Variável Global para segurar os dados (evita passar dataframe gigante no eval)
GLOBAL_DF = None

def eval_genome(individual):
    # Wrapper para avaliar indivíduo
    lucro, _ = simular_trading(GLOBAL_DF, individual[0], individual[1], individual[2])
    return lucro,

toolbox.register("evaluate", eval_genome)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
# Ajuste manual pós-mutação para garantir inteiros nas janelas
def check_bounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                child[0] = int(child[0]) # SMA Short
                child[1] = int(child[1]) # SMA Long
                if child[0] < 5: child[0] = 5
                if child[1] < 10: child[1] = 10
            return offspring
        return wrapper
    return decorator

toolbox.decorate("mate", check_bounds(5, 200))
toolbox.decorate("mutate", check_bounds(5, 200))
toolbox.register("select", tools.selTournament, tournsize=3)

# ==========================================
# 3. RUNNER
# ==========================================
def rodar_algoritmo_genetico():
    global GLOBAL_DF
    print(">>> [MÓDULO 03] Iniciando Trader Evolutivo...")
    
    # 1. Carregar Dados
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, 'data', 'bitcoin.csv')
    GLOBAL_DF = pd.read_csv(csv_path)
    # Limitar dataset para treino ser rápido
    GLOBAL_DF = GLOBAL_DF[-1000:].reset_index(drop=True) 
    
    output_dir = "resultados_visuais"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Experimento: Evolução
    pop_size = 20
    generations = 10
    
    print(f"    [GENETICO] Populacao={pop_size}, Geracoes={generations}")
    pop = toolbox.population(n=pop_size)
    
    # Estatísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Rodar!
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, 
                                       stats=stats, verbose=True)
    
    # 3. Pegar o Campeão
    best_ind = tools.selBest(pop, 1)[0]
    print(f"\n🏆 Melhor Estratégia Encontrada:")
    print(f"   Média Curta: {best_ind[0]} dias")
    print(f"   Média Longa: {best_ind[1]} dias")
    print(f"   Stop Loss: {best_ind[2]*100:.1f}%")
    print(f"   Lucro: ${best_ind.fitness.values[0]:.2f}")
    
    # 4. Validar e Plotar
    lucro, curva_patrimonio = simular_trading(GLOBAL_DF, best_ind[0], best_ind[1], best_ind[2])
    
    # Curva Buy & Hold (Se tivesse só comprado e segurado)
    preco_inicial = GLOBAL_DF['Close'].iloc[0]
    buy_hold = (GLOBAL_DF['Close'] / preco_inicial) * 10000.0
    
    plt.figure(figsize=(12, 6))
    plt.plot(buy_hold, label='Buy & Hold (Passivo)', color='gray', linestyle='--')
    plt.plot(curva_patrimonio, label='IA Trader (Genético)', color='green', linewidth=2)
    plt.title(f"A Evolução do Lucro: Genético vs Passivo")
    plt.ylabel("Patrimônio (USD)")
    plt.xlabel("Dias")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/resultado_trader.png")
    print(f"    [GRAFICO] Salvo em {output_dir}/resultado_trader.png")

if __name__ == "__main__":
    rodar_algoritmo_genetico()
