
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance

def rodar_analise_sazonal():
    print(">>> [MÓDULO 04] Caçador de Padrões Sazonais (XGBoost)...")
    
    # 1. Carregar Dados
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, 'data', 'bitcoin.csv')
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Engenharia de Features (Calendário)
    # Não usamos PREÇO passado. A IA só sabe "Que dia é hoje?".
    df['DayOfWeek'] = df['Date'].dt.dayofweek # 0=Seg, 6=Dom
    df['DayOfMonth'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    
    # Target: Retorno do dia seguinte (t+1) > 0? (1 se subiu, 0 se caiu)
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['Target'] = (df['Return'] > 0).astype(int)
    df = df.dropna()
    
    # Features X
    features = ['DayOfWeek', 'DayOfMonth', 'Month', 'Quarter', 'Year']
    X = df[features]
    y = df['Target']
    
    # 3. Treinar XGBoost
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4)
    model.fit(X, y)
    
    # 4. Analisar Importância (O que define o preço? O dia da semana?)
    output_dir = "resultados_visuais"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plot_importance(model, importance_type='gain', max_num_features=10, height=0.5, show_values=False)
    plt.title("O que move o Bitcoin? (Importância Sazonal)")
    plt.savefig(f"{output_dir}/importancia_features.png")
    print(f"    [GRAFICO 1] Salvo em {output_dir}/importancia_features.png")
    
    # 5. Mapa de Calor: Retorno Médio por Dia da Semana
    # Vamos calcular na mão para plotar o que a IA aprendeu
    avg_return_by_day = df.groupby('DayOfWeek')['Return'].mean() * 100 # Em %
    days_labels = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in avg_return_by_day]
    plt.bar(days_labels, avg_return_by_day, color=colors)
    plt.title("Padrão Semanal do Bitcoin (Média Histórica)")
    plt.ylabel("Retorno Médio Diário (%)")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{output_dir}/padrao_semanal.png")
    print(f"    [GRAFICO 2] Salvo em {output_dir}/padrao_semanal.png")
    
    print("\n>>> Conclusão Sazonal:")
    print(f"   Melhor Dia para Comprar: {days_labels[np.argmax(avg_return_by_day)]}")
    print(f"   Pior Dia para Comprar: {days_labels[np.argmin(avg_return_by_day)]}")

if __name__ == "__main__":
    rodar_analise_sazonal()
