
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from core_rna import treinar_modelo_rna, fazer_previsao

# Configurar diretórios
base_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
output_dir = "resultados_visuais"
os.makedirs(output_dir, exist_ok=True)

def preparar_dados_janela(df, janela_dias):
    """
    Cria dataset supervisionado: X (t-janela ... t-1) -> y (t)
    """
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - janela_dias):
        X.append(data_scaled[i:i+janela_dias].flatten()) # Flatten para RNA simples
        y.append(data_scaled[i+janela_dias])
        
    return np.array(X), np.array(y), scaler

def rodar_bateria_testes():
    print(">>> Iniciando Bateria de Testes RNA (Janelas Temporais)...")
    
    # 1. Carregar Dados Reais
    # Usando o arquivo mais longo disponível
    csv_path = os.path.join(base_data_path, "bitcoin_2010-07-17_2024-06-28.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Erro: Arquivo {csv_path} não encontrado. Usando bitcoin.csv padrão.")
        csv_path = os.path.join(base_data_path, "bitcoin.csv")
        
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    print(f"    [DADOS] Carregados {len(df)} registros de {df['Date'].min().date()} a {df['Date'].max().date()}")

    # 2. Configurar Experimentos
    janelas = [7, 30, 90, 365] # Semanal, Mensal, Trimestral, Anual
    
    # 3. Loop de Execução
    resultados_metrics = []
    
    for janela in janelas:
        print(f"\n--- 🧪 Experimento: Janela de {janela} Dias ---")
        
        # Preparar dados
        X, y, scaler = preparar_dados_janela(df, janela)
        
        # Split Treino/Teste (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Treinar (Simulação Rápida)
        model, history = treinar_modelo_rna(X_train, y_train, epochs=100)
        
        # Prever
        preds_scaled = fazer_previsao(model, X_test)
        
        # Desnormalizar para Plotar Reais (USD)
        preds_real = scaler.inverse_transform(preds_scaled)
        y_test_real = scaler.inverse_transform(y_test)
        
        # Calcular Erro
        erro_medio = np.mean(np.abs(preds_real - y_test_real))
        resultados_metrics.append((janela, erro_medio))
        print(f"    [RESULTADO] Erro Médio Absoluto (MAE): ${erro_medio:.2f}")
        
        # 4. Gerar Gráfico
        plt.figure(figsize=(12, 6))
        # Plotar apenas os últimos 300 dias para ficar visível (zoom)
        zoom = 300
        plt.plot(y_test_real[-zoom:], label='Preço Real (USD)', color='blue', alpha=0.6)
        plt.plot(preds_real[-zoom:], label=f'Previsão RNA (Janela {janela}d)', color='red', linestyle='--')
        
        plt.title(f"Experimento RNA: Janela de Observação = {janela} Dias (Visão Zoom)")
        plt.xlabel("Dias (Amostra Recente)")
        plt.ylabel("Preço Bitcoin (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"{output_dir}/rna_janela_{janela}d.png"
        plt.savefig(filename)
        print(f"    [GRAFICO] Salvo em: {filename}")
        plt.close()

    # 5. Gráfico Comparativo de Erros
    plt.figure(figsize=(8, 5))
    janelas_x = [str(j) for j in janelas]
    erros_y = [r[1] for r in resultados_metrics]
    plt.bar(janelas_x, erros_y, color=['green', 'orange', 'purple', 'red'])
    plt.title("Comparativo de Erro por Janela Temporal (Menor é Melhor)")
    plt.xlabel("Tamanho da Janela (Dias)")
    plt.ylabel("Erro Médio (USD)")
    plt.savefig(f"{output_dir}/comparativo_erros_rna.png")
    
    print("\n>>> Bateria Concluída com Sucesso!")

if __name__ == "__main__":
    rodar_bateria_testes()
