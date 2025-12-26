
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from core_transformer import treinar_transformer
import torch

def rodar_transformer_sota():
    print(">>> [MÓDULO 05] State-of-the-Art (Transformer Attention)...")
    
    # 1. Carregar e Preparar Dados
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, 'data', 'bitcoin.csv')
    df = pd.read_csv(csv_path)
    
    # Dataset multivariado simples
    data = df[['Close', 'Volume']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Criar sequências (Olhar 60 dias para trás)
    seq_len = 60
    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len, 0]) # Predizer apenas Close
        
    X = np.array(X)
    y = np.array(y)
    
    # Split
    split = int(len(X) * 0.9) # Treino maior
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # 2. Treinar
    model = treinar_transformer(X_train, y_train, epochs=50)
    
    # 3. Previsão
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        preds_scaled = model(X_t).cpu().numpy()
        
    # 4. Desnormalizar
    dummy = np.zeros((len(preds_scaled), 2))
    dummy[:, 0] = preds_scaled[:, 0]
    preds_real = scaler.inverse_transform(dummy)[:, 0]
    
    dummy_y = np.zeros((len(y_test), 2))
    dummy_y[:, 0] = y_test
    y_test_real = scaler.inverse_transform(dummy_y)[:, 0]
    
    # 5. Visualização SOTA
    output_dir = "resultados_visuais"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_real, label='Preço Real', color='black', alpha=0.5)
    plt.plot(preds_real, label='Previsão Transformer SOTA', color='purple', linewidth=1.5)
    
    # Destacar acertos em picos (onde a IA brilhou)
    # Lógica simples visual: Círculos onde a variação foi alta e a IA acertou perto
    plt.title("Transformer SOTA: Capacidade de Capturar Tendências de Longo Prazo")
    plt.xlabel("Dias (Conjunto de Teste)")
    plt.ylabel("Preço (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/sota_prediction.png")
    print(f"    [GRAFICO] Salvo em {output_dir}/sota_prediction.png")

if __name__ == "__main__":
    rodar_transformer_sota()
