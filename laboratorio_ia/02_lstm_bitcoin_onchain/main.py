
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Adicionar diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_and_process_data

# Configuração do Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HybridLSTMGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(HybridLSTMGRU, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # Fully connected output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out, _ = self.gru(out)
        # Pegar apenas o último passo temporal para previsão
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

def train_deep_learning():
    print("="*50)
    print("[INI] MODELO 2: DEEP LEARNING (LSTM + GRU) - TESTE ROBUSTO")
    print(f"[INFO] Rodando em: {device}")
    print("="*50)

    # 1. Carregar Dados
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'bitcoin.csv')
    # Deep Learning precisa de sequências temporais
    data_dict = load_and_process_data(csv_path, seq_length=60)
    
    # Converter para Tensores PyTorch
    X_train = torch.from_numpy(data_dict['X_train']).float().to(device)
    y_train = torch.from_numpy(data_dict['y_train']).float().to(device)
    X_test = torch.from_numpy(data_dict['X_test']).float().to(device)
    y_test = torch.from_numpy(data_dict['y_test']).float().to(device)
    
    # Garantir dimensão correta (Batch, Seq, Features)
    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze(-1)
        X_test = X_test.unsqueeze(-1)
        
    # Detectar número de features dinamicamente (Multivariado)
    num_features = X_train.shape[2]
    print(f"[INFO] Features detectadas: {num_features}")

    # 2. Inicializar Modelo
    model = HybridLSTMGRU(input_size=num_features, hidden_size=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # 3. Loop de Treinamento
    epochs = 1000
    print(f"[INFO] Iniciando treinamento por {epochs} epocas (Modo TITAN - GPU)...")
    
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        # Forward
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch+1) % 100 == 0:
            print(f"   Epoca [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # 4. Avaliação
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test).cpu().numpy()
        y_test_real = data_dict['y_test'] # Já está em numpy
        
    # Desfazer normalização
    scaler = data_dict['scaler']
    preds_real = scaler.inverse_transform(preds_scaled)
    target_real = scaler.inverse_transform(y_test_real) # y_test era (N, 1)

    # 5. Métricas
    rmse = np.sqrt(mean_squared_error(target_real, preds_real))
    mae = mean_absolute_error(target_real, preds_real)
    
    print("-" * 30)
    print(f"[RES] RESULTADOS PARA DEEP LEARNING (LSTM-GRU):")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAE:  ${mae:.2f}")
    print("-" * 30)

    # 6. Salvar Gráfico Validação
    plt.figure(figsize=(12, 6))
    plt.plot(target_real, label='Preço Real', color='blue')
    plt.plot(preds_real, label='Previsão LSTM-GRU', color='orange', alpha=0.8)
    plt.title('Validação de Modelo: Real vs Previsto')
    plt.xlabel('Dias')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(True)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, 'result_deep_learning_validation.png')
    plt.savefig(plot_path)
    print(f"[GRAFICO] Validacao salva em: {plot_path}")

    # ==========================================
    # 7. MODO BOLA DE CRISTAL (PREVISÃO FUTURA)
    # ==========================================
    print("\n[FUTURO] Iniciando projecao para os proximos 30 dias (Janeiro 2026)...")
    
    # Pegar os ultimos 60 dias reais do dataset completo
    df_full = data_dict['original_df']
    last_60_days = df_full['Close'].values[-60:]
    
    # Normalizar igual ao treino
    last_60_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    
    # Loop de projeção recursiva
    future_predictions = []
    current_sequence = torch.FloatTensor(last_60_scaled).view(1, 60, 1).to(device)
    
    for _ in range(30): # Prever 30 dias a frente
        with torch.no_grad():
            # Prever o proximo dia
            next_val = model(current_sequence)
            future_predictions.append(next_val.item())
            
            # Atualizar a sequencia: Tira o dia mais antigo, Poe o dia previsto
            # next_val shape: [1, 1] -> precisamos [1, 1, 1] para concatenar
            next_val_reshaped = next_val.view(1, 1, 1)
            current_sequence = torch.cat((current_sequence[:, 1:, :], next_val_reshaped), dim=1)
            
    # Desnormalizar previsões
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_real_price = scaler.inverse_transform(future_predictions)
    
    # Salvar Gráfico Futuro
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 31), future_real_price, marker='o', color='green', label='Projeção Jan/2026')
    plt.title('Projeção de Preço do Bitcoin - Próximos 30 Dias')
    plt.xlabel('Dias Futuros')
    plt.ylabel('Preço Estimado (USD)')
    plt.legend()
    plt.grid(True)
    
    future_plot_path = os.path.join(output_dir, 'result_deep_learning_FORECAST.png')
    plt.savefig(future_plot_path)
    print(f"[GRAFICO] Previsao Futura salva em: {future_plot_path}")
    
    print("-" * 30)
    print(f"Preco Estimado para Daqui 30 dias: ${future_real_price[-1][0]:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    train_deep_learning()