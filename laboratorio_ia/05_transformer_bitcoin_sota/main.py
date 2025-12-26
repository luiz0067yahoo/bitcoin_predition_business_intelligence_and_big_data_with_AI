
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Adicionar diretório pai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_and_process_data

# Configuração do Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=8, num_layers=2, dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        
        # Camada de entrada para projetar dimensão 1 para d_model
        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Decoder final
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (Sequence, Batch, Feature) -> Transformer do PyTorch espera assim por padrão se batch_first=False
        # Mas nosso loader manda (Batch, Sequence, Feature). Vamos ajustar.
        src = src.permute(1, 0, 2) # Agora (Seq, Batch, Feat)
        
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # Pegar apenas o último output da sequência para prever o próximo passo
        output = output[-1, :, :]
        
        output = self.decoder(output)
        return output

def train_transformer():
    print("="*50)
    print("[INI] MODELO 3: TRANSFORMER (TIME SERIES) - TESTE ROBUSTO")
    print(f"[INFO] Rodando em: {device}")
    print("="*50)

    # 1. Carregar Dados
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'bitcoin.csv')
    data_dict = load_and_process_data(csv_path, seq_length=60)
    
    # Converter Tensores
    X_train = torch.from_numpy(data_dict['X_train']).float().to(device)
    y_train = torch.from_numpy(data_dict['y_train']).float().to(device)
    X_test = torch.from_numpy(data_dict['X_test']).float().to(device)
    y_test = torch.from_numpy(data_dict['y_test']).float().to(device)
    
    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze(-1)
        X_test = X_test.unsqueeze(-1)

    # Detectar features dinamicamente
    num_features = X_train.shape[2]
    print(f"[INFO] Features detectadas pelo Transformer: {num_features}")

    # 2. Inicializar Modelo
    model = TimeSeriesTransformer(input_size=num_features, nhead=8, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # Learning rate menor para Transformer

    # 3. Treinamento
    epochs = 600
    print(f"[INFO] Iniciando treinamento por {epochs} epocas (Modo TITAN - GPU)...")
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        
        # Ajustar dimensões se necessário
        loss = criterion(output.squeeze(), y_train.squeeze())
        
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"   Epoca [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # 4. Avaliação
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test).cpu().numpy()
        
    scaler = data_dict['scaler']
    preds_real = scaler.inverse_transform(preds_scaled)
    target_real = scaler.inverse_transform(data_dict['y_test'].reshape(-1, 1))

    # 5. Métricas
    rmse = np.sqrt(mean_squared_error(target_real, preds_real))
    mae = mean_absolute_error(target_real, preds_real)
    
    print("-" * 30)
    print(f"[RES] RESULTADOS PARA TRANSFORMER:")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAE:  ${mae:.2f}")
    print("-" * 30)

    # 6. Salvar Gráfico Validação
    plt.figure(figsize=(12, 6))
    plt.plot(target_real, label='Preço Real', color='blue')
    plt.plot(preds_real, label='Previsão Transformer', color='purple', alpha=0.8)
    plt.title('Validação Transformer: Real vs Previsto')
    plt.xlabel('Dias')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(True)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, 'result_transformer_validation.png')
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
    # Transformer espera (Batch, Seq, Features) -> (1, 60, 1)
    current_sequence = torch.FloatTensor(last_60_scaled).view(1, 60, 1).to(device)
    
    for _ in range(30):
        with torch.no_grad():
            next_val = model(current_sequence)
            future_predictions.append(next_val.item())
            
            # Atualizar sequência
            next_val_reshaped = next_val.view(1, 1, 1)
            current_sequence = torch.cat((current_sequence[:, 1:, :], next_val_reshaped), dim=1)
            
    # Desnormalizar
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_real_price = scaler.inverse_transform(future_predictions)
    
    # Salvar Gráfico Futuro
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 31), future_real_price, marker='o', color='magenta', label='Projeção Transformer Jan/2026')
    plt.title('Projeção Transformer - Próximos 30 Dias')
    plt.xlabel('Dias Futuros')
    plt.ylabel('Preço Estimado (USD)')
    plt.legend()
    plt.grid(True)
    
    future_plot_path = os.path.join(output_dir, 'result_transformer_FORECAST.png')
    plt.savefig(future_plot_path)
    print(f"[GRAFICO] Previsao Futura salva em: {future_plot_path}")
    
    print("-" * 30)
    print(f"Preco Estimado para Daqui 30 dias (Transformer): ${future_real_price[-1][0]:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    train_transformer()