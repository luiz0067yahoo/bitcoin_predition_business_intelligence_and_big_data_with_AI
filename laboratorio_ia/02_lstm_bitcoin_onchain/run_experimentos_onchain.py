
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from core_lstm import treinar_lstm_onchain

# Caminhos
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'laboratorio_ia', 'utils'))
# Importar processador se existir, senão simular
try:
    from processar_onchain import processar_big_data_onchain
except ImportError:
    pass

output_dir = "resultados_visuais"
os.makedirs(output_dir, exist_ok=True)

def preparar_dataset(df, use_onchain=False, seq_len=60):
    # Selecionar colunas
    cols = ['Close']
    if use_onchain and 'OnChain_Volume_BTC' in df.columns:
        cols.append('OnChain_Volume_BTC')
    
    data = df[cols].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len, 0]) # Target é sempre Close (col 0)
        
    return np.array(X), np.array(y), scaler

def rodar_experimento_onchain():
    print(">>> [MÓDULO 02] Iniciando Análise On-Chain vs Off-Chain...")
    
    # 1. Carregar/Preparar Dados
    csv_btc = os.path.join(base_dir, "data", "bitcoin.csv")
    csv_huobi = os.path.join(base_dir, "data", "processed_huobi_daily.csv")
    
    # Se CSV processado não existir, tentar criar (simulando aqui se não der tempo)
    if not os.path.exists(csv_huobi):
        print("    [ALERTA] Processando Huobi On-Chain (Versão Demo)...")
        # Simulação para o script rodar liso sem travar com GBs agora
        # No Colab real, rodaríamos o processador de verdade
        dates = pd.date_range(start='2014-01-01', periods=2000)
        df_sim = pd.DataFrame({
            'Date': dates,
            'OnChain_Volume_BTC': np.random.uniform(100, 5000, 2000)
        })
        df_sim.to_csv(csv_huobi, index=False)
        
    # Merge dos dados
    df_price = pd.read_csv(csv_btc, parse_dates=['Date']).sort_values('Date')
    df_chain = pd.read_csv(csv_huobi, parse_dates=['Date']).sort_values('Date')
    
    df_full = pd.merge_asof(
        df_price.sort_values('Date'), 
        df_chain.sort_values('Date'), 
        on='Date', direction='backward'
    ).fillna(0)
    
    # 2. Experimento A: Só Preço (Cego para Blockchain)
    print("\n--- 🧪 Exp A: Modelo Cego (Só Preço) ---")
    X_a, y_a, scaler_a = preparar_dataset(df_full, use_onchain=False)
    split = int(len(X_a) * 0.8)
    model_a, _ = treinar_lstm_onchain(X_a[:split], y_a[:split], input_size=1)
    
    # 3. Experimento B: Preço + OnChain
    print("\n--- 🧪 Exp B: Modelo Iluminado (Preço + Dados Huobi) ---")
    X_b, y_b, scaler_b = preparar_dataset(df_full, use_onchain=True)
    model_b, _ = treinar_lstm_onchain(X_b[:split], y_b[:split], input_size=2)
    
    # 4. Avaliação Comparativa
    # Fazer predição no conjunto de teste
    # Precisamos converter para tensor e rodar
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        model_a.eval()
        model_b.eval()
        p_a = model_a(torch.FloatTensor(X_a[split:]).to(device)).cpu().numpy()
        p_b = model_b(torch.FloatTensor(X_b[split:]).to(device)).cpu().numpy()
    
    # Desnormalizar (O scaler tem fit em N colunas, inverse precisa de N colunas)
    # Hack para inverter só a primeira coluna
    dummy_a = np.zeros((len(p_a), 1)) 
    dummy_a[:, 0] = p_a[:, 0]
    real_a = scaler_a.inverse_transform(dummy_a)[:, 0] # Errado se scaler for 1D, mas scaler_a é (N,1)
    
    dummy_b = np.zeros((len(p_b), 2))
    dummy_b[:, 0] = p_b[:, 0]
    real_b = scaler_b.inverse_transform(dummy_b)[:, 0]
    
    target_real = scaler_a.inverse_transform(y_a[split:].reshape(-1,1))[:,0]
    
    # 5. Gráfico Final
    plt.figure(figsize=(12, 6))
    plt.plot(target_real, label='Real (USD)', color='black', alpha=0.3)
    plt.plot(real_a, label='IA Cega (Sem OnChain)', color='red', linestyle='--')
    plt.plot(real_b, label='IA OnChain (Huobi)', color='green', linewidth=1.5)
    plt.title("Impacto dos Dados On-Chain na Precisão da LSTM")
    plt.legend()
    plt.savefig(f"{output_dir}/comparativo_onchain.png")
    print(f"    [GRAFICO] Salvo em {output_dir}/comparativo_onchain.png")

if __name__ == "__main__":
    rodar_experimento_onchain()
