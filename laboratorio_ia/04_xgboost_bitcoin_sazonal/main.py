
import sys
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Adicionar diretório pai ao path para importar data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_and_process_data

def train_xgboost():
    print("="*50)
    print("[INI] MODELO 1: MACHINE LEARNING (XGBOOST)")
    print("="*50)

    # 1. Carregar Dados Padronizados
    print("[INFO] Carregando dados via data_loader...")
    
    # Caminho absoluto para evitar erros
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'bitcoin.csv')
    
    data = load_and_process_data(csv_path, seq_length=60)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    scaler = data['scaler']

    # 2. Ajustar Formato para XGBoost
    # XGBoost precisa de (Amostras, Features). Nossos dados estão em (Amostras, 60, 1).
    # Vamos "achatar" para (Amostras, 60).
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    print(f"[INFO] Formato de Treino: {X_train.shape}")
    print(f"[INFO] Formato de Teste: {X_test.shape}")

    # 3. Criar e Treinar Modelo
    print("[INFO] Treinando XGBoost (pode demorar alguns segundos)...")
    model = XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.01, 
        max_depth=6, 
        early_stopping_rounds=20,
        n_jobs=-1
    )
    
    # XGBoost precisa de eval_set para early_stopping
    model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=False
    )
    
    # 4. Previsão
    print("[INFO] Gerando previsoes...")
    preds_scaled = model.predict(X_test)
    
    # Desfazer normalização para ver preços reais
    preds_real = scaler.inverse_transform(preds_scaled.reshape(-1, 1))
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 5. Métricas
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    mae = mean_absolute_error(y_test_real, preds_real)
    
    print("-" * 30)
    print(f"[RES] RESULTADOS PARA XGBOOST:")
    print(f"   RMSE (Erro Médio Quadrático): ${rmse:.2f}")
    print(f"   MAE  (Erro Médio Absoluto):   ${mae:.2f}")
    print("-" * 30)
    
    # 6. Salvar Gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label='Preço Real', color='blue')
    plt.plot(preds_real, label='Previsão XGBoost', color='red', alpha=0.7)
    plt.title('Previsão de Bitcoin - XGBoost')
    plt.xlabel('Dias (Conjunto de Teste)')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(True)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, 'result_xgboost.png')
    plt.savefig(plot_path)
    print(f"[GRAFICO] Salvo em: {plot_path}")
    # plt.show() # Comentado para não travar automação, mas pode descomentar

if __name__ == "__main__":
    train_xgboost()