
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import seaborn as sns
import time
import warnings

# Configurações Globais
RESULTS_DIR = "resultados_benchmark_5_arenas"

# Detectar Colab para ajustar DATA_DIR
if os.path.exists('/content'):
    print("☁️ Ambiente Google Colab detectado (Filesystem)!")
    DATA_DIR = "/content/drive/MyDrive" # Raiz do Drive para busca recursiva
else:
    DATA_DIR = "data"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================================================================================
# UTILITÁRIOS
# ==================================================================================
def setup_dirs():
    for arena in ["arena_1_timeseries", "arena_2_seasonal", "arena_3_trading", "arena_4_exchanges", "arena_5_robustness"]:
        os.makedirs(os.path.join(RESULTS_DIR, arena), exist_ok=True)

def load_data():
    setup_dirs()
    
    # Define caminhos de busca priorizados
    search_paths = [DATA_DIR]
    
    # Se estiver no Colab e detectamos a pasta do mestrado, adiciona ela na frente
    if DATA_DIR.startswith("/content"):
        potential_path = "/content/drive/MyDrive/mestrado business intelligence e big data"
        if os.path.exists(potential_path):
            print(f"   🎯 Caminho alvo detectado: {potential_path}")
            search_paths.insert(0, potential_path)

    # --- 1. CARREGAR PREÇO BITCOIN ---
    path_price = None
    
    # Função auxiliar para busca recursiva
    def find_files_recursive(directories, pattern_func):
        found = []
        if isinstance(directories, str): directories = [directories]
        
        for d in directories:
            print(f"      🏁 Iniciando busca recursiva em: {d}")
            count_dirs_visited = 0
            for root, dirs, files in os.walk(d):
                count_dirs_visited += 1
                # Debug leve para não spammar
                if count_dirs_visited % 10 == 0:
                    print(f"         ... varrendo subpasta #{count_dirs_visited}: {os.path.basename(root)}")
                
                for file in files:
                    if pattern_func(file):
                        full_path = os.path.join(root, file)
                        # print(f"         ✨ Achei candidato: {file}")
                        found.append(full_path)
            print(f"      ✅ Fim da busca em {d}. Pastas visitadas: {count_dirs_visited}. Arquivos achados: {len(found)}")
            
            if found: break # Se achou em um diretorio, para
        return found

    # Prioriza CSVs que parecem ser o historico de preço (IGNORANDO O PROBLEMATICO)
    candidates = find_files_recursive(
        search_paths, 
        lambda f: f.lower().startswith('bitcoin') and f.lower().endswith('.csv') and "bitcoin_15_10" not in f.lower()
    )

    if candidates:
        path_price = candidates[0] # Pega o primeiro (ex: bitcoin_2010...csv) que achar
    else:
        # Fallback especifico
        manual_check = os.path.join(DATA_DIR, "bitcoin.csv")
        if os.path.exists(manual_check):
            path_price = manual_check
        
    if not path_price:
        print("❌ Erro: Nenhum arquivo de preço (bitcoin*.csv) encontrado na árvore de diretórios.")
        return None, None 

    print(f"   📂 Carregando PREÇO: {os.path.basename(path_price)}")
    print(f"   📂 Carregando PREÇO: {os.path.basename(path_price)}")
    
    # Tentativa 1: Leitura padrão
    try:
        df = pd.read_csv(path_price)
        if len(df.columns) < 2:
            # Se só tem 1 coluna, provavelmente o separador está errado (era ; e leu como ,)
            print("      ⚠️ Detectado possível separador ';'. Tentando recarregar...")
            df = pd.read_csv(path_price, sep=';')
    except:
        # Tentativa forçada com ;
        df = pd.read_csv(path_price, sep=';')

    # Normalização Específica para o arquivo do User (CoinMarketCap e outros)
    # Remove aspas extras
    df.columns = [c.replace('"', '').strip() for c in df.columns]
    
    # 1. Identificar coluna de DATA única
    col_date_found = None
    possible_date_cols = ['timestamp', 'timeOpen', 'Start', 'snapped_at', 'date']
    
    for c in df.columns:
        if c in possible_date_cols:
            col_date_found = c
            break
            
    if not col_date_found:
        # Tenta achar alguma coluna que começa com 'time' (ex: timeHigh) se não achou as principais
        for c in df.columns:
            if c.lower().startswith('time') or 'date' in c.lower():
                col_date_found = c
                break

    if col_date_found:
        print(f"      🗓️ Coluna de data identificada: {col_date_found}")
        df.rename(columns={col_date_found: 'Date'}, inplace=True)
        # Remove outras colunas de tempo para não confundir
        cols_to_drop = [c for c in df.columns if c.startswith('time') and c != 'Date']
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
    
    # 2. Mapear OHLCV
    map_ohlcv = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        'marketCap': 'MarketCap'
    }
    df.rename(columns=map_ohlcv, inplace=True)
        
    # Garantir datetime
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except:
        # Se falhar, tenta coluna 0
        print("      ⚠️ Aviso: Coluna 'Start/Date' falhou. Tentando coluna 0.")
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values('Date').set_index('Date')
    
    # Garantir Close
    if 'Close' in df.columns:
        df = df[['Close']] # Fica só com o Close
    else:
        # Fallback para ultima coluna se não achar Close
        print("      ⚠️ Aviso: Coluna 'Close' não achada. Usando última coluna.")
        df = df.iloc[:, -1:]
        df.columns = ['Close']

    print(f"      ✅ Preço carregado. Registros: {len(df)}")

    # --- 2. CARREGAR BIG DATA (WALLET EXPLORER) ---
    print("   🔍 Buscando BIG DATA (WalletExplorer) recursivamente...")
    
    stats = [] # Inicializa lista de estatisticas
    big_files = find_files_recursive(search_paths, lambda f: "walletexplorer" in f.lower() and f.endswith(".csv"))
    
    if not big_files:
        print("      ⚠️ Nenhum arquivo 'walletexplorer' encontrado.")
    
    for path_oc in big_files:
        f_name = os.path.basename(path_oc)
        try:
            print(f"      🏗️ Integrando: {f_name} ...")
            
            # Detecção de Header Dinâmica
            header_idx = 0
            found_header = False
            with open(path_oc, 'r', encoding='utf-8', errors='ignore') as f:
                for i in range(20):
                    line = f.readline().lower()
                    if 'date' in line and ('received' in line or 'balance' in line):
                        header_idx = i
                        found_header = True
                        break
            
            if not found_header:
                print(f"         ⚠️ Header não detectado. Tentando padrão...")
            
            # Carrega com pandas
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                df_oc = pd.read_csv(path_oc, header=header_idx, low_memory=False, on_bad_lines='skip')
            
            # Normaliza nomes de colunas
            df_oc.columns = [c.strip().lower() for c in df_oc.columns]
            
            # Identifica Data e Valor
            col_d = 'date' if 'date' in df_oc.columns else None
            cols_val = [c for c in df_oc.columns if 'received' in c or 'sent' in c]
            
            if col_d and cols_val:
                # Limpeza robusta de data
                df_oc[col_d] = df_oc[col_d].astype(str).str.strip()
                try:
                    df_oc[col_d] = pd.to_datetime(df_oc[col_d], format='%Y-%m-%d %H:%M:%S', errors='raise')
                except:
                    df_oc[col_d] = pd.to_datetime(df_oc[col_d], errors='coerce', dayfirst=False)

                df_oc.dropna(subset=[col_d], inplace=True)
                
                # Agrupa por dia
                df_oc['DateDate'] = df_oc[col_d].dt.floor('D')
                for c in cols_val:
                    df_oc[c] = pd.to_numeric(df_oc[c], errors='coerce').fillna(0)
                
                daily_vol = df_oc.groupby('DateDate')[cols_val].sum().sum(axis=1)
                daily_vol.name = f"Vol_{f_name.split('_')[1] if '_' in f_name else f_name[:10]}"
                
                df = df.join(daily_vol, how='left').fillna(0)
                stats.append(f"{daily_vol.name}: {len(daily_vol)} dias integrados")
                print(f"         ✅ {daily_vol.name} integrado! ({len(daily_vol)} dias)")
            else:
                print(f"         ❌ Colunas invalidas. Achadas: {df_oc.columns}")

        except Exception as e:
            print(f"         ❌ Erro ao processar {f_name}: {e}")

    # Remove dias sem preço
    df = df[df['Close'] > 0]
    
    return df, stats # Retorna tupla corretamente

# ==================================================================================
# 1. ARQUITETURAS DE IA (5 ALGORITMOS)
# ==================================================================================

# 1.1 MLP (Rede Neural Feedforward Clássica)
class WrapperMLP(nn.Module):
    def __init__(self, input_dim, mode='regressor'):
        super().__init__()
        self.name = "MLP (Neural)"
        self.mode = mode
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        if self.mode == 'classifier':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def fit(self, X, y, epochs=50):
        if len(X.shape) == 3: X = X.reshape(X.shape[0], -1)
        
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        self.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(X_t)
            loss = self.loss_fn(out, y_t)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        if len(X.shape) == 3: X = X.reshape(X.shape[0], -1)
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            out = self.model(X_t)
            if self.mode == 'classifier':
                prob = torch.sigmoid(out)
                return torch.round(prob).cpu().numpy().flatten()
            return out.cpu().numpy().flatten()

# 1.2 LSTM (Deep Learning)
class WrapperLSTM(nn.Module):
    def __init__(self, input_dim, mode='regressor'):
        super().__init__()
        self.name = "LSTM (Deep Learning)"
        self.mode = mode
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(64, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.003)
        
        if self.mode == 'classifier':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def fit(self, X, y, epochs=50):
        # Ajuste inteligente de shape para RNNs
        if len(X.shape) == 2: 
            # Se for tabular (N, Features), vira (N, 1, Features)
            # Isso trata Arena 2 corretamente
            X = X.reshape(X.shape[0], 1, X.shape[1])
        elif len(X.shape) == 3 and X.shape[2] == 1 and self.lstm.input_size > 1:
             # Correção para caso venha (N, T, 1) mas o modelo espere (N, T, Features)
             pass

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        self.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            out, _ = self.lstm(X_t)
            logits = self.fc(out[:, -1, :])
            loss = self.loss_fn(logits, y_t)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        if len(X.shape) == 2: 
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            out, _ = self.lstm(X_t)
            logits = self.fc(out[:, -1, :])
            if self.mode == 'classifier':
                return torch.round(torch.sigmoid(logits)).cpu().numpy().flatten()
            return logits.cpu().numpy().flatten()

# 1.3 Temporal Fusion Transformer
class WrapperTFT(nn.Module):
    def __init__(self, input_dim, window_size, mode='regressor'):
        super().__init__()
        self.name = "TemporalFusionTransformer"
        self.mode = mode
        self.d_model = 32
        
        self.lstm_encoder = nn.LSTM(input_dim, self.d_model, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Linear(self.d_model, 1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        if self.mode == 'classifier':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def fit(self, X, y, epochs=50):
        if len(X.shape) == 2: 
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        self.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            lstm_out, _ = self.lstm_encoder(X_t)
            trans_out = self.transformer_encoder(lstm_out)
            logits = self.fc(trans_out[:, -1, :])
            loss = self.loss_fn(logits, y_t)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        if len(X.shape) == 2: 
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            lstm_out, _ = self.lstm_encoder(X_t)
            trans_out = self.transformer_encoder(lstm_out)
            logits = self.fc(trans_out[:, -1, :])
            if self.mode == 'classifier':
                return torch.round(torch.sigmoid(logits)).cpu().numpy().flatten()
            return logits.cpu().numpy().flatten()

# 1.4 Genetic Algorithm
class WrapperGenetic:
    def __init__(self, input_dim, mode='regressor'):
        self.name = "GeneticAlgorithm"
        self.mode = mode
        self.input_dim = input_dim
        self.best_weights = None
        self.pop_size = 50
        self.generations = 10
        
    def fit(self, X, y, epochs=None):
        if len(X.shape) > 2: X = X.reshape(X.shape[0], -1)
        n_features = X.shape[1]
        population = np.random.uniform(-1, 1, (self.pop_size, n_features + 1))
        
        for g in range(self.generations):
            scores = []
            for ind in population:
                w, b = ind[:-1], ind[-1]
                pred = np.dot(X, w) + b
                if self.mode == 'classifier':
                    pred = 1 / (1 + np.exp(-pred))
                    score = np.mean((y - pred)**2)
                else:
                    score = np.mean((y - pred)**2)
                scores.append(score)
            
            sorted_idx = np.argsort(scores)
            elites = population[sorted_idx[:int(self.pop_size * 0.2)]]
            
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = elites[np.random.randint(0, len(elites))]
                p2 = elites[np.random.randint(0, len(elites))]
                mask = np.random.rand(n_features + 1) > 0.5
                child = np.where(mask, p1, p2)
                child += (np.random.normal(0, 0.1, n_features + 1) * (np.random.rand(n_features + 1) < 0.05))
                new_pop.append(child)
            population = np.array(new_pop)
            
        self.best_weights = elites[0]

    def predict(self, X):
        if len(X.shape) > 2: X = X.reshape(X.shape[0], -1)
        w, b = self.best_weights[:-1], self.best_weights[-1]
        logits = np.dot(X, w) + b
        if self.mode == 'classifier':
            prob = 1 / (1 + np.exp(-logits))
            return np.round(prob)
        return logits

# 1.5 XGBoost
class WrapperXGB:
    def __init__(self, mode='regressor'):
        self.name = "XGBoost (Machine Learning)"
        self.mode = mode
        if mode == 'classifier':
            self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
        else:
            self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, objective='reg:squarederror')

    def fit(self, X, y, epochs=None):
        if len(X.shape) > 2: X = X.reshape(X.shape[0], -1)
        self.model.fit(X, y)

    def predict(self, X):
        if len(X.shape) > 2: X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

def get_all_models(window_size, n_features=1, mode='regressor'):
    return [
        WrapperMLP(input_dim=window_size * n_features, mode=mode),
        WrapperLSTM(input_dim=n_features, mode=mode),
        WrapperTFT(input_dim=n_features, window_size=window_size, mode=mode),
        WrapperXGB(mode=mode),
        WrapperGenetic(input_dim=window_size * n_features, mode=mode)
    ]

# ==================================================================================
# ARENAS
# ==================================================================================

def run_arena_1_timeseries(df):
    print("\n🥊 >>> ARENA 1: SÉRIES TEMPORAIS (TODOS OOS)")
    save_path = os.path.join(RESULTS_DIR, "arena_1_timeseries")
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[['Close']].values)
    window = 30
    
    X, y = [], []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    X, y = np.array(X), np.array(y)
    
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    
    models = get_all_models(window)
    metrics = []
    
    for model in models:
        t0 = time.time()
        try:
            model.fit(X_train, y_train, epochs=20)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            
            metrics.append({
                "Arena": "1_TimeSeries", "Cenario": "Base",
                "Modelo": model.name, "MSE": mse, "Tempo_Exec": time.time()-t0
            })
            print(f"   📊 {model.name}: MSE={mse:.6f}")
            
            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(y_test, label='Real')
            plt.plot(pred, label='Predito')
            plt.title(f"Arena 1: {model.name}")
            plt.legend()
            safe_name = model.name.replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(os.path.join(save_path, f"{safe_name}.png"))
            plt.close()
        except Exception as e:
            print(f"   ❌ Erro {model.name}: {e}")
            
    return metrics

def run_arena_2_seasonal(df):
    print("\n🥊 >>> ARENA 2: CLASSIFICAÇÃO SAZONAL (TODOS VS TODOS)")
    save_path = os.path.join(RESULTS_DIR, "arena_2_seasonal")
    
    # Feature Engineering
    df_s = df.copy()
    df_s['Ret'] = df_s['Close'].pct_change()
    df_s['Target'] = (df_s['Ret'] > 0).astype(int)
    df_s['DoW'] = df_s.index.dayofweek
    df_s['Month'] = df_s.index.month
    df_s.dropna(inplace=True)
    
    X = df_s[['DoW', 'Month']].values
    y = df_s['Target'].values
    
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    
    # Busca TODOS em modo classificador
    classifiers = get_all_models(window_size=1, n_features=2, mode='classifier')
    
    metrics = []
    
    for model in classifiers:
        t0 = time.time()
        try:
            model.fit(X_train, y_train, epochs=20)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            
            metrics.append({
                "Arena": "2_Seasonal", "Cenario": "Classificação",
                "Modelo": model.name, "MSE": 0, "Acuracia": acc, "Tempo_Exec": time.time()-t0
            })
            print(f"   📊 {model.name}: {acc:.1%} Acc")
            
            try:
                cm = confusion_matrix(y_test, pred)
                plt.figure(figsize=(4,4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
                plt.title(f'CM - {model.name}')
                safe_name = model.name.replace(" ", "_").replace("(", "").replace(")", "")
                plt.savefig(os.path.join(save_path, f"Arena2_CM_{safe_name}.png"))
                plt.close()
            except: pass
        except Exception as e:
            print(f"   ❌ Falha em {model.name}: {e}")

    return metrics

def run_arena_3_trading(df):
    print("\n🥊 >>> ARENA 3: TRADING BOT (SIMULAÇÃO DE LUCRO)")
    save_path = os.path.join(RESULTS_DIR, "arena_3_trading")
    
    # Lógica simplificada de trading: Preço sobe -> Compra
    # Avaliamos o retorno acumulado
    
    models = get_all_models(window_size=30, n_features=1, mode='classifier')
    
    # Prepara dados (janela deslizante mas target é 'subiu/desceu' futuro)
    scaler = MinMaxScaler()
    noise_close = scaler.fit_transform(df[['Close']].values)
    
    window = 30
    X, y_dir, prices = [], [], []
    
    # y_dir = 1 se preço[t+1] > preço[t], else 0
    actual_prices = df['Close'].values
    
    for i in range(len(noise_close)-window-1):
        X.append(noise_close[i:i+window])
        # Alvo: direção do proximo dia
        is_up = 1 if actual_prices[i+window+1] > actual_prices[i+window] else 0
        y_dir.append(is_up)
        prices.append(actual_prices[i+window+1])
        
    X = np.array(X)
    y_dir = np.array(y_dir)
    prices = np.array(prices)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_dir[:split], y_dir[split:]
    p_test = prices[split:]
    
    metrics = []
    
    for model in models:
        try:
            model.fit(X_train, y_train, epochs=20)
            # Predição (Classificação: Vai subir?)
            signals = model.predict(X_test) # 0 ou 1
            
            # Backtest simples
            initial_wallet = 1000
            wallet = initial_wallet
            position = 0 # 0 usd, 1 btc
            
            history_wallet = []
            
            for i in range(len(signals)):
                price = p_test[i]
                sig = signals[i]
                
                if sig == 1 and position == 0:
                    position = wallet / price
                    wallet = 0
                elif sig == 0 and position > 0:
                    wallet = position * price
                    position = 0
                
                curr_val = wallet if wallet > 0 else position * price
                history_wallet.append(curr_val)
                
            final_val = history_wallet[-1]
            profit = (final_val - initial_wallet) / initial_wallet
            
            metrics.append({
                "Arena": "3_Trading", "Cenario": "Backtest",
                "Modelo": model.name, "MSE": 0, "Acuracia": profit # Usando campo acuracia para guardar ROI
            })
            print(f"   💰 {model.name}: ROI {profit:.2%}")
            
            plt.figure(figsize=(10,4))
            plt.plot(history_wallet, label='Saldo Carteira')
            plt.title(f"Equity Curve - {model.name} (ROI: {profit:.1%})")
            safe_name = model.name.replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(os.path.join(save_path, f"Arena3_Equity_{safe_name}.png"))
            plt.close()
            
        except Exception as e:
            print(f"   ❌ Erro Trading {model.name}: {e}")
            
    return metrics

def run_arena_4_exchanges(df_base):
    print("\n🥊 >>> ARENA 4: DATA QUALITY (TODOS OS MODELOS)")
    save_path = os.path.join(RESULTS_DIR, "arena_4_exchanges")
    
    exchanges = ['Binance', 'Kraken', 'Huobi']
    results = []
    
    for ex in exchanges:
        print(f"   📡 Exchange: {ex}")
        np.random.seed(len(ex))
        noise = np.random.normal(1, 0.002, len(df_base)) 
        df_ex = df_base.copy()
        df_ex['Close'] = df_ex['Close'] * noise
        
        scaler = MinMaxScaler()
        d = scaler.fit_transform(df_ex[['Close']].values)
        X, y = [], []
        for i in range(len(d)-30):
            X.append(d[i:i+30])
            y.append(d[i+30])
        X, y = np.array(X), np.array(y)
        split = int(len(X)*0.8)
        
        models = get_all_models(window_size=30, n_features=1)
        
        for model in models:
            try:
                model.fit(X[:split], y[:split], epochs=10)
                pred = model.predict(X[split:])
                mse = mean_squared_error(y[split:], pred)
                
                results.append({"Arena": "4_Exchanges", "Cenario": ex, "Modelo": model.name, "MSE": mse})
            except Exception as e:
                print(f"      ❌ Erro {model.name}: {e}")
                
    return results

def run_arena_5_robustness(df):
    print("\n🥊 >>> ARENA 5: ROBUSTEZ (TODOS VS FUTURO)")
    save_path = os.path.join(RESULTS_DIR, "arena_5_robustness")
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[['Close']].values)
    window = 30
    
    X, y = [], []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    X, y = np.array(X), np.array(y)
    
    split = int(len(X) * 0.6)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:] # 40% OOS
    
    models = get_all_models(window)
    res = []
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(y_test)), y_test, label='Real (Futuro)', color='black', alpha=0.3, linewidth=3)
    
    for model in models:
        print(f"   🏗️ Treinando {model.name} no passado...")
        try:
            model.fit(X_train, y_train, epochs=20)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            
            res.append({"Arena": "5_Robustness", "Cenario": "OOS", "Modelo": model.name, "MSE": mse})
            plt.plot(pred, label=f'{model.name}', alpha=0.8)
            
            try:
                plt.figure(figsize=(10, 5))
                plt.plot(y_test, label='Real (Futuro)', color='black', alpha=0.5, linewidth=2)
                plt.plot(pred, label=f'Predição {model.name}', color='red', linestyle='--')
                plt.title(f"Robustez Individual: {model.name} (Out-of-Sample)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                safe_name = model.name.replace(" ", "_").replace("(", "").replace(")", "")
                plt.savefig(os.path.join(save_path, f"Arena5_Robustness_{safe_name}.png"))
                plt.close()
            except: pass
        except:
            pass

    plt.title("Generalização Temporal (Quem sobreviveu?)")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Arena5_Robustness.png"))
    plt.close()
    return res

def main():
    setup_dirs()
    print(">>> INICIANDO BENCHMARK CIENTÍFICO (5 ARENAS) <<<")
    
    df, data_stats = load_data()
    
    if df is None or df.empty:
        print("❌ Abortando: Dados insuficientes.")
        return

    with open(os.path.join(RESULTS_DIR, "big_data_stats.txt"), "w") as f:
        for stat in data_stats:
            f.write(stat + "\n")

    full_metrics = []

    full_metrics.extend(run_arena_1_timeseries(df))
    full_metrics.extend(run_arena_2_seasonal(df))
    full_metrics.extend(run_arena_3_trading(df))
    full_metrics.extend(run_arena_4_exchanges(df))
    full_metrics.extend(run_arena_5_robustness(df))
    
    df_metrics = pd.DataFrame(full_metrics)
    metric_path = os.path.join(RESULTS_DIR, "benchmark_5_arenas_metrics.csv")
    df_metrics.to_csv(metric_path, index=False)
    print(f"\n✅ BENCHMARK CIENTÍFICO FINALIZADO!")
    print(f"   📊 Métricas salvas em: {metric_path}")

if __name__ == "__main__":
    main()
