
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ==========================================
# 1. ARQUITETURA DA REDE NEURAL (MLP)
# ==========================================
class BitcoinClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BitcoinClassifier, self).__init__()
        # Arquitetura clássica para classificação binária/regressão simples
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Evitar decorar os dados
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Saída: Previsão do Preço ou Tendência
        )
        
    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. FUNÇÃO DE TREINO OTIMIZADA
# ==========================================
def treinar_modelo_rna(X_train, y_train, epochs=200, lr=0.001):
    """
    Treina uma RNA para um cenário específico de dados.
    """
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    
    model = BitcoinClassifier(input_dim).to(device)
    criterion = nn.MSELoss() # Erro Quadrático Médio
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Converter para tensores
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device).unsqueeze(1) # (N, 1)
    
    # 2. Loop de Treino Silencioso (para rodar rápido em loop)
    model.train()
    history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        
    return model, history

# ==========================================
# 3. PREVISÃO
# ==========================================
def fazer_previsao(model, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    return preds
