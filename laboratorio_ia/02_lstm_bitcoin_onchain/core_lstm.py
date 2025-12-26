
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HybridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(HybridLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Pegar último passo
        return out

def treinar_lstm_onchain(X_train, y_train, input_size, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridLSTM(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device).unsqueeze(1)
    
    model.train()
    history = []
    
    print(f"    [CORE LSTM] Iniciando Treino (Input Size={input_size})...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outs = model(X_t)
        loss = criterion(outs, y_t)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        
    return model, history
