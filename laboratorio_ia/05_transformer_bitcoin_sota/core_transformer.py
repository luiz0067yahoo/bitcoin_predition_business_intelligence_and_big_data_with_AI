
import torch
import torch.nn as nn
import torch.optim as optim
import math

class MiniTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(MiniTransformer, self).__init__()
        
        # Projetar input para dimensão do modelo
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding (Seno/Cosseno para dar nocão de tempo)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model)) 
        
        # Transformer Encoder (Captura padrões)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder Final
        self.decoder = nn.Linear(d_model, 1)
        
        # Armazenar pesos de atenção para visualização (Hack para fins didáticos)
        self.last_attention_weights = None

    def forward(self, x):
        # x shape: (Batch, SeqLen, Features)
        batch_size, seq_len, _ = x.size()
        
        # 1. Embed + Posição
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # 2. Passar pelo Transformer
        # (Nota: PyTorch padrão não retorna pesos de atenção fácil, então focaremos na performance)
        out = self.transformer_encoder(x)
        
        # 3. Pegar apenas o último elemento da sequência
        out = out[:, -1, :] 
        
        # 4. Previsão
        return self.decoder(out)

def treinar_transformer(X_train, y_train, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[2]
    
    model = MiniTransformer(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device).unsqueeze(1)
    
    model.train()
    print("    [TRANSFORMER] Treinando (Atenção Multicabeça)...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
    return model
