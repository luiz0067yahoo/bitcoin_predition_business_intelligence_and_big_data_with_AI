# 🧠 Bitcoin AI Benchmark: Protocolo de Validação em 5 Arenas

Este repositório contém o código-fonte e os resultados de um estudo avançado de **Inteligência Artificial aplicada a Criptoativos**, focado na fusão de dados de mercado (Preço) com Big Data On-Chain (Fluxo Institucional de Carteiras).

O projeto implementa o **"Protocolo de Validação em 5 Dimensões"**, submetendo 5 arquiteturas de IA a testes de estresse rigorosos.

---

## 🔬 Arquiteturas Avaliadas (Modelos)

1.  **Redes Neurais (MLP)**: Perceptron Multicamadas para captura de não-linearidades globais.
2.  **Deep Learning (LSTM)**: Redes Neurais Recorrentes (Long Short-Term Memory) para padrões sequenciais.
3.  **Temporal Fusion Transformer (TFT)**: Estado da arte em mecanismos de atenção para séries temporais.
4.  **Machine Learning (XGBoost)**: Gradient Boosting em árvores de decisão (focado em dados tabulares/sazonais).
5.  **Algoritmo Genético**: Otimização evolutiva de estratégias de trading (bio-inspirado).

## 🥊 As 5 Arenas de Validação

O sistema executa automaticamente 5 cenários experimentais:

*   **Arena 1 (Séries Temporais):** Minimização de erro quadrático (MSE) em janelas de tempo.
*   **Arena 2 (Sazonalidade):** Classificação de padrões de calendário (Dia da Semana/Mês).
*   **Arena 3 (Trading):** Simulação financeira de lucro/prejuízo (ROI) e curvas de equity.
*   **Arena 4 (Data Quality):** Teste de sensibilidade a ruídos em diferentes Exchanges.
*   **Arena 5 (Robustez):** Teste Out-of-Sample (Futuro Desconhecido) para medir Concept Drift.

---

## 🛠️ Como Rodar Localmente

Se você deseja reproduzir os experimentos em sua própria máquina, siga os passos abaixo.

### Pré-requisitos
*   Python 3.8 ou superior.
*   Git instalado.

### 1. Clonar o Repositório
```bash
git clone https://github.com/SEU_USUARIO/bitcoin-ai-benchmark.git
cd bitcoin-ai-benchmark
```

### 2. Instalar Dependências
Recomenda-se criar um ambiente virtual (venv):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

Instale as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```

### 3. Preparar os Dados
Crie uma pasta chamada `data` na raiz do projeto e coloque seus arquivos CSV lá:
*   `bitcoin_price.csv` (Histórico de preços)
*   Arquivos do WalletExplorer (opcional, para fusão on-chain)

> **Nota:** O script procura recursivamente por arquivos CSV válidos.

### 4. Executar o Benchmark
Para rodar todas as 5 Arenas sequencialmente:
```bash
python laboratorio_ia/benchmark_5_arenas.py
```

### 5. Gerar Relatório PDF/DOCX
Após a conclusão do benchmark, os resultados estarão na pasta `resultados_benchmark_5_arenas`. Para gerar o relatório técnico automatizado:
```bash
python gerar_relatorio_tecnico_final.py
```

---

## 📂 Estrutura do Projeto

*   `/laboratorio_ia`: Código-fonte principal das redes neurais e lógica das Arenas.
    *   `benchmark_5_arenas.py`: Script mestre que orquestra todo o experimento.
*   `/utils`: Utilitários para processamento de dados on-chain.
*   `gerar_relatorio_tecnico_final.py`: Gerador de documentos com os resultados.
*   `requirements.txt`: Lista de dependências.

---

## 📊 Resultados Esperados

Ao final da execução, você terá:
1.  **Métricas Consolidadas:** Arquivo CSV com MSE, Acurácia e ROI de todos os modelos.
2.  **Galeria de Gráficos:**
    *   Curvas de Predição (Real vs Previsto)
    *   Matrizes de Confusão (Sazonalidade)
    *   Curvas de Equity (Lucro acumulado)
    *   Testes de Estresse (Projeções futuras)

---
**Autor:** Felipe (Mestrado em Business Intelligence & Big Data)
