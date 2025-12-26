
import pandas as pd
import os
import gc

def processar_big_data_onchain(input_file, output_file):
    """
    Kneads Big Data into Small Data.
    Lê arquivos gigantes (GBs) de transações bloco a bloco e 
    converte em um CSV diário (KB) com Volume Total Agregado.
    """
    print(f"🚀 Iniciando processamento de: {os.path.basename(input_file)}")
    print(f"💾 Saída será salva em: {output_file}")
    
    # Prepara um dicionário para acumular os volumes por dia
    # { '2024-01-01': 500.2, '2024-01-02': 1020.5 ... }
    daily_volume = {}
    
    # Ler em chunks de 100.000 linhas para não explodir a RAM
    chunk_size = 100000
    total_rows = 0
    
    try:
        # Tenta detectar se o arquivo tem cabeçalho comentado ou pula linhas iniciais
        # O arquivo do WalletExplorer tem um header chato na linha 1
        for chunk in pd.read_csv(input_file, chunksize=chunk_size, skiprows=1):
            
            # Converter coluna date
            # Formato esperado: '2025-01-28 07:26:42'
            chunk['date'] = pd.to_datetime(chunk['date']).dt.date
            
            # Garantir que amounts sejam numéricos
            cols_to_sum = ['received amount', 'sent amount']
            for col in cols_to_sum:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
            
            # Agrupar este chunk por data
            # Somamos received + sent para ter o "Volume Movimentado Total"
            chunk['total_vol'] = chunk['received amount'] + chunk['sent amount']
            agg = chunk.groupby('date')['total_vol'].sum()
            
            # Adicionar ao acumulador global
            for data, vol in agg.items():
                date_str = str(data)
                daily_volume[date_str] = daily_volume.get(date_str, 0) + vol
            
            total_rows += len(chunk)
            print(f"   Processadas {total_rows} linhas...", end='\r')
            
            # Limpar memória
            del chunk
            gc.collect()
            
    except Exception as e:
        print(f"\n❌ Erro crítico: {e}")
        return

    print(f"\n✅ Leitura concluída! Salvando CSV agregado...")
    
    # Converter dicionário para DataFrame final e salvar
    df_final = pd.DataFrame(list(daily_volume.items()), columns=['Date', 'OnChain_Volume_BTC'])
    df_final = df_final.sort_values('Date')
    df_final.to_csv(output_file, index=False)
    
    print(f"🎉 Sucesso! Arquivo gerado: {output_file}")
    print(df_final.head())

if __name__ == "__main__":
    # Exemplo de uso: Rodar apenas se chamado diretamente
    # Caminhos relativos assumindo execução da raiz
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # Lista de arquivos para processar (Exemplo: Huobi que é menor p/ teste local)
    target = "walletexplorer-Huobi_com-000012a55e988d91.csv"
    input_path = os.path.join(data_dir, target)
    output_path = os.path.join(data_dir, "processed_huobi_daily.csv")
    
    if os.path.exists(input_path):
        processar_big_data_onchain(input_path, output_path)
    else:
        print(f"Arquivo não encontrado: {input_path}")
