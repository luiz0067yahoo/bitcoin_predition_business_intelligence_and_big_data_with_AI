
import os
import pandas as pd
import glob
import warnings

import sys

# Detectar se está no Colab (Checagem robusta via FileSystem)
if os.path.exists('/content'):
    print("☁️ Ambiente Colab Detectado (Filesystem Check).")
    # No Colab, o Drive já deve estar montado pelo passo 1 do script
    DATA_DIR = "/content/drive/MyDrive"
    
    # Validação de Segurança
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ Alerta: Drive não parece montado em {DATA_DIR}. Tentando montar...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except:
            print("❌ Falha ao montar Drive. Verifique se deu permissão.")
    
    # DEBUG: Listar pastas na raiz para ver o que o Colab enxerga
    print(f"📂 Listando pastas na raiz de {DATA_DIR}:")
    try:
        root_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        for d in root_dirs[:15]: # Mostra as 15 primeiras pastas
            print(f"   - {d}")
    except Exception as e:
        print(f"   ❌ Erro ao listar raiz: {e}")
else:
    # Local
    DATA_DIR = "data"

def auditar_csvs():
    print("=======================================================")
    print(f"🕵️  AUDITORIA DE DADOS ON-CHAIN e PREÇO")
    print("=======================================================")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Erro: Pasta '{DATA_DIR}' não existe.")
        return

    # Função auxiliar para busca recursiva
    def find_files_recursive(directory, pattern_func):
        found = []
        for root, _, files in os.walk(directory):
            for file in files:
                if pattern_func(file):
                    found.append(os.path.join(root, file))
        return found

    # 1. Auditar Preço
    print(f"   🔍 Buscando arquivo de preço (bitcoin*.csv) em {DATA_DIR}...")
    files_price = find_files_recursive(DATA_DIR, lambda f: f.lower().startswith('bitcoin') and f.lower().endswith('.csv'))
    
    if files_price:
        f_price = files_price[0] # Pega o primeiro encontrado
        print(f"   ✅ Arquivo de preço encontrado: {os.path.basename(f_price)}")
        try:
            df = pd.read_csv(f_price)
            # Tenta identificar coluna de data
            col_date = 'Date' if 'Date' in df.columns else ('Start' if 'Start' in df.columns else df.columns[0])
            df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
            validos = df[col_date].notna().sum()
            total = len(df)
            
            print(f"\n💰 PREÇO BITCOIN ({os.path.basename(f_price)})")
            print(f"   -> Total Linhas: {total}")
            print(f"   -> Datas Válidas: {validos} ({validos/total:.1%})")
            if validos > 0:
                print(f"   -> Periodo: {df[col_date].min().date()} até {df[col_date].max().date()}")
        except Exception as e:
            print(f"   ❌ Erro ao ler preço: {e}")
    else:
        print("   ❌ NENHUM arquivo de preço encontrado!")
        print("   🕵️  Modo Detetive: Listando TODOS os CSVs encontrados para debug...")
        all_csvs = find_files_recursive(DATA_DIR, lambda f: f.lower().endswith('.csv'))
        if all_csvs:
            print(f"      Encontrei {len(all_csvs)} arquivos CSV. Veja os primeiros 20:")
            for c in all_csvs[:20]:
                print(f"      - {c}")
        else:
            print("      ❌ NENHUM CSV encontrado em lugar nenhum! Verifique se seu Drive tem arquivos.")

    # 2. Auditar WalletExplorer (Big Data)
    print("\n🐋 BIG DATA ON-CHAIN (WalletExplorer)")
    big_files = find_files_recursive(DATA_DIR, lambda f: "walletexplorer" in f.lower() and f.endswith(".csv"))
    
    if not big_files:
        print("   ⚠️ Nenhum arquivo 'walletexplorer' encontrado.")
    
    for path in big_files:
        fname = os.path.basename(path)
        print(f"\n📄 Aquivo: {fname}")
        
        try:
            # Contagem bruta de linhas (rápida)
            with open(path, 'rb') as f:
                total_linhas_raw = sum(1 for _ in f)
            print(f"   -> Linhas Brutas (Total Físico): {total_linhas_raw}")
            
            # Simulação da Carga Inteligente (Igual ao Benchmark)
            header_idx = 0
            found_header = False
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for i in range(20): # Scanear 20 linhas
                    line = f.readline().lower()
                    if 'date' in line and ('received' in line or 'balance' in line):
                        header_idx = i
                        found_header = True
                        break
            
            if not found_header:
                print("   ❌ Header não detectado nas primeiras 20 linhas!")
                continue
                
            print(f"   -> Header detectado na linha: {header_idx}")
            
            # Carga com Pandas
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Carrega dinamicamente, sem forçar nomes fixos, mas normaliza depois
                df_oc = pd.read_csv(path, header=header_idx, low_memory=False, on_bad_lines='skip')
                
            # Normalizar Colunas
            df_oc.columns = [c.strip().lower() for c in df_oc.columns]
            
            col_date = 'date' if 'date' in df_oc.columns else df_oc.columns[0]
            
            # Nova Logica Robusta do Benchmark (Blindada)
            df_oc[col_date] = df_oc[col_date].astype(str).str.strip()
            
            # 1. Tenta formato padrao rapido
            try:
                # Tenta converter inferindo, mas forçando erros se não der
                # O benchmark usa format='%Y-%m-%d %H:%M:%S', vamos tentar ele
                # Se falhar, o except pega o coerce
                dates_fast = pd.to_datetime(df_oc[col_date], format='%Y-%m-%d %H:%M:%S', errors='raise')
                df_oc['dt_clean'] = dates_fast
            except:
                # 2. Fallback lento mas seguro
                df_oc['dt_clean'] = pd.to_datetime(df_oc[col_date], errors='coerce', dayfirst=False)

            df_valid = df_oc.dropna(subset=['dt_clean'])
            
            validos = len(df_valid)
            total_pandas = len(df_oc)
            trash = total_pandas - validos
            
            print(f"   -> Linhas Carregadas: {total_pandas}")
            print(f"   -> 🗑️ Lixo DESCARTADO: {trash}")
            print(f"   -> ✅ Linhas VÁLIDAS: {validos} ({validos/total_linhas_raw:.1%})")
            
            if validos > 0:
                print(f"   -> Periodo: {df_valid['dt_clean'].min().date()} até {df_valid['dt_clean'].max().date()}")
                
                # Tenta somar volume se achar coluna received
                cols_vol = [c for c in df_oc.columns if 'received' in c]
                if cols_vol:
                     vol = pd.to_numeric(df_valid[cols_vol[0]], errors='coerce').sum()
                     print(f"   -> Volume '{cols_vol[0]}' total: {vol:,.2f}")

        except Exception as e:
            print(f"   ❌ Erro ao auditar arquivo: {e}")

if __name__ == "__main__":
    auditar_csvs()
