
import sys
import os
import subprocess
import time
import platform
import psutil
import torch
import datetime

# Configuração dos Módulos a Rodar
MODULES = [
    {
        "name": "01_Arbitragem_RNA",
        "path": "01_rna_bitcoin_arbitragem/run_experimentos.py"
    },
    {
        "name": "02_OnChain_LSTM",
        "path": "02_lstm_bitcoin_onchain/run_experimentos_onchain.py"
    },
    {
        "name": "03_Trading_Genetico",
        "path": "03_genetico_bitcoin_trading/run_trader_genetico.py"
    },
    {
        "name": "04_Sazonal_XGBoost",
        "path": "04_xgboost_bitcoin_sazonal/run_xgboost_sazonal.py"
    },
    {
        "name": "05_SOTA_Transformer",
        "path": "05_transformer_bitcoin_sota/run_atencao_visual.py"
    },
    {
        "name": "99_BENCHMARK_CIENTIFICO_5_ARENAS",
        "path": "benchmark_5_arenas.py"
    }
]

LOG_FILE = "execution_log.txt"

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")

def get_system_info():
    log("="*50)
    log(">>> AUDITORIA DE HARDWARE <<<")
    log(f"Sistema: {platform.system()} {platform.release()}")
    log(f"CPU Cores: {psutil.cpu_count(logical=True)}")
    log(f"RAM Total: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
    else:
        log("GPU: NÃO DETECTADA (Rodando em CPU Lenta)")
    log("="*50)

def main():
    # Limpar log anterior
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    
    get_system_info()
    
    total_start = time.time()
    
    log(f"Iniciando Bateria de Testes ({len(MODULES)} módulos)...")
    
    for mod in MODULES:
        log(f"\n>>> EXECUTANDO MÓDULO: {mod['name']} <<<")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mod['path'])
        
        if not os.path.exists(script_path):
            log(f"ERRO: Script não encontrado em {script_path}")
            continue
            
        start_t = time.time()
        
        # Executar como subprocesso para garantir limpeza de memória a cada run
        # python path_to_script
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                encoding='utf-8' # Forçar UTF-8
            )
            
            end_t = time.time()
            duration = round(end_t - start_t, 2)
            
            if result.returncode == 0:
                log(f"[SUCESSO] {mod['name']} finalizado em {duration} segundos.")
                # Opcional: Salvar output detalhado se quiser
                # with open(f"{mod['name']}_details.log", "w") as f: f.write(result.stdout)
            else:
                log(f"[FALHA] {mod['name']} encerrou com erro (Exit Code {result.returncode}).")
                log(f"Erro Output:\n{result.stderr}")
                
        except Exception as e:
            log(f"[CRÍTICO] Falha ao tentar iniciar subprocesso: {e}")

    total_duration = round(time.time() - total_start, 2)
    log("\n" + "="*50)
    log(f"BATERIA COMPLETA.")
    log(f"Tempo Total: {total_duration} segundos.")
    log(f"Relatório salvo em: {LOG_FILE}")
    log("="*50)

if __name__ == "__main__":
    main()
