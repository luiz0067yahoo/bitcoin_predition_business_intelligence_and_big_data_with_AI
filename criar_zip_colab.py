
import zipfile
import os

def criar_pacote_colab_inteligente():
    zip_name = "Laboratorio_Bitcoin_AI_Completo.zip"
    
    # Arquivos/Pastas a incluir
    include_roots = [
        "laboratorio_ia",
        "data_loader.py",
        "requirements.txt",
        "README_COLAB.md",
        "gerar_relatorio_tecnico_final.py",
        "organizar_resultados.py",
        "auditoria_dados.py"
    ]
    
    # Pasta Data (tratamento especial)
    data_dir = "data"
    
    print(f"Compactando arquivos para: {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Adicionar Raízes e Laboratório
        for item in include_roots:
            if os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    for file in files:
                        # Ignorar caches e lixo
                        if "__pycache__" in root or file.endswith(".pyc"):
                            continue
                        
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, file_path)
            elif os.path.exists(item):
                zipf.write(item, item)
                
        # 2. Dados ignorados pois estão no Drive
        print(f"  [INFO] Pasta 'data' ignorada (Arquivos já estão no Drive).")

    print(f"\n[OK] ZIP Criado com Sucesso!")

if __name__ == "__main__":
    criar_pacote_colab_inteligente()
