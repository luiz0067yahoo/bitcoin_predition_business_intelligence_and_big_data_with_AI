
# ☁️ Instruções para Execução no Google Colab

Este laboratório foi desenhado para ser executado em ambiente de nuvem de alta performance (GPU).

## 🚀 Passo a Passo

1. **Upload:**
   - Faça o upload da pasta inteira `laboratorio_ia` para o seu Google Drive.
   - Ou faça upload direto para a sessão do Colab (lembrando que apaga ao desconectar).

2. **Ativar GPU:**
   - No Colab, vá em: `Runtime` > `Change runtime type` > `T4 GPU`.

3. **Instalar Dependências:**
   - Execute a célula:
     ```python
     !pip install deap xgboost torch numpy matplotlib
     ```

4. **Executar um Módulo:**
   - Exemplo para rodar o Módulo de Marte (RNA):
     ```python
     %cd /content/laboratorio_ia/01_rna_marte
     !python run_experimentos_marte.py
     ```

5. **Visualizar Resultados:**
   - Os gráficos serão salvos na pasta `resultados_visuais` dentro de cada módulo.
   - Você pode baixá-los ou visualizá-los diretamente no notebook com:
     ```python
     from IPython.display import Image
     Image('resultados_visuais/expA_olympus_mons.png')
     ```

---
*Nota: Certifique-se de que a estrutura de diretórios foi mantida ao fazer o upload.*
