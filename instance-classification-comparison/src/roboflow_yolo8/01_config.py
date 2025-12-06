"""
01_config.py - Configuração e Verificação do Ambiente
=====================================================

ETAPA 1: Configuração inicial para o pipeline YOLOv8/Roboflow

O QUE FAZ:
- Verifica a instalação das bibliotecas necessárias
- Valida a API key do Roboflow (carregada do .env via config.py)
- Verifica a estrutura do dataset (simulações)
- Cria as pastas de resultados necessárias

MODELO UTILIZADO:
- Workspace: emotions-dectection
- Projeto: human-face-emotions  
- Versão: 28
- Tipo: YOLOv8 Object Detection

USO:
python src/roboflow_yolo8/01_config.py
"""

import sys
from pathlib import Path

# Adicionar raiz do projeto ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PATHS,
    CLASSES,
    NUM_SIMULATIONS,
    IMAGES_PER_CLASS,
    ROBOFLOW_API_KEY,
    validate_api_keys,
    validate_paths,
    create_directories,
    get_simulation_path
)

# ============================================================================
# INFORMAÇÕES DO MODELO ROBOFLOW
# ============================================================================

MODEL_INFO = {
    'workspace': 'emotions-dectection',
    'project': 'human-face-emotions',
    'version': 28,
    'classes': {
        0: 'anger',
        1: 'content',
        2: 'disgust',
        3: 'fear',
        4: 'happy',      # ← Vamos usar esta
        5: 'neutral',
        6: 'sad',        # ← Vamos usar esta
        7: 'surprise'
    },
    'target_classes': ['happy', 'sad']
}

# ============================================================================
# PARTE 1: VERIFICAÇÃO DAS BIBLIOTECAS
# ============================================================================

def verify_libraries():
    """
    Verifica se todas as bibliotecas necessárias estão instaladas
    
    BIBLIOTECAS:
    - roboflow: API para baixar datasets e modelos
    - ultralytics: Biblioteca oficial do YOLOv8
    - opencv-python: Processamento de imagens
    - pillow: Manipulação de imagens
    - pandas: Organização de dados tabulares
    - scikit-learn: Cálculo de métricas
    - matplotlib/seaborn: Visualizações
    
    RETORNA:
    --------
    bool : True se todas instaladas, False caso contrário
    """
    
    print("=" * 80)
    print(" " * 25 + "VERIFICANDO BIBLIOTECAS")
    print("=" * 80)
    
    libraries = {
        'roboflow': 'roboflow',
        'ultralytics': 'ultralytics',
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    all_ok = True
    
    for module, package in libraries.items():
        try:
            __import__(module)
            print(f"✅ {package} instalado")
        except ImportError:
            print(f"❌ {package} NÃO instalado - Execute: pip install {package}")
            all_ok = False
    
    # Verificar imports específicos
    try:
        from roboflow import Roboflow
        from ultralytics import YOLO
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        print("\n✅ Imports específicos verificados")
    except ImportError as e:
        print(f"\n❌ Erro em import específico: {e}")
        all_ok = False
    
    return all_ok

# ============================================================================
# PARTE 2: VERIFICAÇÃO DA API KEY
# ============================================================================

def verify_api_key():
    """
    Verifica se a API key do Roboflow está configurada no .env
    
    A API key é carregada automaticamente pelo config.py via python-dotenv
    
    COMO OBTER SUA API KEY:
    1. Acesse: https://app.roboflow.com/
    2. Faça login ou crie uma conta (gratuito)
    3. Vá em: Settings → API → Private API Key
    4. Adicione ao arquivo .env: ROBOFLOW_API_KEY=sua_chave
    
    RETORNA:
    --------
    bool : True se configurada, False caso contrário
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICAÇÃO DA API KEY ROBOFLOW")
    print("=" * 80)
    
    if ROBOFLOW_API_KEY:
        print(f"\n✅ ROBOFLOW_API_KEY configurada: {ROBOFLOW_API_KEY[:10]}***")
        return True
    else:
        print("\n❌ ROBOFLOW_API_KEY não encontrada!")
        print("\n📝 SOLUÇÃO:")
        print("   1. Crie um arquivo .env na raiz do projeto")
        print("   2. Adicione: ROBOFLOW_API_KEY=sua_chave_aqui")
        print("   3. Obtenha sua chave em: https://app.roboflow.com/ → Settings → API")
        return False

# ============================================================================
# PARTE 3: INFORMAÇÕES DO MODELO
# ============================================================================

def show_model_info():
    """
    Exibe informações sobre o modelo YOLOv8 do Roboflow
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "INFORMAÇÕES DO MODELO")
    print("=" * 80)
    
    print(f"""
📊 MODELO ROBOFLOW:
   Workspace: {MODEL_INFO['workspace']}
   Projeto:   {MODEL_INFO['project']}
   Versão:    {MODEL_INFO['version']}
   Tipo:      YOLOv8 Object Detection

🏷️  CLASSES DO MODELO (8 emoções):
   0: anger (raiva)
   1: content (contente)
   2: disgust (nojo)
   3: fear (medo)
   4: happy (feliz)     ← NOSSA CLASSE 1
   5: neutral (neutro)
   6: sad (triste)      ← NOSSA CLASSE 2
   7: surprise (surpresa)

🎯 NOSSO OBJETIVO:
   • Usar apenas as classes "happy" e "sad"
   • Ignorar as outras emoções
   • Calcular métricas para classificação binária
""")

# ============================================================================
# PARTE 4: VERIFICAÇÃO DO DATASET
# ============================================================================

def verify_dataset():
    """
    Verifica se as simulações estão prontas em data/simulations/
    
    ESTRUTURA ESPERADA:
    data/simulations/
    ├── SIM01/
    │   ├── happy/  (100 imagens)
    │   └── sad/    (100 imagens)
    ├── SIM02/
    └── ... até SIM30/
    
    RETORNA:
    --------
    bool : True se dataset OK, False caso contrário
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICAÇÃO DO DATASET")
    print("=" * 80)
    
    simulations_path = PATHS['simulations']
    
    # Verificar se pasta existe
    if not simulations_path.exists():
        print(f"\n❌ ERRO: Pasta '{simulations_path}' não encontrada!")
        print("   Execute antes: python src/data/data_prep.py")
        return False
    
    print(f"\n✅ Pasta de simulações encontrada: {simulations_path}")
    
    # Verificar primeira simulação como exemplo
    sim01 = get_simulation_path(1)
    
    if sim01.exists():
        print(f"\n📁 Estrutura de SIM01:")
        
        for class_name in CLASSES:
            class_path = sim01 / class_name
            if class_path.exists():
                extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                num_images = sum(len(list(class_path.glob(f"*{ext}"))) for ext in extensions)
                status = "✅" if num_images == IMAGES_PER_CLASS else "⚠️"
                print(f"   {status} {class_name}: {num_images} imagens")
            else:
                print(f"   ❌ {class_name}: pasta não encontrada")
    
    # Contar simulações
    sims_found = []
    for i in range(1, NUM_SIMULATIONS + 1):
        sim_path = get_simulation_path(i)
        if sim_path.exists():
            sims_found.append(i)
    
    print(f"\n📊 RESUMO:")
    print(f"   Simulações encontradas: {len(sims_found)}/{NUM_SIMULATIONS}")
    
    if len(sims_found) == NUM_SIMULATIONS:
        print("   ✅ Todas as simulações estão prontas!")
        return True
    else:
        missing = set(range(1, NUM_SIMULATIONS + 1)) - set(sims_found)
        print(f"   ❌ Simulações faltando: {sorted(missing)}")
        return False

# ============================================================================
# PARTE 5: CRIAR PASTAS DE RESULTADOS
# ============================================================================

def setup_results_directories():
    """
    Cria as pastas necessárias para salvar os resultados
    
    PASTAS CRIADAS:
    - results/roboflow_yolo8/
    - results/roboflow_yolo8/roboflow_sims/
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "PREPARANDO PASTAS DE RESULTADOS")
    print("=" * 80)
    
    dirs_to_create = [
        'results_roboflow',
        'roboflow_sims'
    ]
    
    create_directories(dirs_to_create)
    
    # Verificar
    results_path = PATHS['results_roboflow']
    sims_path = PATHS['roboflow_sims']
    
    print(f"\n📁 PASTAS DE RESULTADOS:")
    print(f"   {'✅' if results_path.exists() else '❌'} {results_path}")
    print(f"   {'✅' if sims_path.exists() else '❌'} {sims_path}")

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Executa todas as verificações da Etapa 1
    
    ETAPAS:
    1. Verificar bibliotecas instaladas
    2. Verificar API key do Roboflow
    3. Mostrar informações do modelo
    4. Verificar dataset (simulações)
    5. Criar pastas de resultados
    
    RETORNA:
    --------
    bool : True se tudo OK, False caso contrário
    """
    
    print("\n" + "🤖 " * 25)
    print(" " * 15 + "ETAPA 1: CONFIGURAÇÃO E VERIFICAÇÃO")
    print(" " * 20 + "YOLOv8 + Roboflow")
    print("🤖 " * 25 + "\n")
    
    all_ok = True
    
    # 1. Verificar bibliotecas
    print("[1/5] Verificando bibliotecas...")
    if not verify_libraries():
        all_ok = False
    
    # 2. Verificar API key
    print("\n[2/5] Verificando API key...")
    if not verify_api_key():
        all_ok = False
    
    # 3. Mostrar info do modelo
    print("\n[3/5] Informações do modelo...")
    show_model_info()
    
    # 4. Verificar dataset
    print("\n[4/5] Verificando dataset...")
    if not verify_dataset():
        all_ok = False
    
    # 5. Criar pastas de resultados
    print("\n[5/5] Preparando pastas de resultados...")
    setup_results_directories()
    
    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(" " * 25 + "RESUMO DA ETAPA 1")
    print("=" * 80)
    
    if all_ok:
        print("""
✅ TUDO PRONTO!

📝 O que foi verificado:
   1. ✅ Bibliotecas instaladas
   2. ✅ API key configurada
   3. ✅ Modelo identificado
   4. ✅ Dataset verificado
   5. ✅ Pastas criadas

🎯 PRÓXIMA ETAPA:
   Etapa 2: Carregar o modelo do Roboflow
   
   Execute: python src/roboflow_yolo8/02_connector.py
""")
    else:
        print("""
❌ HÁ PROBLEMAS!

   Verifique os erros acima e corrija antes de continuar.
   
   Problemas comuns:
   • Bibliotecas faltando → pip install <pacote>
   • API key não configurada → criar arquivo .env
   • Dataset não preparado → python src/data/data_prep.py
""")
    
    print("=" * 80)
    
    return all_ok

# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)