"""
config.py - Configuração Centralizada do Projeto
================================================

Este arquivo contém:
- Caminhos relativos do projeto
- API Keys (carregadas do .env)
- Constantes globais
- Funções utilitárias

ESTRUTURA DO PROJETO:
--------------------
.
├── config.py              ← ESTE ARQUIVO
├── data/
│   ├── raw/               ← Dados originais (Happy, Sad)
│   └── simulations/       ← 30 simulações preparadas
├── models/
│   ├── gemini_flash/
│   └── roboflow_yolo8/
├── results/
│   ├── comparison/        ← Comparação entre modelos
│   ├── gemini/           ← Resultados Gemini
│   └── roboflow_yolo8/   ← Resultados YOLOv8
└── src/
    ├── data/             ← Scripts de preparação
    ├── evaluation/       ← Scripts de comparação
    ├── gemini/          ← Scripts Gemini
    └── roboflow_yolo8/  ← Scripts YOLOv8

USO:
----
from config import PATHS, ROBOFLOW_API_KEY, GEMINI_API_KEY, NUM_SIMULATIONS

# Acessar caminhos
simulations = PATHS['simulations']
results_gemini = PATHS['results_gemini']

# Usar API Keys
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# CARREGAR VARIÁVEIS DE AMBIENTE (.env)
# ============================================================================

# Carregar .env da raiz do projeto
load_dotenv()

# API Keys
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ============================================================================
# ESTRUTURA DE CAMINHOS DO PROJETO
# ============================================================================

def setup_project_paths():
    """
    Configura todos os caminhos do projeto de forma robusta
    
    ESTRUTURA:
    ----------
    .
    ├── config.py
    ├── data/
    │   ├── raw/
    │   │   ├── Happy/
    │   │   └── Sad/
    │   └── simulations/
    │       ├── SIM01/
    │       ├── SIM02/
    │       └── ... SIM30/
    ├── models/
    │   ├── gemini_flash/
    │   └── roboflow_yolo8/
    ├── results/
    │   ├── comparison/
    │   ├── gemini/
    │   └── roboflow_yolo8/
    └── src/
        ├── data/
        ├── evaluation/
        ├── gemini/
        └── roboflow_yolo8/
    
    RETORNA:
    --------
    dict : Todos os caminhos do projeto
    """
    
    # Diretório raiz do projeto (onde está config.py)
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    # Definir todos os caminhos
    paths = {
        # ====================================================================
        # RAIZ
        # ====================================================================
        'root': PROJECT_ROOT,
        
        # ====================================================================
        # DATA
        # ====================================================================
        'data': PROJECT_ROOT / 'data',
        'data_raw': PROJECT_ROOT / 'data' / 'raw',
        'data_raw_happy': PROJECT_ROOT / 'data' / 'raw' / 'Happy',
        'data_raw_sad': PROJECT_ROOT / 'data' / 'raw' / 'Sad',
        'simulations': PROJECT_ROOT / 'data' / 'simulations',
        
        # ====================================================================
        # MODELS
        # ====================================================================
        'models': PROJECT_ROOT / 'models',
        'models_gemini': PROJECT_ROOT / 'models' / 'gemini_flash',
        'models_roboflow': PROJECT_ROOT / 'models' / 'roboflow_yolo8',
        'gemini_config': PROJECT_ROOT / 'models' / 'gemini_flash' / 'gemini_config.json',
        'roboflow_config': PROJECT_ROOT / 'models' / 'roboflow_yolo8' / 'roboflow_config.json',
        
        # ====================================================================
        # RESULTS
        # ====================================================================
        'results': PROJECT_ROOT / 'results',
        
        # Results - Comparison
        'results_comparison': PROJECT_ROOT / 'results' / 'comparison',
        'comparison_plots': PROJECT_ROOT / 'results' / 'comparison' / 'plots',
        'comparison_report': PROJECT_ROOT / 'results' / 'comparison' / 'comparison_report.txt',
        'wilcoxon_results': PROJECT_ROOT / 'results' / 'comparison' / 'wilcoxon_test_results.json',
        
        # Results - Gemini
        'results_gemini': PROJECT_ROOT / 'results' / 'gemini',
        'gemini_metrics': PROJECT_ROOT / 'results' / 'gemini' / 'all_metrics.csv',
        'gemini_stats': PROJECT_ROOT / 'results' / 'gemini' / 'summary_statistics.json',
        'gemini_sims': PROJECT_ROOT / 'results' / 'gemini' / 'gemini_sims',
        'gemini_confusion': PROJECT_ROOT / 'results' / 'gemini' / 'confusion_matrix_sim01.png',
        
        # Results - Roboflow YOLOv8
        'results_roboflow': PROJECT_ROOT / 'results' / 'roboflow_yolo8',
        'roboflow_metrics': PROJECT_ROOT / 'results' / 'roboflow_yolo8' / 'all_metrics.csv',
        'roboflow_stats': PROJECT_ROOT / 'results' / 'roboflow_yolo8' / 'summary_statistics.json',
        'roboflow_sims': PROJECT_ROOT / 'results' / 'roboflow_yolo8' / 'roboflow_sims',
        'roboflow_confusion': PROJECT_ROOT / 'results' / 'roboflow_yolo8' / 'confusion_matrix_sim01.png',
        
        # ====================================================================
        # SRC
        # ====================================================================
        'src': PROJECT_ROOT / 'src',
        'src_data': PROJECT_ROOT / 'src' / 'data',
        'src_evaluation': PROJECT_ROOT / 'src' / 'evaluation',
        'src_gemini': PROJECT_ROOT / 'src' / 'gemini',
        'src_roboflow': PROJECT_ROOT / 'src' / 'roboflow_yolo8',
        
        # ====================================================================
        # NOTEBOOKS
        # ====================================================================
        'notebooks': PROJECT_ROOT / 'notebooks',
    }
    
    return paths

# Configurar caminhos globais
PATHS = setup_project_paths()

# ============================================================================
# CONSTANTES DO PROJETO
# ============================================================================

# Número de simulações
NUM_SIMULATIONS = 30

# Imagens por classe por simulação
IMAGES_PER_CLASS = 100

# Configurações de imagem
IMG_SIZE = ("variado")
BATCH_SIZE = 32

# Rate limiting (Gemini)
GEMINI_REQUESTS_PER_MINUTE = 15
GEMINI_SECONDS_PER_REQUEST = 60 / GEMINI_REQUESTS_PER_MINUTE

# Métricas para análise
METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
METRIC_NAMES = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']

# Classes
CLASSES = ['happy', 'sad']
CLASS_MAPPING = {'sad': 0, 'happy': 1}

# Teste estatístico
WILCOXON_ALPHA = 0.05  # 95% de confiança

# ============================================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================================

def validate_api_keys():
    """
    Valida se as API Keys estão configuradas
    
    RETORNA:
    --------
    bool : True se tudo OK, False caso contrário
    """
    
    issues = []
    
    if not ROBOFLOW_API_KEY:
        issues.append("ROBOFLOW_API_KEY")
    
    if not GEMINI_API_KEY:
        issues.append("GEMINI_API_KEY")
    
    if issues:
        print("="*80)
        print("❌ API KEYS FALTANDO")
        print("="*80)
        print(f"\nAs seguintes chaves não foram encontradas no .env:")
        for key in issues:
            print(f"   • {key}")
        
        print("\n📝 SOLUÇÃO:")
        print("1. Crie um arquivo .env na raiz do projeto")
        print("2. Adicione as chaves:")
        print("-"*80)
        print("ROBOFLOW_API_KEY=sua_chave_roboflow")
        print("GEMINI_API_KEY=sua_chave_gemini")
        print("-"*80)
        
        return False
    
    return True

def validate_paths(required_paths):
    """
    Valida se os caminhos necessários existem
    
    PARÂMETROS:
    -----------
    required_paths : list
        Lista de nomes de caminhos para validar
        Exemplo: ['simulations', 'results']
    
    RETORNA:
    --------
    bool : True se todos existem, False caso contrário
    
    EXEMPLO:
    --------
    >>> from config import validate_paths
    >>> if not validate_paths(['simulations', 'data_raw']):
    ...     exit(1)
    """
    
    print("="*80)
    print(" "*25 + "VALIDAÇÃO DE CAMINHOS")
    print("="*80)
    
    all_ok = True
    
    for path_name in required_paths:
        if path_name in PATHS:
            path = PATHS[path_name]
            
            if path.exists():
                print(f"   ✅ {path_name:25s}: {path}")
            else:
                print(f"   ❌ {path_name:25s}: {path} (NÃO EXISTE)")
                all_ok = False
        else:
            print(f"   ⚠️  {path_name:25s}: Não configurado em PATHS")
            all_ok = False
    
    print("="*80)
    
    if not all_ok:
        print("\n💡 Alguns caminhos não existem. Certifique-se de:")
        print("   1. Ter executado os scripts de preparação de dados")
        print("   2. Estar executando do diretório correto")
    
    return all_ok

def create_directories(dir_list):
    """
    Cria diretórios se não existirem
    
    PARÂMETROS:
    -----------
    dir_list : list
        Lista de nomes de diretórios para criar
        Exemplo: ['results_gemini', 'comparison_plots']
    
    EXEMPLO:
    --------
    >>> from config import create_directories
    >>> create_directories(['results_gemini', 'results_roboflow', 'comparison_plots'])
    """
    
    created = []
    
    for dir_name in dir_list:
        if dir_name in PATHS:
            path = PATHS[dir_name]
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(dir_name)
    
    if created:
        print(f"\n📁 Diretórios criados: {', '.join(created)}")

def get_simulation_path(sim_number):
    """
    Retorna o caminho de uma simulação específica
    
    PARÂMETROS:
    -----------
    sim_number : int
        Número da simulação (1-30)
    
    RETORNA:
    --------
    Path : Caminho da simulação
    
    EXEMPLO:
    --------
    >>> from config import get_simulation_path
    >>> sim01 = get_simulation_path(1)
    >>> print(sim01)
    /path/to/project/data/simulations/SIM01
    >>> print(sim01.exists())
    True
    """
    
    return PATHS['simulations'] / f"SIM{sim_number:02d}"

def get_simulation_metrics_path(sim_number, model='gemini'):
    """
    Retorna o caminho do arquivo de métricas de uma simulação
    
    PARÂMETROS:
    -----------
    sim_number : int
        Número da simulação (1-30)
    model : str
        Nome do modelo ('gemini' ou 'roboflow')
    
    RETORNA:
    --------
    Path : Caminho do arquivo de métricas
    
    EXEMPLO:
    --------
    >>> from config import get_simulation_metrics_path
    >>> metrics_file = get_simulation_metrics_path(1, 'gemini')
    >>> print(metrics_file)
    /path/to/project/results/gemini/gemini_sims/sim01_metrics.json
    """
    
    if model == 'gemini':
        return PATHS['gemini_sims'] / f"sim{sim_number:02d}_metrics.json"
    elif model == 'roboflow':
        return PATHS['roboflow_sims'] / f"sim{sim_number:02d}_metrics.json"
    else:
        raise ValueError(f"Modelo '{model}' inválido. Use 'gemini' ou 'roboflow'.")

def print_config():
    """
    Imprime a configuração atual do projeto
    (útil para debug e verificação)
    
    EXEMPLO:
    --------
    >>> from config import print_config
    >>> print_config()
    """
    
    print("="*80)
    print(" "*25 + "CONFIGURAÇÃO DO PROJETO")
    print("="*80)
    
    print("\n📁 CAMINHOS PRINCIPAIS:")
    print("-"*80)
    main_paths = ['root', 'data', 'simulations', 'models', 'results', 'src']
    for name in main_paths:
        if name in PATHS:
            path = PATHS[name]
            exists = "✅" if path.exists() else "❌"
            print(f"   {exists} {name:20s}: {path}")
    
    print("\n📁 DADOS:")
    print("-"*80)
    data_paths = ['data_raw', 'data_raw_happy', 'data_raw_sad', 'simulations']
    for name in data_paths:
        if name in PATHS:
            path = PATHS[name]
            exists = "✅" if path.exists() else "❌"
            
            # Contar itens se existir
            count_str = ""
            if path.exists():
                if path.is_dir():
                    items = list(path.iterdir())
                    if name == 'simulations':
                        sims = [d for d in items if d.is_dir() and d.name.startswith('SIM')]
                        count_str = f"({len(sims)} simulações)"
                    else:
                        count_str = f"({len(items)} itens)"
            
            print(f"   {exists} {name:20s}: {path} {count_str}")
    
    print("\n📁 RESULTADOS:")
    print("-"*80)
    result_paths = ['results_gemini', 'results_roboflow', 'results_comparison']
    for name in result_paths:
        if name in PATHS:
            path = PATHS[name]
            exists = "✅" if path.exists() else "❌"
            print(f"   {exists} {name:20s}: {path}")
    
    print("\n🔑 API KEYS:")
    print("-"*80)
    if ROBOFLOW_API_KEY:
        print(f"   ✅ ROBOFLOW_API_KEY: {ROBOFLOW_API_KEY[:10]}***")
    else:
        print(f"   ❌ ROBOFLOW_API_KEY: NÃO CONFIGURADA")
    
    if GEMINI_API_KEY:
        print(f"   ✅ GEMINI_API_KEY: {GEMINI_API_KEY[:10]}***")
    else:
        print(f"   ❌ GEMINI_API_KEY: NÃO CONFIGURADA")
    
    print("\n⚙️  CONSTANTES:")
    print("-"*80)
    print(f"   NUM_SIMULATIONS: {NUM_SIMULATIONS}")
    print(f"   IMAGES_PER_CLASS: {IMAGES_PER_CLASS}")
    print(f"   IMG_SIZE: {IMG_SIZE}")
    print(f"   BATCH_SIZE: {BATCH_SIZE}")
    print(f"   CLASSES: {CLASSES}")
    print(f"   METRICS: {METRICS}")
    print(f"   WILCOXON_ALPHA: {WILCOXON_ALPHA}")
    
    print("\n" + "="*80)

def get_project_summary():
    """
    Retorna um resumo do projeto em formato de dicionário
    
    RETORNA:
    --------
    dict : Resumo do projeto
    
    EXEMPLO:
    --------
    >>> from config import get_project_summary
    >>> summary = get_project_summary()
    >>> print(f"Simulações: {summary['num_simulations_found']}/{summary['num_simulations_expected']}")
    """
    
    # Contar simulações
    sims_path = PATHS['simulations']
    num_sims_found = 0
    if sims_path.exists():
        sims = [d for d in sims_path.iterdir() if d.is_dir() and d.name.startswith('SIM')]
        num_sims_found = len(sims)
    
    # Verificar resultados
    gemini_metrics_exists = PATHS['gemini_metrics'].exists() if 'gemini_metrics' in PATHS else False
    roboflow_metrics_exists = PATHS['roboflow_metrics'].exists() if 'roboflow_metrics' in PATHS else False
    
    return {
        'project_root': str(PATHS['root']),
        'num_simulations_expected': NUM_SIMULATIONS,
        'num_simulations_found': num_sims_found,
        'simulations_ready': num_sims_found == NUM_SIMULATIONS,
        'data_raw_exists': PATHS['data_raw'].exists(),
        'gemini_results_exists': gemini_metrics_exists,
        'roboflow_results_exists': roboflow_metrics_exists,
        'api_keys_configured': bool(ROBOFLOW_API_KEY and GEMINI_API_KEY),
    }

# ============================================================================
# VALIDAÇÃO INICIAL (executa quando importa o módulo)
# ============================================================================

# Validar API Keys ao importar (opcional - comente se não quiser)
# _api_keys_ok = validate_api_keys()

# ============================================================================
# EXEMPLO DE USO (quando executado diretamente)
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 TESTANDO CONFIG.PY")
    print_config()
    
    print("\n📊 RESUMO DO PROJETO:")
    summary = get_project_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n🔍 TESTANDO FUNÇÕES:")
    
    # Teste 1: get_simulation_path
    print("\n1. get_simulation_path(1):")
    sim01 = get_simulation_path(1)
    print(f"   Caminho: {sim01}")
    print(f"   Existe: {sim01.exists()}")
    
    # Teste 2: validate_paths
    print("\n2. validate_paths(['simulations', 'data_raw']):")
    validate_paths(['simulations', 'data_raw'])
    
    # Teste 3: create_directories
    print("\n3. create_directories(['results_gemini', 'results_roboflow']):")
    create_directories(['results_gemini', 'results_roboflow', 'comparison_plots', 
                       'gemini_sims', 'roboflow_sims'])
    
    print("\n✅ Teste concluído!")
    print("="*80)