"""
01_config.py - Configuração e Verificação do Ambiente Gemini
============================================================

ETAPA 1: Configuração inicial para o pipeline Gemini Flash

O QUE FAZ:
- Verifica a instalação das bibliotecas necessárias
- Valida a API key do Gemini (carregada do .env via config.py)
- Verifica a estrutura do dataset (simulações)
- Cria as pastas de resultados necessárias
- Salva configuração do modelo

MODELO:
- Nome: Gemini 2.0 Flash
- Tipo: Multimodal Large Language Model
- Capacidades: Texto + Imagem
- Rate Limit (grátis): 15 req/min

USO:
python src/gemini/01_config.py
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
    GEMINI_API_KEY,
    GEMINI_REQUESTS_PER_MINUTE,
    validate_api_keys,
    validate_paths,
    create_directories,
    get_simulation_path
)

# ============================================================================
# INFORMAÇÕES DO MODELO GEMINI
# ============================================================================

MODEL_INFO = {
    'name': 'Gemini 2.0 Flash',
    'model_id': 'gemini-2.0-flash',
    'type': 'Multimodal Large Language Model',
    'capabilities': ['Text', 'Image', 'Video'],
    'image_size': [224, 224],
    'rate_limit': GEMINI_REQUESTS_PER_MINUTE,
    'target_classes': ['happy', 'sad']
}

# ============================================================================
# PARTE 1: VERIFICAÇÃO DAS BIBLIOTECAS
# ============================================================================

def verify_libraries():
    """
    Verifica se todas as bibliotecas necessárias estão instaladas
    
    BIBLIOTECAS:
    - google-generativeai: API oficial do Google Gemini
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
        'google.generativeai': 'google-generativeai',
        'PIL': 'pillow',
        'pandas': 'pandas',
        'numpy': 'numpy',
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
    
    # Verificar import específico do genai
    try:
        import google.generativeai as genai
        print("\n✅ Import google.generativeai OK")
    except ImportError as e:
        print(f"\n❌ Erro em import específico: {e}")
        all_ok = False
    
    return all_ok

# ============================================================================
# PARTE 2: VERIFICAÇÃO DA API KEY
# ============================================================================

def verify_api_key():
    """
    Verifica se a API key do Gemini está configurada no .env
    
    A API key é carregada automaticamente pelo config.py via python-dotenv
    
    COMO OBTER SUA API KEY:
    1. Acesse: https://aistudio.google.com/app/apikey
    2. Faça login com sua conta Google
    3. Clique em "Create API Key"
    4. Adicione ao arquivo .env: GEMINI_API_KEY=sua_chave
    
    RETORNA:
    --------
    bool : True se configurada, False caso contrário
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICAÇÃO DA API KEY GEMINI")
    print("=" * 80)
    
    if GEMINI_API_KEY:
        print(f"\n✅ GEMINI_API_KEY configurada: {GEMINI_API_KEY[:10]}***")
        return True
    else:
        print("\n❌ GEMINI_API_KEY não encontrada!")
        print("\n📝 SOLUÇÃO:")
        print("   1. Crie um arquivo .env na raiz do projeto")
        print("   2. Adicione: GEMINI_API_KEY=sua_chave_aqui")
        print("   3. Obtenha sua chave em: https://aistudio.google.com/app/apikey")
        return False

# ============================================================================
# PARTE 3: INFORMAÇÕES DO MODELO
# ============================================================================

def show_model_info():
    """
    Exibe informações sobre o modelo Gemini Flash
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "INFORMAÇÕES DO MODELO")
    print("=" * 80)
    
    print(f"""
📊 MODELO GEMINI:
   Nome:         {MODEL_INFO['name']}
   Model ID:     {MODEL_INFO['model_id']}
   Tipo:         {MODEL_INFO['type']}
   Capacidades:  {', '.join(MODEL_INFO['capabilities'])}
   Rate Limit:   {MODEL_INFO['rate_limit']} req/min (grátis)

🎯 NOSSO OBJETIVO:
   • Classificar emoções: happy vs sad
   • Usar prompt em linguagem natural
   • Comparar com YOLOv8 (modelo especializado)

🆚 COMPARAÇÃO YOLOv8 vs GEMINI:
   ┌─────────────────┬──────────────────────┬──────────────────────┐
   │ Característica  │ YOLOv8               │ Gemini Flash         │
   ├─────────────────┼──────────────────────┼──────────────────────┤
   │ Tipo            │ Detecção de Objetos  │ Multimodal (LLM)     │
   │ Especialização  │ Emoções faciais      │ Propósito geral      │
   │ Entrada         │ Apenas imagem        │ Imagem + Texto       │
   │ Saída           │ JSON estruturado     │ Texto natural        │
   │ Confiança       │ Sim (0-100%)         │ Não                  │
   └─────────────────┴──────────────────────┴──────────────────────┘
""")

# ============================================================================
# PARTE 4: VERIFICAÇÃO DO DATASET
# ============================================================================

def verify_dataset():
    """
    Verifica se as simulações estão prontas em data/simulations/
    
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
    - results/gemini/
    - results/gemini/gemini_sims/
    - models/gemini_flash/
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "PREPARANDO PASTAS")
    print("=" * 80)
    
    dirs_to_create = [
        'results_gemini',
        'gemini_sims',
        'models_gemini'
    ]
    
    create_directories(dirs_to_create)
    
    # Verificar
    print(f"\n📁 PASTAS:")
    print(f"   {'✅' if PATHS['results_gemini'].exists() else '❌'} {PATHS['results_gemini']}")
    print(f"   {'✅' if PATHS['gemini_sims'].exists() else '❌'} {PATHS['gemini_sims']}")
    print(f"   {'✅' if PATHS['models_gemini'].exists() else '❌'} {PATHS['models_gemini']}")

# ============================================================================
# PARTE 6: SALVAR CONFIGURAÇÃO
# ============================================================================

def save_gemini_config():
    """
    Salva a configuração do modelo em JSON
    
    O arquivo é salvo em: models/gemini_flash/gemini_config.json
    """
    
    import json
    
    print("\n" + "=" * 80)
    print(" " * 20 + "SALVANDO CONFIGURAÇÃO")
    print("=" * 80)
    
    # Criar pasta se não existir
    create_directories(['models_gemini'])
    
    config = {
        'model_name': MODEL_INFO['name'],
        'model_id': MODEL_INFO['model_id'],
        'target_classes': MODEL_INFO['target_classes'],
        'image_size': MODEL_INFO['image_size'],
        'rate_limit': MODEL_INFO['rate_limit']
    }
    
    config_path = PATHS['gemini_config']
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuração salva em: {config_path}")
    print("\n📋 Conteúdo:")
    print(json.dumps(config, indent=2))

# ============================================================================
# PARTE 7: VERIFICAR CONFIGURAÇÃO EXISTENTE
# ============================================================================

def verify_existing_config():
    """
    Verifica se já existe configuração salva e se está correta
    
    RETORNA:
    --------
    bool : True se config existe e está OK, False caso contrário
    """
    
    import json
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICANDO CONFIGURAÇÃO EXISTENTE")
    print("=" * 80)
    
    config_path = PATHS['gemini_config']
    
    if not config_path.exists():
        print(f"\n❌ Arquivo de configuração não encontrado: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\n✅ Configuração encontrada: {config_path}")
        print("\n📋 Conteúdo:")
        print(json.dumps(config, indent=2))
        
        # Verificar campos obrigatórios
        required_fields = ['model_name', 'model_id', 'target_classes']
        missing = [f for f in required_fields if f not in config]
        
        if missing:
            print(f"\n⚠️  Campos faltando: {missing}")
            return False
        
        print("\n✅ Configuração válida!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro ao verificar: {e}")
        return False

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Executa todas as verificações da Etapa 1
    
    OPÇÕES:
    1. Configuração completa (verificar tudo + salvar config)
    2. Apenas verificar (bibliotecas + API + dataset + config existente)
    3. Cancelar
    """
    
    print("\n" + "🤖 " * 25)
    print(" " * 15 + "ETAPA 1: CONFIGURAÇÃO E VERIFICAÇÃO")
    print(" " * 20 + "Gemini Flash")
    print("🤖 " * 25 + "\n")
    
    try:
        # Menu
        print("📋 OPÇÕES:")
        print("   1. Configuração completa (verificar tudo + salvar config)")
        print("   2. Apenas verificar (bibliotecas + API + dataset + config)")
        print("   3. Cancelar")
        
        choice = input("\n❓ Escolha uma opção (1/2/3): ").strip()
        
        if choice == '3':
            print("\n❌ Operação cancelada.")
            return False
        
        elif choice == '2':
            # ================================================================
            # MODO: APENAS VERIFICAÇÃO
            # ================================================================
            print("\n" + "=" * 80)
            print("MODO: VERIFICAÇÃO")
            print("=" * 80)
            
            results = {}
            
            # 1. Verificar bibliotecas
            print("\n[1/4] Verificando bibliotecas...")
            results['libraries'] = verify_libraries()
            
            # 2. Verificar API key
            print("\n[2/4] Verificando API key...")
            results['api_key'] = verify_api_key()
            
            # 3. Verificar dataset
            print("\n[3/4] Verificando dataset...")
            results['dataset'] = verify_dataset()
            
            # 4. Verificar config existente
            print("\n[4/4] Verificando configuração...")
            results['config'] = verify_existing_config()
            
            # Resumo
            print("\n" + "=" * 80)
            print(" " * 25 + "RESUMO DA VERIFICAÇÃO")
            print("=" * 80)
            
            all_ok = all(results.values())
            
            print(f"\n   Bibliotecas:    {'✅ OK' if results['libraries'] else '❌ PROBLEMA'}")
            print(f"   API Key:        {'✅ OK' if results['api_key'] else '❌ PROBLEMA'}")
            print(f"   Dataset:        {'✅ OK' if results['dataset'] else '❌ PROBLEMA'}")
            print(f"   Configuração:   {'✅ OK' if results['config'] else '❌ PROBLEMA'}")
            
            if all_ok:
                print("\n✅ Tudo verificado e pronto!")
            else:
                print("\n⚠️  Há problemas. Execute a opção 1 para configurar.")
            
            return all_ok
        
        elif choice == '1':
            # ================================================================
            # MODO: CONFIGURAÇÃO COMPLETA
            # ================================================================
            
            all_ok = True
            
            # 1. Verificar bibliotecas
            print("\n[1/5] Verificando bibliotecas...")
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
            
            # 5. Criar pastas e salvar config
            print("\n[5/5] Preparando ambiente...")
            setup_results_directories()
            save_gemini_config()
            
            if not all_ok:
                print("\n" + "=" * 80)
                print("❌ HÁ PROBLEMAS! Verifique os erros acima.")
                print("=" * 80)
                return False
        
        else:
            print("\n❌ Opção inválida.")
            return False
        
        # ====================================================================
        # SUCESSO
        # ====================================================================
        print("\n" + "=" * 80)
        print(" " * 25 + "🎉 ETAPA 1 CONCLUÍDA!")
        print("=" * 80)
        
        print("""
✅ O que foi verificado/configurado:
   1. Bibliotecas instaladas
   2. API key configurada
   3. Modelo identificado
   4. Dataset verificado
   5. Pastas criadas
   6. Configuração salva

🎯 PRÓXIMA ETAPA:
   Etapa 2: Conectar e testar o modelo
   
   Execute: python src/gemini/02_connector.py
""")
        
        print("=" * 80)
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operação interrompida.")
        return False
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)