"""
import_data.py - Download e Organização do Dataset
===================================================

Dataset: Human Face Emotions (Happy vs Sad)
Fonte: Kaggle - samithsachidanandan/human-face-emotions

O QUE FAZ:
- Baixa o dataset do Kaggle usando kagglehub
- Organiza na estrutura do projeto (data/raw/)
- Verifica se está tudo OK
- Usa config.py para caminhos centralizados

ESTRUTURA CRIADA:
data/
└── raw/
    ├── Happy/  (imagens)
    └── Sad/    (imagens)

USO:
python src/data/import_data.py
"""

import os
import sys
import shutil
from pathlib import Path

# Adicionar raiz do projeto ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS, CLASSES, print_config

# ============================================================================
# FUNÇÕES PRINCIPAIS
# ============================================================================

def download_dataset():
    """
    Baixa o dataset usando kagglehub
    
    RETORNA:
    --------
    str : Caminho onde o dataset foi baixado
    
    RAISES:
    -------
    ImportError : Se kagglehub não estiver instalado
    Exception : Erro durante o download
    """
    
    print("="*80)
    print(" "*25 + "DOWNLOAD DO DATASET")
    print("="*80)
    
    # Verificar/instalar kagglehub
    try:
        import kagglehub
        print("✅ kagglehub instalado")
    except ImportError:
        print("❌ kagglehub não encontrado!")
        print("\n📦 Instalando kagglehub...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kagglehub'])
        import kagglehub
        print("✅ kagglehub instalado com sucesso!")
    
    print("\n📥 Iniciando download do dataset...")
    print("⏳ Isso pode levar alguns minutos dependendo da sua conexão...")
    
    # Download do dataset
    download_path = kagglehub.dataset_download("samithsachidanandan/human-face-emotions")
    
    print(f"\n✅ Download concluído!")
    print(f"📁 Dataset baixado em: {download_path}")
    
    return download_path

def setup_dataset_structure(download_path):
    """
    Organiza o dataset na estrutura do projeto
    
    ESTRUTURA:
    data/
    └── raw/
        ├── Happy/
        └── Sad/
    
    PARÂMETROS:
    -----------
    download_path : str
        Caminho onde o kagglehub baixou o dataset
    
    RETORNA:
    --------
    bool : True se sucesso, False caso contrário
    """
    
    print("\n" + "="*80)
    print(" "*25 + "ORGANIZANDO ESTRUTURA")
    print("="*80)
    
    # Usar caminho do config.py
    project_data_path = PATHS['data_raw']
    
    print(f"\n📁 Destino: {project_data_path}")
    
    # Verificar se já existe
    if project_data_path.exists():
        print(f"\n⚠️  A pasta '{project_data_path}' já existe!")
        response = input("   Sobrescrever? (s/n): ").lower()
        
        if response != 's':
            print("❌ Operação cancelada.")
            return False
        
        print("\n🗑️  Removendo pasta antiga...")
        shutil.rmtree(project_data_path)
    
    # Criar diretório
    project_data_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Diretório criado: {project_data_path}")
    
    # Classes para copiar (usando config.py)
    classes_to_copy = [c.capitalize() for c in CLASSES]  # ['happy', 'sad'] → ['Happy', 'Sad']
    
    print(f"\n📂 Copiando classes: {', '.join(classes_to_copy)}")
    
    # Copiar cada classe
    for class_name in classes_to_copy:
        # Caminho de origem (onde kagglehub baixou)
        source = Path(download_path) / "Data" / class_name
        
        # Caminho de destino (estrutura do projeto)
        destination = project_data_path / class_name
        
        if source.exists():
            print(f"\n📁 Copiando {class_name}...")
            shutil.copytree(source, destination)
            
            # Contar imagens copiadas
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            num_images = sum(len(list(destination.glob(f"*{ext}"))) for ext in extensions)
            
            print(f"   ✅ {num_images:,} imagens copiadas")
        else:
            print(f"   ❌ Pasta {class_name} não encontrada em {source}")
            return False
    
    print("\n" + "="*80)
    print("✅ ESTRUTURA ORGANIZADA COM SUCESSO!")
    print("="*80)
    print(f"\n📁 Dataset pronto em: {project_data_path.resolve()}")
    print("\nEstrutura criada:")
    print("data/")
    print("└── raw/")
    print("    ├── Happy/")
    print("    └── Sad/")
    
    return True

def verify_dataset():
    """
    Verifica se o dataset está pronto para uso
    
    VERIFICAÇÕES:
    - Pasta data/raw existe
    - Pastas Happy e Sad existem
    - Cada classe tem pelo menos 3000 imagens
    
    RETORNA:
    --------
    bool : True se tudo OK, False caso contrário
    """
    
    print("\n" + "="*80)
    print(" "*25 + "VERIFICAÇÃO FINAL")
    print("="*80)
    
    # Verificar pasta raw
    data_raw = PATHS['data_raw']
    
    if not data_raw.exists():
        print(f"❌ Pasta '{data_raw}' não encontrada!")
        return False
    
    print(f"✅ Pasta 'data/raw' encontrada")
    
    # Verificar cada classe
    classes_to_check = [c.capitalize() for c in CLASSES]  # ['Happy', 'Sad']
    all_ok = True
    
    print(f"\n📊 VERIFICANDO CLASSES:")
    print("-"*80)
    
    for class_name in classes_to_check:
        class_path = data_raw / class_name
        
        if not class_path.exists():
            print(f"❌ Pasta '{class_name}' não encontrada!")
            all_ok = False
            continue
        
        # Contar imagens
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        num_images = sum(len(list(class_path.glob(f"*{ext}"))) for ext in extensions)
        
        # Verificar se tem imagens suficientes
        min_required = 3000
        status = "✅" if num_images >= min_required else "⚠️ "
        
        print(f"{status} {class_name:10s}: {num_images:,} imagens", end="")
        
        if num_images >= min_required:
            print(" (OK)")
        else:
            print(f" (mínimo recomendado: {min_required:,})")
            all_ok = False
    
    print("-"*80)
    
    if all_ok:
        print("\n✅ Dataset verificado e pronto para uso!")
        
        # Mostrar totais
        total_happy = sum(len(list((data_raw / 'Happy').glob(f"*{ext}"))) 
                         for ext in ['.jpg', '.jpeg', '.png', '.bmp'])
        total_sad = sum(len(list((data_raw / 'Sad').glob(f"*{ext}"))) 
                       for ext in ['.jpg', '.jpeg', '.png', '.bmp'])
        total = total_happy + total_sad
        
        print(f"\n📊 TOTAIS:")
        print(f"   Happy: {total_happy:,} imagens")
        print(f"   Sad:   {total_sad:,} imagens")
        print(f"   Total: {total:,} imagens")
        
        return True
    else:
        print("\n❌ Há problemas com o dataset. Verifique os erros acima.")
        return False

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal - executa todo o processo
    
    ETAPAS:
    1. Download do dataset do Kaggle
    2. Organização na estrutura do projeto
    3. Verificação de integridade
    
    RETORNA:
    --------
    bool : True se sucesso, False caso contrário
    """
    
    print("\n" + "🎭 "*30)
    print(" "*20 + "SETUP AUTOMÁTICO DO DATASET")
    print(" "*25 + "Happy vs Sad")
    print("🎭 "*30 + "\n")
    
    try:
        # Mostrar configuração atual
        print("📋 CONFIGURAÇÃO DO PROJETO:")
        print("-"*80)
        print(f"   Projeto: {PATHS['root']}")
        print(f"   Destino: {PATHS['data_raw']}")
        print(f"   Classes: {CLASSES}")
        print("-"*80)
        
        # Menu de opções
        print("\n📋 OPÇÕES:")
        print("   1. Download completo (baixar do Kaggle + organizar + verificar)")
        print("   2. Apenas verificar (se já tem os dados)")
        print("   3. Cancelar")
        
        choice = input("\n❓ Escolha uma opção (1/2/3): ").strip()
        
        if choice == '3':
            print("❌ Operação cancelada pelo usuário.")
            return False
        
        elif choice == '2':
            # ================================================================
            # MODO: APENAS VERIFICAÇÃO
            # ================================================================
            print("\n" + "="*80)
            print("MODO: VERIFICAÇÃO DE DADOS EXISTENTES")
            print("="*80)
            
            if not verify_dataset():
                print("\n⚠️  Dataset não está completo ou tem problemas.")
                print("💡 Execute a opção 1 para baixar e organizar novamente.")
                return False
            
            print("\n✅ Dataset verificado com sucesso!")
            return True
        
        elif choice == '1':
            # ================================================================
            # MODO: DOWNLOAD COMPLETO
            # ================================================================
            
            # ETAPA 1: DOWNLOAD
            print("\n" + "="*80)
            print("ETAPA 1/3: DOWNLOAD DO DATASET")
            print("="*80)
            
            download_path = download_dataset()
            
            # ETAPA 2: ORGANIZAR ESTRUTURA
            print("\n" + "="*80)
            print("ETAPA 2/3: ORGANIZAÇÃO DA ESTRUTURA")
            print("="*80)
            
            if not setup_dataset_structure(download_path):
                return False
            
            # ETAPA 3: VERIFICAÇÃO
            print("\n" + "="*80)
            print("ETAPA 3/3: VERIFICAÇÃO DE INTEGRIDADE")
            print("="*80)
            
            if not verify_dataset():
                return False
        
        else:
            print("❌ Opção inválida. Escolha 1, 2 ou 3.")
            return False
        
        # ====================================================================
        # SUCESSO!
        # ====================================================================
        print("\n" + "="*80)
        print(" "*25 + "🎉 TUDO PRONTO!")
        print("="*80)
        
        print("\n✅ Dataset organizado com sucesso!")
        print(f"📁 Localização: {PATHS['data_raw']}")
        
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("   1. Dataset está em: data/raw/")
        print("   2. Próximo: gerar simulações com data_prep.py")
        print("   3. Execute: python src/data/data_prep.py")
        
        print("\n" + "="*80)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operação interrompida pelo usuário.")
        return False
        
    except Exception as e:
        print(f"\n❌ ERRO durante o setup: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Setup concluído com sucesso!")
        exit(0)
    else:
        print("\n❌ Setup falhou. Verifique os erros acima.")
        exit(1)