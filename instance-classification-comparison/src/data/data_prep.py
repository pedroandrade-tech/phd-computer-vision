"""
data_prep.py - Preparação das Simulações
========================================

Dataset: Human Face Emotions (Happy vs Sad)
Estrutura: Cada simulação tem 100 imagens Happy + 100 imagens Sad
Total: 30 simulações × 200 imagens = 6000 imagens (3000 Happy + 3000 Sad)

O QUE FAZ:
- Cria 30 simulações a partir dos dados em data/raw/
- Cada simulação tem 100 imagens únicas de cada classe
- Nenhuma imagem se repete entre simulações
- Usa config.py para caminhos e constantes centralizados

ESTRUTURA CRIADA:
data/
└── simulations/
    ├── SIM01/
    │   ├── happy/  (100 imagens)
    │   └── sad/    (100 imagens)
    ├── SIM02/
    │   └── ...
    └── SIM30/

USO:
python src/data/data_prep.py
"""

import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path

# Adicionar raiz do projeto ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PATHS, 
    CLASSES, 
    NUM_SIMULATIONS, 
    IMAGES_PER_CLASS,
    print_config
)

# ============================================================================
# MAPEAMENTO DE CLASSES
# ============================================================================

# Mapeia nome da pasta original (capitalizado) para nome da pasta de destino (minúsculo)
CLASS_MAPPING = {
    "Happy": "happy",
    "Sad": "sad"
}

# ============================================================================
# FUNÇÕES DE VERIFICAÇÃO
# ============================================================================

def verify_source_data():
    """
    Verifica se os dados originais estão disponíveis em data/raw/
    
    VERIFICAÇÕES:
    - Pasta data/raw existe
    - Pastas Happy e Sad existem
    - Cada classe tem imagens suficientes para todas as simulações
    
    RETORNA:
    --------
    bool : True se tudo OK, False caso contrário
    """
    
    print("=" * 80)
    print(" " * 25 + "VERIFICAÇÃO DOS DADOS ORIGINAIS")
    print("=" * 80)
    
    data_raw = PATHS['data_raw']
    
    # Verificar se pasta raw existe
    if not data_raw.exists():
        print(f"\n❌ ERRO: Pasta '{data_raw}' não encontrada!")
        print("   Execute primeiro: python src/data/import_data.py")
        return False
    
    print(f"\n✅ Pasta 'data/raw' encontrada: {data_raw}")
    
    all_ok = True
    total_images = {}
    required_total = NUM_SIMULATIONS * IMAGES_PER_CLASS
    
    print(f"\n📊 VERIFICANDO CLASSES:")
    print("-" * 80)
    
    for original_class, folder_name in CLASS_MAPPING.items():
        class_path = data_raw / original_class
        
        if not class_path.exists():
            print(f"❌ Pasta '{original_class}' não encontrada em '{data_raw}'")
            all_ok = False
            continue
        
        # Contar imagens
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        num_images = sum(len(list(class_path.glob(f"*{ext}"))) for ext in extensions)
        total_images[original_class] = num_images
        
        # Verificar se tem imagens suficientes
        status = "✅" if num_images >= required_total else "❌"
        print(f"{status} {original_class}: {num_images:,} imagens disponíveis")
        print(f"   Necessário: {required_total:,} ({NUM_SIMULATIONS} sims × {IMAGES_PER_CLASS} imgs)")
        
        if num_images < required_total:
            print(f"   ❌ ERRO: Imagens insuficientes!")
            all_ok = False
    
    print("-" * 80)
    
    if all_ok:
        print(f"\n✅ Requisitos atendidos!")
        print(f"   Total necessário: {required_total * len(CLASS_MAPPING):,} imagens ({required_total:,} por classe)")
        print(f"   Total disponível: {sum(total_images.values()):,} imagens")
    
    return all_ok


def verify_simulations():
    """
    Verifica se as simulações foram criadas corretamente
    
    VERIFICAÇÕES:
    - Pasta data/simulations existe
    - Todas as 30 simulações existem (SIM01 a SIM30)
    - Cada simulação tem as pastas happy e sad
    - Cada classe tem exatamente IMAGES_PER_CLASS imagens
    
    RETORNA:
    --------
    bool : True se tudo OK, False caso contrário
    """
    
    print("\n" + "=" * 80)
    print(" " * 25 + "VERIFICAÇÃO DAS SIMULAÇÕES")
    print("=" * 80)
    
    simulations_path = PATHS['simulations']
    
    # Verificar se pasta simulations existe
    if not simulations_path.exists():
        print(f"\n❌ Pasta '{simulations_path}' não encontrada!")
        print("   Execute a opção 1 para criar as simulações.")
        return False
    
    print(f"\n✅ Pasta 'data/simulations' encontrada: {simulations_path}")
    
    all_ok = True
    total_images_per_class = {folder_name: 0 for folder_name in CLASS_MAPPING.values()}
    simulations_ok = 0
    simulations_with_issues = []
    
    print(f"\n📊 VERIFICANDO {NUM_SIMULATIONS} SIMULAÇÕES:")
    print("-" * 80)
    
    for sim_num in range(1, NUM_SIMULATIONS + 1):
        sim_folder = simulations_path / f"SIM{sim_num:02d}"
        sim_ok = True
        
        if not sim_folder.exists():
            print(f"❌ SIM{sim_num:02d} não encontrada!")
            all_ok = False
            simulations_with_issues.append(sim_num)
            continue
        
        # Verificar cada classe
        for folder_name in CLASS_MAPPING.values():
            class_path = sim_folder / folder_name
            
            if not class_path.exists():
                print(f"❌ SIM{sim_num:02d}/{folder_name} não encontrada!")
                all_ok = False
                sim_ok = False
                continue
            
            # Contar imagens
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            num_images = sum(len(list(class_path.glob(f"*{ext}"))) for ext in extensions)
            total_images_per_class[folder_name] += num_images
            
            if num_images != IMAGES_PER_CLASS:
                print(f"❌ SIM{sim_num:02d}/{folder_name}: {num_images} imagens (esperado: {IMAGES_PER_CLASS})")
                all_ok = False
                sim_ok = False
        
        if sim_ok:
            simulations_ok += 1
        else:
            simulations_with_issues.append(sim_num)
    
    print("-" * 80)
    
    # Resumo
    print(f"\n📊 RESUMO:")
    print(f"   Simulações OK: {simulations_ok}/{NUM_SIMULATIONS}")
    
    if simulations_with_issues:
        print(f"   Simulações com problemas: {simulations_with_issues}")
    
    print(f"\n📊 TOTAIS POR CLASSE:")
    for folder_name, total in total_images_per_class.items():
        expected = NUM_SIMULATIONS * IMAGES_PER_CLASS
        status = "✅" if total == expected else "❌"
        print(f"   {status} {folder_name}: {total:,} imagens (esperado: {expected:,})")
    
    total_geral = sum(total_images_per_class.values())
    expected_total = NUM_SIMULATIONS * IMAGES_PER_CLASS * len(CLASS_MAPPING)
    
    print(f"\n   Total geral: {total_geral:,} imagens (esperado: {expected_total:,})")
    
    if all_ok:
        print(f"\n✅ Todas as {NUM_SIMULATIONS} simulações estão corretas!")
        print(f"✅ Cada simulação tem {IMAGES_PER_CLASS} imagens por classe")
        print(f"✅ Total: {total_geral:,} imagens únicas")
    else:
        print(f"\n❌ Há problemas com as simulações. Verifique os erros acima.")
        print("   Execute a opção 1 para recriar as simulações.")
    
    return all_ok

# ============================================================================
# FUNÇÕES DE CRIAÇÃO
# ============================================================================

def collect_all_images():
    """
    Coleta todas as imagens de cada classe em data/raw/
    
    RETORNA:
    --------
    dict : Dicionário com listas de paths das imagens por classe
           {'Happy': [Path, Path, ...], 'Sad': [Path, Path, ...]}
    """
    
    print("\n" + "=" * 80)
    print(" " * 25 + "COLETANDO IMAGENS")
    print("=" * 80)
    
    data_raw = PATHS['data_raw']
    all_images = {}
    
    for original_class in CLASS_MAPPING.keys():
        class_path = data_raw / original_class
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        for ext in extensions:
            images.extend(list(class_path.glob(f"*{ext}")))
        
        all_images[original_class] = images
        print(f"✅ {original_class}: {len(images):,} imagens coletadas")
    
    return all_images


def create_single_simulation(sim_num, all_images, used_indices, base_seed=42):
    """
    Cria uma única simulação com IMAGES_PER_CLASS imagens de cada classe
    
    IMPORTANTE: Garante que nenhuma imagem seja repetida entre simulações
    
    PARÂMETROS:
    -----------
    sim_num : int
        Número da simulação (1-30)
    all_images : dict
        Dicionário com todas as imagens disponíveis por classe
    used_indices : dict
        Dicionário com índices das imagens já usadas em outras simulações
    base_seed : int
        Seed base para reprodutibilidade
    
    RETORNA:
    --------
    dict : Estatísticas da simulação criada, ou None se falhou
    """
    
    # Configurar seed único para esta simulação (reprodutibilidade)
    seed = base_seed + sim_num
    random.seed(seed)
    np.random.seed(seed)
    
    # Criar pasta da simulação
    sim_folder = PATHS['simulations'] / f"SIM{sim_num:02d}"
    sim_folder.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'simulation': sim_num,
        'seed': seed,
        'classes': {}
    }
    
    # Processar cada classe
    for original_class, folder_name in CLASS_MAPPING.items():
        # Pegar imagens disponíveis (que ainda não foram usadas)
        available_indices = [
            i for i in range(len(all_images[original_class]))
            if i not in used_indices[original_class]
        ]
        
        if len(available_indices) < IMAGES_PER_CLASS:
            print(f"⚠️  AVISO: Não há imagens suficientes não-usadas para {original_class} na SIM{sim_num:02d}")
            print(f"   Disponíveis: {len(available_indices)}, Necessárias: {IMAGES_PER_CLASS}")
            return None
        
        # Selecionar aleatoriamente IMAGES_PER_CLASS índices
        selected_indices = random.sample(available_indices, IMAGES_PER_CLASS)
        
        # Marcar como usadas
        used_indices[original_class].update(selected_indices)
        
        # Criar diretório da classe
        class_dir = sim_folder / folder_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copiar imagens
        for idx in selected_indices:
            img_path = all_images[original_class][idx]
            dest_path = class_dir / img_path.name
            shutil.copy2(img_path, dest_path)
        
        # Registrar estatísticas
        stats['classes'][folder_name] = len(selected_indices)
    
    return stats


def create_all_simulations():
    """
    Cria todas as NUM_SIMULATIONS simulações
    
    Cada simulação tem IMAGES_PER_CLASS imagens únicas de cada classe.
    Nenhuma imagem se repete entre simulações.
    
    RETORNA:
    --------
    list : Lista com estatísticas de todas as simulações, ou None se falhou
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + f"CRIANDO {NUM_SIMULATIONS} SIMULAÇÕES")
    print(" " * 15 + f"({IMAGES_PER_CLASS} imagens por classe por simulação)")
    print("=" * 80)
    
    simulations_path = PATHS['simulations']
    
    # Verificar se já existe
    if simulations_path.exists():
        print(f"\n⚠️  A pasta '{simulations_path}' já existe!")
        response = input("   Sobrescrever? (s/n): ").lower()
        
        if response != 's':
            print("❌ Operação cancelada.")
            return None
        
        print("\n🗑️  Removendo pasta antiga...")
        shutil.rmtree(simulations_path)
    
    # Criar pasta principal
    simulations_path.mkdir(parents=True, exist_ok=True)
    
    # Coletar todas as imagens
    all_images = collect_all_images()
    
    # Controlar quais imagens já foram usadas (para evitar repetição)
    used_indices = {class_name: set() for class_name in CLASS_MAPPING.keys()}
    
    # Informações
    total_per_sim = IMAGES_PER_CLASS * len(CLASS_MAPPING)
    total_experiment = NUM_SIMULATIONS * total_per_sim
    
    print(f"\n📊 Cada simulação terá {IMAGES_PER_CLASS} imagens por classe...")
    print(f"📊 Total por simulação: {total_per_sim} imagens")
    print(f"📊 Total no experimento: {total_experiment:,} imagens")
    print("=" * 80)
    
    # Criar cada simulação
    all_stats = []
    
    for sim_num in range(1, NUM_SIMULATIONS + 1):
        stats = create_single_simulation(sim_num, all_images, used_indices)
        
        if stats is None:
            print(f"\n❌ ERRO: Não foi possível criar SIM{sim_num:02d}")
            print("   Imagens insuficientes!")
            return None
        
        all_stats.append(stats)
        
        # Mostrar progresso a cada 5 simulações (ou na primeira)
        if sim_num % 5 == 0 or sim_num == 1:
            classes_info = " ".join([f"{k}={v}" for k, v in stats['classes'].items()])
            print(f"✅ SIM{sim_num:02d} criada: {classes_info} (seed={stats['seed']})")
    
    print("\n" + "=" * 80)
    print(f"✅ Todas as {NUM_SIMULATIONS} simulações criadas com sucesso!")
    print("=" * 80)
    
    # Mostrar uso de imagens
    print(f"\n📊 USO DE IMAGENS:")
    for original_class in CLASS_MAPPING.keys():
        total_available = len(all_images[original_class])
        total_used = len(used_indices[original_class])
        percentage = (total_used / total_available) * 100
        print(f"   {original_class}: {total_used:,}/{total_available:,} usadas ({percentage:.1f}%)")
    
    return all_stats


def generate_summary_report(all_stats):
    """
    Gera relatório resumido das simulações criadas
    
    PARÂMETROS:
    -----------
    all_stats : list
        Lista com estatísticas de todas as simulações
    """
    
    print("\n" + "=" * 80)
    print(" " * 25 + "RELATÓRIO RESUMIDO")
    print("=" * 80)
    
    print(f"\n📊 RESUMO DE {len(all_stats)} SIMULAÇÕES:")
    print("-" * 80)
    
    # Verificar consistência
    happy_counts = [s['classes']['happy'] for s in all_stats]
    sad_counts = [s['classes']['sad'] for s in all_stats]
    
    print(f"\n✅ CLASSE HAPPY:")
    print(f"   Por simulação: {IMAGES_PER_CLASS} imagens")
    print(f"   Total usado: {sum(happy_counts):,} imagens")
    is_consistent_happy = len(set(happy_counts)) == 1 and happy_counts[0] == IMAGES_PER_CLASS
    print(f"   Consistência: {'✅ OK' if is_consistent_happy else '❌ ERRO'}")
    
    print(f"\n✅ CLASSE SAD:")
    print(f"   Por simulação: {IMAGES_PER_CLASS} imagens")
    print(f"   Total usado: {sum(sad_counts):,} imagens")
    is_consistent_sad = len(set(sad_counts)) == 1 and sad_counts[0] == IMAGES_PER_CLASS
    print(f"   Consistência: {'✅ OK' if is_consistent_sad else '❌ ERRO'}")
    
    print(f"\n📁 ESTRUTURA CRIADA:")
    print(f"   {PATHS['simulations'].relative_to(PATHS['root'])}/")
    print(f"   ├── SIM01/")
    print(f"   │   ├── happy/    ({IMAGES_PER_CLASS} imagens)")
    print(f"   │   └── sad/      ({IMAGES_PER_CLASS} imagens)")
    print(f"   ├── SIM02/")
    print(f"   │   └── ... (mesma estrutura)")
    print(f"   └── ... até SIM{NUM_SIMULATIONS:02d}/")
    
    # Salvar relatório em arquivo
    report_file = PATHS['simulations'] / "simulations_summary.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE SIMULAÇÕES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total de Simulações: {len(all_stats)}\n")
        f.write(f"Imagens por classe por simulação: {IMAGES_PER_CLASS}\n")
        f.write(f"Total de imagens por simulação: {IMAGES_PER_CLASS * len(CLASS_MAPPING)}\n")
        f.write(f"Total geral: {NUM_SIMULATIONS * IMAGES_PER_CLASS * len(CLASS_MAPPING)} imagens\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("DETALHES POR SIMULAÇÃO:\n")
        f.write("-" * 80 + "\n\n")
        
        for stats in all_stats:
            f.write(f"SIM{stats['simulation']:02d} (seed={stats['seed']}):\n")
            for folder_name, count in stats['classes'].items():
                f.write(f"  {folder_name}: {count} imagens\n")
            f.write("\n")
    
    print(f"\n💾 Relatório salvo em: {report_file}")

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal - Menu interativo
    
    OPÇÕES:
    1. Criar simulações (baixar dados raw → criar 30 simulações → verificar)
    2. Apenas verificar (verifica dados raw e simulações existentes)
    3. Cancelar
    
    RETORNA:
    --------
    bool : True se sucesso, False caso contrário
    """
    
    print("\n" + "🎭 " * 25)
    print(" " * 20 + "PREPARAÇÃO DAS SIMULAÇÕES")
    print(" " * 25 + "Happy vs Sad")
    print("🎭 " * 25 + "\n")
    
    # Mostrar configuração atual
    print("📋 CONFIGURAÇÃO DO PROJETO:")
    print("-" * 80)
    print(f"   Projeto: {PATHS['root']}")
    print(f"   Dados raw: {PATHS['data_raw']}")
    print(f"   Simulações: {PATHS['simulations']}")
    print(f"   Número de simulações: {NUM_SIMULATIONS}")
    print(f"   Imagens por classe: {IMAGES_PER_CLASS}")
    print(f"   Classes: {CLASSES}")
    print("-" * 80)
    
    try:
        # Menu de opções
        print("\n📋 OPÇÕES:")
        print("   1. Criar simulações (verificar dados + criar 30 simulações)")
        print("   2. Apenas verificar (verifica dados raw e simulações existentes)")
        print("   3. Cancelar")
        
        choice = input("\n❓ Escolha uma opção (1/2/3): ").strip()
        
        if choice == '3':
            print("\n❌ Operação cancelada pelo usuário.")
            return False
        
        elif choice == '2':
            # ================================================================
            # MODO: APENAS VERIFICAÇÃO
            # ================================================================
            print("\n" + "=" * 80)
            print("MODO: VERIFICAÇÃO DE DADOS EXISTENTES")
            print("=" * 80)
            
            # Verificar dados raw
            print("\n[ETAPA 1/2] Verificação dos Dados Originais")
            raw_ok = verify_source_data()
            
            # Verificar simulações
            print("\n[ETAPA 2/2] Verificação das Simulações")
            sims_ok = verify_simulations()
            
            # Resumo
            print("\n" + "=" * 80)
            print(" " * 25 + "RESUMO DA VERIFICAÇÃO")
            print("=" * 80)
            
            print(f"\n   Dados originais (data/raw): {'✅ OK' if raw_ok else '❌ PROBLEMA'}")
            print(f"   Simulações (data/simulations): {'✅ OK' if sims_ok else '❌ PROBLEMA'}")
            
            if raw_ok and sims_ok:
                print("\n✅ Tudo verificado e pronto para uso!")
                return True
            else:
                print("\n⚠️  Há problemas. Verifique os erros acima.")
                if not raw_ok:
                    print("   💡 Execute: python src/data/import_data.py")
                if not sims_ok:
                    print("   💡 Execute a opção 1 para criar as simulações")
                return False
        
        elif choice == '1':
            # ================================================================
            # MODO: CRIAR SIMULAÇÕES
            # ================================================================
            
            # ETAPA 1: Verificar dados originais
            print("\n[ETAPA 1/4] Verificação dos Dados Originais")
            if not verify_source_data():
                print("\n❌ Dados originais não estão prontos!")
                print("   Execute: python src/data/import_data.py")
                return False
            
            input("\n⏸️  Pressione ENTER para continuar...")
            
            # ETAPA 2: Criar simulações
            print("\n[ETAPA 2/4] Criação das Simulações")
            all_stats = create_all_simulations()
            
            if not all_stats:
                return False
            
            # ETAPA 3: Gerar relatório
            print("\n[ETAPA 3/4] Geração do Relatório")
            generate_summary_report(all_stats)
            
            # ETAPA 4: Verificação final
            print("\n[ETAPA 4/4] Verificação Final")
            if not verify_simulations():
                return False
        
        else:
            print("\n❌ Opção inválida. Escolha 1, 2 ou 3.")
            return False
        
        # ====================================================================
        # SUCESSO!
        # ====================================================================
        print("\n" + "=" * 80)
        print(" " * 25 + "🎉 TUDO PRONTO!")
        print("=" * 80)
        
        print("\n✅ Simulações preparadas com sucesso!")
        print(f"📁 Localização: {PATHS['simulations']}")
        
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("   1. Simulações estão em: data/simulations/")
        print("   2. Execute o treinamento dos modelos:")
        print("      • Gemini: python src/gemini/run_gemini.py")
        print("      • YOLOv8: python src/roboflow_yolo8/run_yolo.py")
        print("   3. Após o treinamento, compare os resultados:")
        print("      • python src/evaluation/compare_models.py")
        
        print("\n" + "=" * 80)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operação interrompida pelo usuário.")
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
    
    if success:
        print("\n✅ Script executado com sucesso!")
        exit(0)
    else:
        print("\n❌ Script finalizado com erros.")
        exit(1)