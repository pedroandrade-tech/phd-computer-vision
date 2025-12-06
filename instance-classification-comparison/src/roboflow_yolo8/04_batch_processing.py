"""
04_batch_processing.py - Processar Todas as 30 Simulações
=========================================================

ETAPA 4: Processamento em lote de todas as simulações

O QUE FAZ:
- Processa automaticamente SIM01 até SIM30
- Calcula métricas para cada simulação
- Salva resultados individuais (CSV + JSON)
- Cria tabela consolidada com TODAS as métricas
- Calcula estatísticas descritivas (média, desvio padrão, etc.)

ESTRUTURA DOS RESULTADOS:
results/roboflow_yolo8/
├── roboflow_sims/
│   ├── sim01_detalhado.csv
│   ├── sim01_metrics.json
│   ├── sim02_detalhado.csv
│   ├── sim02_metrics.json
│   └── ... até sim30
├── all_metrics.csv              ← TODAS métricas consolidadas ⭐
└── summary_statistics.json      ← Estatísticas resumidas ⭐

USO:
python src/roboflow_yolo8/04_batch_processing.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Adicionar raiz do projeto ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PATHS,
    CLASSES,
    ROBOFLOW_API_KEY,
    CLASS_MAPPING,
    NUM_SIMULATIONS,
    IMAGES_PER_CLASS,
    METRICS,
    get_simulation_path,
    create_directories
)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def load_model_config():
    """
    Carrega a configuração do modelo salva na Etapa 2
    
    RETORNA:
    --------
    dict : Configuração do modelo, ou None se erro
    """
    
    config_path = PATHS['roboflow_config']
    
    if not config_path.exists():
        print(f"❌ Configuração não encontrada: {config_path}")
        print("   Execute primeiro: python src/roboflow_yolo8/02_connector.py")
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)


def connect_and_load_model(config):
    """
    Conecta ao Roboflow e carrega o modelo
    
    PARÂMETROS:
    -----------
    config : dict
        Configuração do modelo
    
    RETORNA:
    --------
    model : Modelo carregado, ou None se erro
    """
    
    if not ROBOFLOW_API_KEY:
        print("\n❌ ROBOFLOW_API_KEY não configurada!")
        return None
    
    try:
        from roboflow import Roboflow
        
        print(f"\n🔌 Conectando ao Roboflow...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        
        project = rf.workspace(config['workspace']).project(config['project'])
        version = project.version(config['version'])
        model = version.model
        
        print("✅ Modelo carregado!")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Erro ao carregar modelo: {e}")
        return None


def predict_emotion(model, image_path, confidence_threshold=40):
    """
    Faz predição de emoção em uma imagem
    
    RETORNA:
    --------
    dict com: predicted_class, confidence, detected, error
    """
    
    try:
        prediction = model.predict(
            image_path,
            confidence=confidence_threshold,
            overlap=30
        )
        
        result = prediction.json()
        predictions_list = result.get('predictions', [])
        
        if len(predictions_list) == 0:
            return {
                'predicted_class': None,
                'confidence': 0.0,
                'detected': False,
                'error': 'Nenhum rosto detectado'
            }
        
        first_detection = predictions_list[0]
        detected_class = first_detection.get('class', 'unknown')
        confidence = first_detection.get('confidence', 0.0)
        
        return {
            'predicted_class': detected_class,
            'confidence': confidence,
            'detected': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'predicted_class': None,
            'confidence': 0.0,
            'detected': False,
            'error': str(e)
        }


def process_single_simulation(model, sim_number, confidence_threshold=40, save_detailed=True):
    """
    Processa uma simulação completa e calcula métricas
    
    PARÂMETROS:
    -----------
    model : roboflow.Model
        Modelo carregado
    sim_number : int
        Número da simulação (1 a 30)
    confidence_threshold : int
        Threshold de confiança
    save_detailed : bool
        Se True, salva CSV detalhado
    
    RETORNA:
    --------
    dict : Métricas da simulação, ou None se erro
    """
    
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    
    sim_folder = get_simulation_path(sim_number)
    
    if not sim_folder.exists():
        print(f"   ❌ {sim_folder.name} não encontrada")
        return None
    
    results_list = []
    
    # Processar cada classe
    for class_name in CLASSES:
        class_folder = sim_folder / class_name
        
        # Pegar imagens
        image_files = list(class_folder.glob("*.jpg")) + \
                      list(class_folder.glob("*.jpeg")) + \
                      list(class_folder.glob("*.png"))
        
        # Processar cada imagem
        for image_path in image_files:
            result = predict_emotion(model, str(image_path), confidence_threshold)
            
            results_list.append({
                'image_name': image_path.name,
                'true_class': class_name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'detected': result['detected'],
                'error': result['error']
            })
    
    # Criar DataFrame
    df = pd.DataFrame(results_list)
    
    # Tratar None
    df['predicted_class'] = df['predicted_class'].fillna('unknown')
    
    # Filtrar apenas predições válidas (happy ou sad)
    valid_mask = df['predicted_class'].isin(['happy', 'sad'])
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print(f"   ⚠️  Nenhuma predição válida em SIM{sim_number:02d}")
        return None
    
    # Converter para numérico
    y_true = df_valid['true_class'].map(CLASS_MAPPING)
    y_pred = df_valid['predicted_class'].map(CLASS_MAPPING)
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Montar resultado
    metrics = {
        'simulation': f'SIM{sim_number:02d}',
        'simulation_number': sim_number,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_images': len(df),
        'valid_predictions': len(df_valid),
        'detected_count': int(df['detected'].sum()),
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    }
    
    # Salvar arquivos individuais
    if save_detailed:
        # CSV detalhado
        csv_path = PATHS['roboflow_sims'] / f"sim{sim_number:02d}_detalhado.csv"
        df.to_csv(csv_path, index=False)
        
        # JSON métricas
        json_path = PATHS['roboflow_sims'] / f"sim{sim_number:02d}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics


def create_consolidated_table(all_metrics):
    """
    Cria tabela consolidada com todas as métricas
    
    PARÂMETROS:
    -----------
    all_metrics : list
        Lista de dicionários com métricas de cada simulação
    
    RETORNA:
    --------
    pd.DataFrame : Tabela consolidada
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "CONSOLIDANDO RESULTADOS")
    print("=" * 80)
    
    df = pd.DataFrame(all_metrics)
    
    # Selecionar e ordenar colunas
    columns = [
        'simulation_number',
        'simulation',
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'total_images',
        'valid_predictions'
    ]
    
    df = df[columns].sort_values('simulation_number')
    
    print(f"\n📊 TABELA CONSOLIDADA ({len(df)} simulações):")
    print(df.head(10).to_string(index=False))
    if len(df) > 10:
        print(f"   ... e mais {len(df) - 10} simulações")
    
    # Salvar
    csv_path = PATHS['roboflow_metrics']
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Tabela consolidada salva: {csv_path.name}")
    
    return df


def calculate_summary_statistics(df):
    """
    Calcula estatísticas descritivas
    
    PARÂMETROS:
    -----------
    df : pd.DataFrame
        Tabela consolidada
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "ESTATÍSTICAS DESCRITIVAS")
    print("=" * 80)
    
    metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Mostrar estatísticas
    print("\n📊 RESUMO ESTATÍSTICO:")
    print(df[metrics_cols].describe().to_string())
    
    # Detalhes por métrica
    print("\n" + "-" * 80)
    
    statistics = {
        'model': 'YOLOv8_Roboflow',
        'num_simulations': len(df),
        'timestamp': datetime.now().isoformat(),
        'metrics': {}
    }
    
    for metric in metrics_cols:
        values = df[metric]
        
        stats = {
            'mean': float(values.mean()),
            'median': float(values.median()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'q1': float(values.quantile(0.25)),
            'q3': float(values.quantile(0.75))
        }
        
        statistics['metrics'][metric] = stats
        
        print(f"\n📈 {metric.upper()}:")
        print(f"   Média:         {stats['mean']:.4f}")
        print(f"   Mediana:       {stats['median']:.4f}")
        print(f"   Desvio Padrão: {stats['std']:.4f}")
        print(f"   Mínimo:        {stats['min']:.4f}")
        print(f"   Máximo:        {stats['max']:.4f}")
        print(f"   Q1 (25%):      {stats['q1']:.4f}")
        print(f"   Q3 (75%):      {stats['q3']:.4f}")
    
    # Salvar
    json_path = PATHS['roboflow_stats']
    with open(json_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\n💾 Estatísticas salvas: {json_path.name}")
    
    return statistics


def verify_all_results():
    """
    Verifica se todos os resultados existem e estão corretos
    
    RETORNA:
    --------
    bool : True se tudo OK, False caso contrário
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICANDO RESULTADOS EXISTENTES")
    print("=" * 80)
    
    all_ok = True
    simulations_ok = 0
    simulations_with_issues = []
    
    # Verificar pasta principal
    if not PATHS['results_roboflow'].exists():
        print(f"\n❌ Pasta de resultados não encontrada: {PATHS['results_roboflow']}")
        return False
    
    print(f"\n✅ Pasta de resultados: {PATHS['results_roboflow']}")
    
    # Verificar cada simulação
    print(f"\n📊 VERIFICANDO {NUM_SIMULATIONS} SIMULAÇÕES:")
    print("-" * 80)
    
    for sim_num in range(1, NUM_SIMULATIONS + 1):
        csv_path = PATHS['roboflow_sims'] / f"sim{sim_num:02d}_detalhado.csv"
        json_path = PATHS['roboflow_sims'] / f"sim{sim_num:02d}_metrics.json"
        
        csv_ok = csv_path.exists()
        json_ok = json_path.exists()
        
        if csv_ok and json_ok:
            simulations_ok += 1
        else:
            simulations_with_issues.append(sim_num)
            all_ok = False
    
    print(f"\n   Simulações completas: {simulations_ok}/{NUM_SIMULATIONS}")
    
    if simulations_with_issues:
        print(f"   ❌ Simulações com problemas: {simulations_with_issues[:10]}")
        if len(simulations_with_issues) > 10:
            print(f"      ... e mais {len(simulations_with_issues) - 10}")
    
    # Verificar arquivos consolidados
    print(f"\n📊 ARQUIVOS CONSOLIDADOS:")
    
    metrics_ok = PATHS['roboflow_metrics'].exists()
    stats_ok = PATHS['roboflow_stats'].exists()
    
    print(f"   {'✅' if metrics_ok else '❌'} all_metrics.csv")
    print(f"   {'✅' if stats_ok else '❌'} summary_statistics.json")
    
    if not metrics_ok or not stats_ok:
        all_ok = False
    
    # Se tudo OK, mostrar estatísticas
    if all_ok and stats_ok:
        print("\n" + "-" * 80)
        print("📈 ESTATÍSTICAS SALVAS:")
        
        with open(PATHS['roboflow_stats'], 'r') as f:
            stats = json.load(f)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in stats.get('metrics', {}):
                m = stats['metrics'][metric]
                print(f"   {metric.upper():12s}: {m['mean']:.4f} ± {m['std']:.4f}")
    
    # Resumo final
    print("\n" + "=" * 80)
    
    if all_ok:
        print("✅ TODOS OS RESULTADOS VERIFICADOS COM SUCESSO!")
        print(f"   • {simulations_ok} simulações processadas")
        print(f"   • Tabela consolidada: all_metrics.csv")
        print(f"   • Estatísticas: summary_statistics.json")
    else:
        print("❌ HÁ PROBLEMAS NOS RESULTADOS")
        print("   Execute a opção 1 para processar as simulações faltantes")
    
    return all_ok


def process_all_simulations(model, config, start_sim=1, end_sim=None):
    """
    Processa todas as simulações
    
    PARÂMETROS:
    -----------
    model : roboflow.Model
        Modelo carregado
    config : dict
        Configuração do modelo
    start_sim : int
        Simulação inicial (default: 1)
    end_sim : int
        Simulação final (default: NUM_SIMULATIONS)
    
    RETORNA:
    --------
    list : Lista com métricas de todas as simulações
    """
    
    if end_sim is None:
        end_sim = NUM_SIMULATIONS
    
    num_to_process = end_sim - start_sim + 1
    
    print("\n" + "=" * 80)
    print(f" " * 15 + f"PROCESSANDO SIMULAÇÕES {start_sim} A {end_sim}")
    print("=" * 80)
    
    print(f"\n📊 Total a processar: {num_to_process} simulações")
    print(f"📊 Imagens por simulação: {IMAGES_PER_CLASS * len(CLASSES)}")
    print(f"📊 Total de imagens: {num_to_process * IMAGES_PER_CLASS * len(CLASSES)}")
    
    # Criar pastas
    create_directories(['results_roboflow', 'roboflow_sims'])
    
    all_metrics = []
    start_time = time.time()
    
    for idx, sim_num in enumerate(range(start_sim, end_sim + 1), 1):
        sim_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"📁 [{idx}/{num_to_process}] Processando SIM{sim_num:02d}...")
        
        metrics = process_single_simulation(
            model,
            sim_num,
            config.get('confidence_threshold', 40),
            save_detailed=True
        )
        
        if metrics:
            all_metrics.append(metrics)
            
            sim_elapsed = time.time() - sim_start_time
            
            print(f"   ✅ Concluída em {sim_elapsed:.1f}s")
            print(f"   📈 Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   📈 Precision: {metrics['precision']:.4f}")
            print(f"   📈 Recall:    {metrics['recall']:.4f}")
            print(f"   📈 F1-Score:  {metrics['f1_score']:.4f}")
            
            # Estimativa de tempo restante
            if idx < num_to_process:
                avg_time = (time.time() - start_time) / idx
                remaining = avg_time * (num_to_process - idx)
                print(f"   ⏱️  Tempo restante estimado: {remaining/60:.1f} min")
        else:
            print(f"   ❌ Erro ao processar SIM{sim_num:02d}")
        
        # Pausa para não sobrecarregar API
        if idx < num_to_process:
            time.sleep(1)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(" " * 20 + "PROCESSAMENTO CONCLUÍDO!")
    print("=" * 80)
    print(f"\n⏱️  Tempo total: {total_time/60:.2f} minutos")
    print(f"✅ Simulações processadas: {len(all_metrics)}/{num_to_process}")
    
    return all_metrics

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal - Menu interativo
    
    OPÇÕES:
    1. Processar todas as simulações (SIM01 a SIM30)
    2. Processar intervalo específico
    3. Apenas verificar resultados existentes
    4. Cancelar
    """
    
    print("\n" + "🚀 " * 25)
    print(" " * 10 + "ETAPA 4: PROCESSAMENTO EM LOTE - TODAS AS SIMULAÇÕES")
    print(" " * 25 + "YOLOv8 + Roboflow")
    print("🚀 " * 25 + "\n")
    
    print("📋 CONFIGURAÇÃO:")
    print("-" * 80)
    print(f"   Simulações: SIM01 a SIM{NUM_SIMULATIONS:02d}")
    print(f"   Classes: {CLASSES}")
    print(f"   Imagens por classe: {IMAGES_PER_CLASS}")
    print(f"   Total por simulação: {IMAGES_PER_CLASS * len(CLASSES)}")
    print(f"   Total geral: {NUM_SIMULATIONS * IMAGES_PER_CLASS * len(CLASSES)} imagens")
    print("-" * 80)
    
    try:
        # Menu
        print("\n📋 OPÇÕES:")
        print("   1. Processar TODAS as simulações (SIM01 a SIM30)")
        print("   2. Processar intervalo específico (ex: SIM14 a SIM30)")
        print("   3. Apenas verificar resultados existentes")
        print("   4. Cancelar")
        
        choice = input("\n❓ Escolha uma opção (1/2/3/4): ").strip()
        
        if choice == '4':
            print("\n❌ Operação cancelada.")
            return False
        
        elif choice == '3':
            # ================================================================
            # MODO: APENAS VERIFICAÇÃO
            # ================================================================
            return verify_all_results()
        
        elif choice in ['1', '2']:
            # ================================================================
            # MODO: PROCESSAR
            # ================================================================
            
            # Definir intervalo
            if choice == '1':
                start_sim = 1
                end_sim = NUM_SIMULATIONS
            else:
                try:
                    start_sim = int(input("   Simulação inicial (1-30): ").strip())
                    end_sim = int(input("   Simulação final (1-30): ").strip())
                    
                    if not (1 <= start_sim <= NUM_SIMULATIONS and 1 <= end_sim <= NUM_SIMULATIONS):
                        print("❌ Valores devem estar entre 1 e 30")
                        return False
                    if start_sim > end_sim:
                        print("❌ Simulação inicial deve ser menor ou igual à final")
                        return False
                except ValueError:
                    print("❌ Digite números válidos")
                    return False
            
            print(f"\n📊 Processando SIM{start_sim:02d} a SIM{end_sim:02d}")
            
            # Confirmação
            num_to_process = end_sim - start_sim + 1
            total_images = num_to_process * IMAGES_PER_CLASS * len(CLASSES)
            estimated_time = num_to_process * 2  # ~2 min por simulação
            
            print(f"\n⚠️  ATENÇÃO:")
            print(f"   • {num_to_process} simulações serão processadas")
            print(f"   • {total_images} imagens no total")
            print(f"   • Tempo estimado: {estimated_time} minutos")
            
            confirm = input("\n   Continuar? (s/n): ").strip().lower()
            if confirm != 's':
                print("❌ Operação cancelada.")
                return False
            
            # 1. Carregar configuração
            print("\n[1/4] Carregando configuração...")
            config = load_model_config()
            if config is None:
                return False
            
            # 2. Carregar modelo
            print("\n[2/4] Carregando modelo...")
            model = connect_and_load_model(config)
            if model is None:
                return False
            
            # 3. Processar simulações
            print("\n[3/4] Processando simulações...")
            all_metrics = process_all_simulations(model, config, start_sim, end_sim)
            
            if len(all_metrics) == 0:
                print("\n❌ Nenhuma simulação foi processada com sucesso!")
                return False
            
            # 4. Consolidar resultados
            print("\n[4/4] Consolidando resultados...")
            
            # Se processou todas, criar tabela consolidada
            if start_sim == 1 and end_sim == NUM_SIMULATIONS:
                df = create_consolidated_table(all_metrics)
                calculate_summary_statistics(df)
            else:
                # Se processou parcial, tentar consolidar com existentes
                print("\n📊 Processamento parcial - consolidando com resultados existentes...")
                
                # Carregar todos os JSONs existentes
                all_existing = []
                for sim_num in range(1, NUM_SIMULATIONS + 1):
                    json_path = PATHS['roboflow_sims'] / f"sim{sim_num:02d}_metrics.json"
                    if json_path.exists():
                        with open(json_path, 'r') as f:
                            all_existing.append(json.load(f))
                
                if len(all_existing) == NUM_SIMULATIONS:
                    print(f"✅ Todas as {NUM_SIMULATIONS} simulações encontradas!")
                    df = create_consolidated_table(all_existing)
                    calculate_summary_statistics(df)
                else:
                    print(f"⚠️  Apenas {len(all_existing)}/{NUM_SIMULATIONS} simulações encontradas")
                    print("   Execute novamente para processar as faltantes")
        
        else:
            print("\n❌ Opção inválida.")
            return False
        
        # ====================================================================
        # SUCESSO
        # ====================================================================
        print("\n" + "=" * 80)
        print(" " * 25 + "🎉 ETAPA 4 CONCLUÍDA!")
        print("=" * 80)
        
        print(f"""
✅ O que fizemos:
   1. Processamos as simulações automaticamente
   2. Calculamos métricas para cada uma
   3. Salvamos resultados individuais (CSV + JSON)
   4. Criamos tabela consolidada
   5. Calculamos estatísticas descritivas

📁 ESTRUTURA DE ARQUIVOS:
   {PATHS['results_roboflow'].name}/
   ├── roboflow_sims/
   │   ├── sim01_detalhado.csv ... sim{NUM_SIMULATIONS:02d}_detalhado.csv
   │   └── sim01_metrics.json ... sim{NUM_SIMULATIONS:02d}_metrics.json
   ├── all_metrics.csv              ⭐ Tabela consolidada
   └── summary_statistics.json      ⭐ Estatísticas

🎯 PRÓXIMOS PASSOS:
   • Gerar gráficos (BoxPlot, Linha)
   • Comparar com outros modelos (Gemini)
   • Executar teste de Wilcoxon
   
   Execute: python src/evaluation/compare_models.py
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