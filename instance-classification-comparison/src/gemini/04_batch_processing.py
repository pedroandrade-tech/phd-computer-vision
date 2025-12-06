"""
04_batch_processing.py - Processar Todas as 30 Simulações
=========================================================

ETAPA 4: Processamento em lote de todas as simulações com Gemini Flash

O QUE FAZ:
- Processa automaticamente SIM01 até SIM30
- Calcula métricas para cada simulação
- Salva resultados individuais (CSV + JSON)
- Cria tabela consolidada com TODAS as métricas
- Calcula estatísticas descritivas (média, desvio padrão, etc.)
- Pula simulações já processadas (permite retomar)

ESTRUTURA DOS RESULTADOS:
results/gemini/
├── gemini_sims/
│   ├── sim01_detalhado.csv
│   ├── sim01_metrics.json
│   ├── sim02_detalhado.csv
│   ├── sim02_metrics.json
│   └── ... até sim30
├── all_metrics.csv              ← TODAS métricas consolidadas ⭐
└── summary_statistics.json      ← Estatísticas resumidas ⭐

TEMPO ESTIMADO:
- 1 simulação = ~13-15 minutos
- 30 simulações = ~6.5-7.5 horas
- Rate limit: 15 requisições/minuto

USO:
python src/gemini/04_batch_processing.py
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
    GEMINI_API_KEY,
    CLASS_MAPPING,
    NUM_SIMULATIONS,
    IMAGES_PER_CLASS,
    GEMINI_REQUESTS_PER_MINUTE,
    GEMINI_SECONDS_PER_REQUEST,
    get_simulation_path,
    create_directories
)

# ============================================================================
# CLASSE GEMINI CLASSIFIER
# ============================================================================

class GeminiClassifier:
    """Classificador de emoções usando Google Gemini Flash"""
    
    def __init__(self, api_key, model_id="gemini-2.0-flash"):
        import google.generativeai as genai
        import PIL.Image
        
        self.genai = genai
        self.PIL = PIL
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)
        self.model_id = model_id
    
    def predict(self, image_path):
        """Faz predição de emoção em uma imagem"""
        
        try:
            img = self.PIL.Image.open(image_path)
            img = img.resize((224, 224), self.PIL.Image.LANCZOS)
            
            prompt = """Look at this face carefully. 
Classify the emotion as either 'Happy' or 'Sad'. 
Answer with ONLY ONE WORD: either 'Happy' or 'Sad'.
Do not add any explanation."""
            
            response = self.model.generate_content([prompt, img])
            
            if not response.text:
                return {
                    'predicted_class': None,
                    'confidence': None,
                    'detected': False,
                    'error': 'Resposta vazia',
                    'raw_response': None
                }
            
            result_text = response.text.strip().lower()
            
            predicted_class = None
            if "happy" in result_text and "sad" not in result_text:
                predicted_class = "happy"
            elif "sad" in result_text and "happy" not in result_text:
                predicted_class = "sad"
            
            if predicted_class is None:
                return {
                    'predicted_class': None,
                    'confidence': None,
                    'detected': False,
                    'error': f'Resposta ambígua: {result_text}',
                    'raw_response': result_text
                }
            
            return {
                'predicted_class': predicted_class,
                'confidence': None,
                'detected': True,
                'error': None,
                'raw_response': result_text
            }
            
        except Exception as e:
            return {
                'predicted_class': None,
                'confidence': None,
                'detected': False,
                'error': str(e),
                'raw_response': None
            }

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def load_gemini_config():
    """Carrega a configuração do modelo"""
    
    config_path = PATHS['gemini_config']
    
    if not config_path.exists():
        print(f"❌ Configuração não encontrada: {config_path}")
        print("   Execute primeiro: python src/gemini/01_config.py")
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)


def create_classifier():
    """Cria o classificador Gemini"""
    
    if not GEMINI_API_KEY:
        print("\n❌ GEMINI_API_KEY não configurada!")
        return None
    
    config = load_gemini_config()
    if config is None:
        return None
    
    model_id = config.get('model_id', 'gemini-2.0-flash')
    
    print(f"\n🔌 Conectando ao {model_id}...")
    
    try:
        classifier = GeminiClassifier(
            api_key=GEMINI_API_KEY,
            model_id=model_id
        )
        print("✅ Classificador pronto!")
        return classifier
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return None


def process_single_simulation(classifier, sim_number):
    """
    Processa uma simulação completa
    
    PARÂMETROS:
    -----------
    classifier : GeminiClassifier
        Classificador conectado
    sim_number : int
        Número da simulação (1 a 30)
    
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
    
    # Verificar se já foi processada
    json_path = PATHS['gemini_sims'] / f"sim{sim_number:02d}_metrics.json"
    if json_path.exists():
        print(f"   ⏭️  SIM{sim_number:02d} já processada, pulando...")
        with open(json_path, 'r') as f:
            return json.load(f)
    
    results_list = []
    sim_start_time = time.time()
    
    # Processar cada classe
    for class_name in CLASSES:
        class_folder = sim_folder / class_name
        
        # Pegar imagens
        image_files = list(class_folder.glob("*.jpg")) + \
                      list(class_folder.glob("*.jpeg")) + \
                      list(class_folder.glob("*.png"))
        
        num_images = len(image_files)
        
        # Processar cada imagem
        for idx, image_path in enumerate(image_files, 1):
            result = classifier.predict(str(image_path))
            
            results_list.append({
                'image_name': image_path.name,
                'true_class': class_name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'detected': result['detected'],
                'error': result['error'],
                'raw_response': result['raw_response']
            })
            
            # Mostrar progresso a cada 20 imagens
            if idx % 20 == 0:
                total = len(results_list)
                expected = IMAGES_PER_CLASS * len(CLASSES)
                print(f"      {class_name}: {idx}/{num_images} | Total: {total}/{expected}")
            
            # Pausa para respeitar rate limit
            if idx < num_images or class_name == CLASSES[0]:
                time.sleep(GEMINI_SECONDS_PER_REQUEST)
    
    processing_time = time.time() - sim_start_time
    
    # Criar DataFrame
    df = pd.DataFrame(results_list)
    
    # Tratar None
    df['predicted_class'] = df['predicted_class'].fillna('unknown')
    
    # Filtrar predições válidas
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
    config = load_gemini_config()
    model_id = config.get('model_id', 'gemini-2.0-flash') if config else 'gemini-2.0-flash'
    
    metrics = {
        'simulation': f'SIM{sim_number:02d}',
        'simulation_number': sim_number,
        'model': 'Gemini Flash',
        'model_id': model_id,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_images': len(df),
        'valid_predictions': len(df_valid),
        'detected_count': int(df['detected'].sum()),
        'processing_time_minutes': float(processing_time / 60),
        'timestamp': datetime.now().isoformat(),
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    }
    
    # Salvar arquivos individuais
    # CSV detalhado
    csv_path = PATHS['gemini_sims'] / f"sim{sim_number:02d}_detalhado.csv"
    df.to_csv(csv_path, index=False)
    
    # JSON métricas
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def create_consolidated_table(all_metrics):
    """
    Cria tabela consolidada com todas as métricas
    
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
    csv_path = PATHS['gemini_metrics']
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Tabela consolidada salva: {csv_path.name}")
    
    return df


def calculate_summary_statistics(df):
    """Calcula estatísticas descritivas"""
    
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
        'model': 'Gemini_Flash',
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
    json_path = PATHS['gemini_stats']
    with open(json_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\n💾 Estatísticas salvas: {json_path.name}")
    
    return statistics


def verify_all_results():
    """Verifica se todos os resultados existem e estão corretos"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICANDO RESULTADOS EXISTENTES")
    print("=" * 80)
    
    all_ok = True
    simulations_ok = 0
    simulations_with_issues = []
    
    # Verificar pasta principal
    if not PATHS['results_gemini'].exists():
        print(f"\n❌ Pasta de resultados não encontrada: {PATHS['results_gemini']}")
        return False
    
    print(f"\n✅ Pasta de resultados: {PATHS['results_gemini']}")
    
    # Verificar cada simulação
    print(f"\n📊 VERIFICANDO {NUM_SIMULATIONS} SIMULAÇÕES:")
    print("-" * 80)
    
    for sim_num in range(1, NUM_SIMULATIONS + 1):
        csv_path = PATHS['gemini_sims'] / f"sim{sim_num:02d}_detalhado.csv"
        json_path = PATHS['gemini_sims'] / f"sim{sim_num:02d}_metrics.json"
        
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
    
    metrics_ok = PATHS['gemini_metrics'].exists()
    stats_ok = PATHS['gemini_stats'].exists()
    
    print(f"   {'✅' if metrics_ok else '❌'} all_metrics.csv")
    print(f"   {'✅' if stats_ok else '❌'} summary_statistics.json")
    
    if not metrics_ok or not stats_ok:
        all_ok = False
    
    # Se tudo OK, mostrar estatísticas
    if all_ok and stats_ok:
        print("\n" + "-" * 80)
        print("📈 ESTATÍSTICAS SALVAS:")
        
        with open(PATHS['gemini_stats'], 'r') as f:
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
        print("   Execute a opção 1 ou 2 para processar as simulações faltantes")
    
    return all_ok


def process_all_simulations(classifier, start_sim=1, end_sim=None):
    """
    Processa todas as simulações
    
    PARÂMETROS:
    -----------
    classifier : GeminiClassifier
        Classificador conectado
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
    create_directories(['results_gemini', 'gemini_sims'])
    
    all_metrics = []
    start_time = time.time()
    
    for idx, sim_num in enumerate(range(start_sim, end_sim + 1), 1):
        sim_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"📁 [{idx}/{num_to_process}] Processando SIM{sim_num:02d}...")
        
        metrics = process_single_simulation(classifier, sim_num)
        
        if metrics:
            all_metrics.append(metrics)
            
            sim_elapsed = time.time() - sim_start_time
            
            print(f"   ✅ Concluída em {sim_elapsed/60:.1f}min")
            print(f"   📈 Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   📈 Precision: {metrics['precision']:.4f}")
            print(f"   📈 Recall:    {metrics['recall']:.4f}")
            print(f"   📈 F1-Score:  {metrics['f1_score']:.4f}")
            
            # Estimativa de tempo restante
            if idx < num_to_process:
                total_elapsed = time.time() - start_time
                avg_time = total_elapsed / idx
                remaining = avg_time * (num_to_process - idx)
                print(f"   ⏱️  Tempo restante estimado: {remaining/60:.1f}min ({remaining/3600:.2f}h)")
        else:
            print(f"   ❌ Erro ao processar SIM{sim_num:02d}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(" " * 20 + "PROCESSAMENTO CONCLUÍDO!")
    print("=" * 80)
    print(f"\n⏱️  Tempo total: {total_time/60:.2f} minutos ({total_time/3600:.2f} horas)")
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
    print(" " * 25 + "Gemini Flash")
    print("🚀 " * 25 + "\n")
    
    print("📋 CONFIGURAÇÃO:")
    print("-" * 80)
    print(f"   Simulações: SIM01 a SIM{NUM_SIMULATIONS:02d}")
    print(f"   Classes: {CLASSES}")
    print(f"   Imagens por classe: {IMAGES_PER_CLASS}")
    print(f"   Total por simulação: {IMAGES_PER_CLASS * len(CLASSES)}")
    print(f"   Total geral: {NUM_SIMULATIONS * IMAGES_PER_CLASS * len(CLASSES)} imagens")
    print(f"   Rate limit: {GEMINI_REQUESTS_PER_MINUTE} req/min")
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
            estimated_time_min = num_to_process * 13  # ~13 min por simulação
            estimated_time_h = estimated_time_min / 60
            
            print(f"\n⚠️  ATENÇÃO:")
            print(f"   • {num_to_process} simulações serão processadas")
            print(f"   • {total_images} imagens no total")
            print(f"   • Tempo estimado: ~{estimated_time_min} minutos ({estimated_time_h:.1f} horas)")
            print(f"   • Simulações já processadas serão PULADAS")
            
            confirm = input("\n   Continuar? (s/n): ").strip().lower()
            if confirm != 's':
                print("❌ Operação cancelada.")
                return False
            
            # 1. Carregar modelo
            print("\n[1/3] Carregando modelo...")
            classifier = create_classifier()
            if classifier is None:
                return False
            
            # 2. Processar simulações
            print("\n[2/3] Processando simulações...")
            all_metrics = process_all_simulations(classifier, start_sim, end_sim)
            
            if len(all_metrics) == 0:
                print("\n❌ Nenhuma simulação foi processada com sucesso!")
                return False
            
            # 3. Consolidar resultados
            print("\n[3/3] Consolidando resultados...")
            
            # Carregar todos os JSONs existentes para consolidar
            print("\n📊 Consolidando com todos os resultados existentes...")
            
            all_existing = []
            for sim_num in range(1, NUM_SIMULATIONS + 1):
                json_path = PATHS['gemini_sims'] / f"sim{sim_num:02d}_metrics.json"
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        all_existing.append(json.load(f))
            
            if len(all_existing) == NUM_SIMULATIONS:
                print(f"✅ Todas as {NUM_SIMULATIONS} simulações encontradas!")
                df = create_consolidated_table(all_existing)
                calculate_summary_statistics(df)
            elif len(all_existing) > 0:
                print(f"⚠️  {len(all_existing)}/{NUM_SIMULATIONS} simulações encontradas")
                df = create_consolidated_table(all_existing)
                calculate_summary_statistics(df)
                print("\n   Execute novamente para processar as faltantes")
            else:
                print("❌ Nenhuma simulação encontrada para consolidar")
        
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
   {PATHS['results_gemini'].name}/
   ├── gemini_sims/
   │   ├── sim01_detalhado.csv ... sim{NUM_SIMULATIONS:02d}_detalhado.csv
   │   └── sim01_metrics.json ... sim{NUM_SIMULATIONS:02d}_metrics.json
   ├── all_metrics.csv              ⭐ Tabela consolidada
   └── summary_statistics.json      ⭐ Estatísticas

🎯 PRÓXIMOS PASSOS:
   • Gerar gráficos (BoxPlot, Linha)
   • Comparar com YOLOv8
   • Executar teste de Wilcoxon
   
   Execute: python src/evaluation/compare_models.py
""")
        
        print("=" * 80)
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operação interrompida.")
        print("   O progresso foi salvo. Execute novamente para retomar.")
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