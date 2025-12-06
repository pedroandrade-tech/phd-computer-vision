"""
03_inference.py - Processar Uma Simulação Completa
==================================================

ETAPA 3: Inferência em uma simulação (SIM01) com Gemini Flash

O QUE FAZ:
- Processa TODAS as 200 imagens da SIM01 (100 happy + 100 sad)
- Faz predição com Gemini Flash
- Respeita rate limit (15 req/min)
- Calcula as 4 métricas: Accuracy, Precision, Recall, F1-Score
- Gera Matriz de Confusão
- Salva resultados detalhados

RATE LIMIT:
- API gratuita: 15 requisições/minuto
- 200 imagens ÷ 15 req/min = ~13-15 minutos
- Pausas automáticas para respeitar limite

USO:
python src/gemini/03_inference.py
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
    """
    Classificador de emoções usando Google Gemini Flash
    """
    
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
    
    print("=" * 80)
    print(" " * 20 + "CARREGANDO MODELO")
    print("=" * 80)
    
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


def process_simulation(classifier, sim_number):
    """
    Processa uma simulação completa com rate limiting
    
    PARÂMETROS:
    -----------
    classifier : GeminiClassifier
        Classificador conectado
    sim_number : int
        Número da simulação
    
    RETORNA:
    --------
    pd.DataFrame : Resultados, ou None se erro
    """
    
    sim_folder = get_simulation_path(sim_number)
    
    if not sim_folder.exists():
        print(f"❌ Simulação não encontrada: {sim_folder}")
        return None
    
    print(f"\n📁 Processando: {sim_folder.name}")
    print(f"⏱️  Rate limit: {GEMINI_REQUESTS_PER_MINUTE} req/min")
    print(f"⏱️  Pausa entre requisições: {GEMINI_SECONDS_PER_REQUEST:.1f}s")
    
    results_list = []
    overall_start = time.time()
    
    # Processar cada classe
    for class_name in CLASSES:
        class_folder = sim_folder / class_name
        
        print(f"\n   📂 Classe: {class_name}")
        
        # Pegar imagens
        image_files = list(class_folder.glob("*.jpg")) + \
                      list(class_folder.glob("*.jpeg")) + \
                      list(class_folder.glob("*.png"))
        
        num_images = len(image_files)
        print(f"      📸 Imagens: {num_images}")
        
        if num_images == 0:
            continue
        
        # Processar cada imagem
        for idx, image_path in enumerate(image_files, 1):
            # Fazer predição
            result = classifier.predict(str(image_path))
            
            results_list.append({
                'image_name': image_path.name,
                'image_path': str(image_path),
                'true_class': class_name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'detected': result['detected'],
                'error': result['error'],
                'raw_response': result['raw_response']
            })
            
            # Mostrar progresso a cada 10 imagens
            if idx % 10 == 0 or idx == num_images:
                elapsed = time.time() - overall_start
                total_processed = len(results_list)
                total_expected = IMAGES_PER_CLASS * len(CLASSES)
                progress = total_processed / total_expected * 100
                
                # Estimar tempo restante
                if total_processed > 0:
                    avg_time = elapsed / total_processed
                    remaining = avg_time * (total_expected - total_processed)
                    print(f"      {class_name}: {idx}/{num_images} | "
                          f"Total: {total_processed}/{total_expected} ({progress:.1f}%) | "
                          f"Restante: ~{remaining/60:.1f}min")
            
            # Pausa para respeitar rate limit
            if idx < num_images or class_name == CLASSES[0]:
                time.sleep(GEMINI_SECONDS_PER_REQUEST)
    
    total_time = time.time() - overall_start
    
    print(f"\n✅ Processamento concluído!")
    print(f"   ⏱️  Tempo total: {total_time/60:.2f} minutos")
    print(f"   📊 Imagens processadas: {len(results_list)}")
    
    return pd.DataFrame(results_list), total_time


def calculate_metrics(df):
    """
    Calcula métricas a partir do DataFrame de resultados
    
    RETORNA:
    --------
    dict : Métricas calculadas
    """
    
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    
    print("\n" + "=" * 80)
    print(" " * 20 + "CALCULANDO MÉTRICAS")
    print("=" * 80)
    
    # Tratar valores None
    df['predicted_class'] = df['predicted_class'].fillna('unknown')
    
    # Estatísticas de detecção
    detected_count = df['detected'].sum()
    not_detected_count = len(df) - detected_count
    
    print(f"\n🔍 DETECÇÃO:")
    print(f"   Detectadas: {detected_count} ({detected_count/len(df)*100:.1f}%)")
    print(f"   Não detectadas: {not_detected_count} ({not_detected_count/len(df)*100:.1f}%)")
    
    # Filtrar predições válidas
    valid_mask = df['predicted_class'].isin(['happy', 'sad'])
    df_valid = df[valid_mask].copy()
    
    print(f"\n📊 PREDIÇÕES VÁLIDAS:")
    print(f"   Total: {len(df)}")
    print(f"   Válidas (happy/sad): {len(df_valid)}")
    print(f"   Inválidas: {len(df) - len(df_valid)}")
    
    if len(df_valid) == 0:
        print("\n❌ Nenhuma predição válida!")
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
    
    # Mostrar resultados
    print("\n" + "=" * 80)
    print(" " * 25 + "📊 RESULTADOS")
    print("=" * 80)
    
    print(f"\n✅ MÉTRICAS:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\n📊 MATRIZ DE CONFUSÃO:")
    print(f"\n                Predito")
    print(f"              Sad    Happy")
    print(f"Real  Sad  |  {cm[0,0]:3d}  |  {cm[0,1]:3d}  |")
    print(f"      Happy|  {cm[1,0]:3d}  |  {cm[1,1]:3d}  |")
    
    print(f"\n📝 INTERPRETAÇÃO:")
    print(f"   TN (sad→sad):     {cm[0,0]}")
    print(f"   FP (sad→happy):   {cm[0,1]} ❌")
    print(f"   FN (happy→sad):   {cm[1,0]} ❌")
    print(f"   TP (happy→happy): {cm[1,1]}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_images': len(df),
        'valid_predictions': len(df_valid),
        'detected_count': int(detected_count),
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    }


def save_confusion_matrix_plot(df, sim_number):
    """Salva o gráfico da matriz de confusão"""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        df_valid = df[df['predicted_class'].isin(['happy', 'sad'])].copy()
        
        if len(df_valid) == 0:
            return
        
        y_true = df_valid['true_class'].map(CLASS_MAPPING)
        y_pred = df_valid['predicted_class'].map(CLASS_MAPPING)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Sad', 'Happy'],
                    yticklabels=['Sad', 'Happy'])
        plt.title(f'Matriz de Confusão - SIM{sim_number:02d} (Gemini Flash)')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        
        output_path = PATHS['results_gemini'] / f"confusion_matrix_sim{sim_number:02d}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"\n💾 Matriz de confusão salva: {output_path.name}")
        
    except ImportError:
        print("\n⚠️  matplotlib/seaborn não instalados - gráfico não gerado")


def save_results(df, metrics, sim_number, processing_time):
    """Salva os resultados da simulação"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "SALVANDO RESULTADOS")
    print("=" * 80)
    
    # Criar pastas
    create_directories(['results_gemini', 'gemini_sims'])
    
    # Salvar CSV detalhado
    csv_path = PATHS['gemini_sims'] / f"sim{sim_number:02d}_detalhado.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Resultados detalhados: {csv_path.name}")
    
    # Carregar config para pegar model_id
    config = load_gemini_config()
    model_id = config.get('model_id', 'gemini-2.0-flash') if config else 'gemini-2.0-flash'
    
    # Salvar métricas JSON
    metrics_data = {
        'simulation': f'SIM{sim_number:02d}',
        'simulation_number': sim_number,
        'model': 'Gemini Flash',
        'model_id': model_id,
        'processing_time_minutes': float(processing_time / 60),
        'timestamp': datetime.now().isoformat(),
        **metrics
    }
    
    json_path = PATHS['gemini_sims'] / f"sim{sim_number:02d}_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"💾 Métricas: {json_path.name}")


def verify_existing_results(sim_number):
    """Verifica se já existem resultados para uma simulação"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICANDO RESULTADOS EXISTENTES")
    print("=" * 80)
    
    csv_path = PATHS['gemini_sims'] / f"sim{sim_number:02d}_detalhado.csv"
    json_path = PATHS['gemini_sims'] / f"sim{sim_number:02d}_metrics.json"
    
    print(f"\n📁 Verificando SIM{sim_number:02d}:")
    
    csv_ok = csv_path.exists()
    json_ok = json_path.exists()
    
    print(f"   {'✅' if csv_ok else '❌'} CSV detalhado: {csv_path.name}")
    print(f"   {'✅' if json_ok else '❌'} Métricas JSON: {json_path.name}")
    
    if not (csv_ok and json_ok):
        print(f"\n❌ Resultados incompletos para SIM{sim_number:02d}")
        return False
    
    # Carregar e mostrar métricas
    try:
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"\n📊 MÉTRICAS SALVAS:")
        print(f"   Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"   Precision: {metrics.get('precision', 0):.4f}")
        print(f"   Recall:    {metrics.get('recall', 0):.4f}")
        print(f"   F1-Score:  {metrics.get('f1_score', 0):.4f}")
        print(f"   Total imagens: {metrics.get('total_images', 0)}")
        print(f"   Predições válidas: {metrics.get('valid_predictions', 0)}")
        print(f"   Tempo processamento: {metrics.get('processing_time_minutes', 0):.1f} min")
        
        # Verificar CSV
        df = pd.read_csv(csv_path)
        expected = IMAGES_PER_CLASS * len(CLASSES)
        print(f"\n📋 CSV tem {len(df)} registros (esperado: {expected})")
        
        print(f"\n✅ Resultados verificados para SIM{sim_number:02d}!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro ao verificar: {e}")
        return False

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal - Menu interativo
    
    OPÇÕES:
    1. Processar SIM01 (carregar modelo → processar → métricas → salvar)
    2. Apenas verificar resultados existentes da SIM01
    3. Cancelar
    """
    
    print("\n" + "🔬 " * 25)
    print(" " * 15 + "ETAPA 3: PROCESSAR SIMULAÇÃO")
    print(" " * 25 + "Gemini Flash")
    print("🔬 " * 25 + "\n")
    
    SIM_NUMBER = 1
    
    print("📋 CONFIGURAÇÃO:")
    print("-" * 80)
    print(f"   Simulação: SIM{SIM_NUMBER:02d}")
    print(f"   Classes: {CLASSES}")
    print(f"   Imagens por classe: {IMAGES_PER_CLASS}")
    print(f"   Total de imagens: {IMAGES_PER_CLASS * len(CLASSES)}")
    print(f"   Rate limit: {GEMINI_REQUESTS_PER_MINUTE} req/min")
    print(f"   Tempo estimado: ~{IMAGES_PER_CLASS * len(CLASSES) * GEMINI_SECONDS_PER_REQUEST / 60:.0f} minutos")
    print("-" * 80)
    
    try:
        # Menu
        print("\n📋 OPÇÕES:")
        print("   1. Processar SIM01 (⚠️ ~13-15 minutos)")
        print("   2. Apenas verificar resultados existentes da SIM01")
        print("   3. Cancelar")
        
        choice = input("\n❓ Escolha uma opção (1/2/3): ").strip()
        
        if choice == '3':
            print("\n❌ Operação cancelada.")
            return False
        
        elif choice == '2':
            # ================================================================
            # MODO: APENAS VERIFICAÇÃO
            # ================================================================
            return verify_existing_results(SIM_NUMBER)
        
        elif choice == '1':
            # ================================================================
            # MODO: PROCESSAR
            # ================================================================
            
            # Confirmação (demora ~13-15 min)
            print(f"\n⚠️  ATENÇÃO: O processamento leva ~13-15 minutos")
            confirm = input("   Deseja continuar? (s/n): ").strip().lower()
            if confirm != 's':
                print("❌ Operação cancelada.")
                return False
            
            # 1. Criar classificador
            print("\n[1/5] Carregando modelo...")
            classifier = create_classifier()
            if classifier is None:
                return False
            
            # 2. Processar simulação
            print("\n[2/5] Processando simulação...")
            result = process_simulation(classifier, SIM_NUMBER)
            if result is None:
                return False
            
            df, processing_time = result
            
            # 3. Calcular métricas
            print("\n[3/5] Calculando métricas...")
            metrics = calculate_metrics(df)
            if metrics is None:
                return False
            
            # 4. Salvar resultados
            print("\n[4/5] Salvando resultados...")
            save_results(df, metrics, SIM_NUMBER, processing_time)
            
            # 5. Gerar gráfico
            print("\n[5/5] Gerando gráfico...")
            save_confusion_matrix_plot(df, SIM_NUMBER)
        
        else:
            print("\n❌ Opção inválida.")
            return False
        
        # ====================================================================
        # SUCESSO
        # ====================================================================
        print("\n" + "=" * 80)
        print(" " * 25 + "🎉 ETAPA 3 CONCLUÍDA!")
        print("=" * 80)
        
        print(f"""
✅ O que fizemos:
   1. Carregamos o modelo Gemini Flash
   2. Processamos {IMAGES_PER_CLASS * len(CLASSES)} imagens da SIM{SIM_NUMBER:02d}
   3. Respeitamos rate limit ({GEMINI_REQUESTS_PER_MINUTE} req/min)
   4. Calculamos Accuracy, Precision, Recall, F1-Score
   5. Geramos Matriz de Confusão
   6. Salvamos resultados em CSV e JSON

📁 ARQUIVOS GERADOS:
   • {PATHS['gemini_sims'].name}/sim{SIM_NUMBER:02d}_detalhado.csv
   • {PATHS['gemini_sims'].name}/sim{SIM_NUMBER:02d}_metrics.json
   • confusion_matrix_sim{SIM_NUMBER:02d}.png

🎯 PRÓXIMA ETAPA:
   Etapa 4: Processar TODAS as 30 simulações
   
   ⚠️  Tempo estimado: ~6-8 horas (30 × 13 min)
   
   Execute: python src/gemini/04_batch_processing.py
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