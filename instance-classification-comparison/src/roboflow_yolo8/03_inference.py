"""
03_inference.py - Processar Uma Simulação Completa
==================================================

ETAPA 3: Inferência em uma simulação (SIM01)

O QUE FAZ:
- Processa TODAS as 200 imagens da SIM01 (100 happy + 100 sad)
- Faz predição para cada imagem
- Compara predição vs realidade
- Calcula as 4 métricas: Accuracy, Precision, Recall, F1-Score
- Gera Matriz de Confusão
- Salva resultados detalhados

MÉTRICAS:
- Accuracy: proporção de acertos gerais
- Precision: das predições "happy", quantas estavam certas?
- Recall: das imagens happy, quantas foram identificadas?
- F1-Score: média harmônica de Precision e Recall

USO:
python src/roboflow_yolo8/03_inference.py
"""

import os
import sys
import json
from pathlib import Path
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
    IMAGES_PER_CLASS,
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
    
    print("=" * 80)
    print(" " * 20 + "CARREGANDO MODELO")
    print("=" * 80)
    
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
        print(f"   Projeto: {config['project']}")
        print(f"   Versão: {config['version']}")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Erro ao carregar modelo: {e}")
        return None


def predict_emotion(model, image_path, confidence_threshold=40):
    """
    Faz predição de emoção em uma imagem
    
    PARÂMETROS:
    -----------
    model : roboflow.Model
        Modelo do Roboflow já carregado
    image_path : str
        Caminho para a imagem
    confidence_threshold : int (default=40)
        Confiança mínima para aceitar predição (0-100)
        
    RETORNA:
    --------
    dict com:
        - 'predicted_class': classe detectada
        - 'confidence': confiança (0.0 a 1.0)
        - 'detected': True/False
        - 'error': mensagem de erro (ou None)
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


def process_simulation(model, sim_number, confidence_threshold=40):
    """
    Processa uma simulação completa
    
    PARÂMETROS:
    -----------
    model : roboflow.Model
        Modelo carregado
    sim_number : int
        Número da simulação (1 a 30)
    confidence_threshold : int
        Threshold de confiança
    
    RETORNA:
    --------
    pd.DataFrame : DataFrame com resultados, ou None se erro
    """
    
    sim_folder = get_simulation_path(sim_number)
    
    if not sim_folder.exists():
        print(f"❌ Simulação não encontrada: {sim_folder}")
        return None
    
    print(f"\n📁 Processando: {sim_folder.name}")
    
    results_list = []
    
    # Processar cada classe
    for class_name in CLASSES:
        class_folder = sim_folder / class_name
        
        print(f"\n   📂 Classe: {class_name}")
        
        # Pegar todas as imagens
        image_files = list(class_folder.glob("*.jpg")) + \
                      list(class_folder.glob("*.jpeg")) + \
                      list(class_folder.glob("*.png"))
        
        num_images = len(image_files)
        print(f"      📸 Imagens: {num_images}")
        
        if num_images == 0:
            print(f"      ⚠️  Nenhuma imagem encontrada!")
            continue
        
        # Processar cada imagem
        print(f"      🔄 Processando...")
        
        for idx, image_path in enumerate(image_files, 1):
            result = predict_emotion(model, str(image_path), confidence_threshold)
            
            results_list.append({
                'image_name': image_path.name,
                'image_path': str(image_path),
                'true_class': class_name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'detected': result['detected'],
                'error': result['error']
            })
            
            # Mostrar progresso a cada 20 imagens
            if idx % 20 == 0 or idx == num_images:
                print(f"         Processadas: {idx}/{num_images}")
    
    # Criar DataFrame
    df = pd.DataFrame(results_list)
    print(f"\n✅ Total processado: {len(df)} imagens")
    
    return df


def calculate_metrics(df):
    """
    Calcula métricas a partir do DataFrame de resultados
    
    PARÂMETROS:
    -----------
    df : pd.DataFrame
        DataFrame com resultados das predições
    
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
    
    print(f"\n🔍 DETECÇÃO DE ROSTOS:")
    print(f"   Detectados: {detected_count} ({detected_count/len(df)*100:.1f}%)")
    print(f"   Não detectados: {not_detected_count} ({not_detected_count/len(df)*100:.1f}%)")
    
    # Filtrar apenas predições válidas (happy ou sad)
    valid_mask = df['predicted_class'].isin(['happy', 'sad'])
    df_valid = df[valid_mask].copy()
    
    print(f"\n📊 PREDIÇÕES VÁLIDAS:")
    print(f"   Total: {len(df)}")
    print(f"   Válidas (happy/sad): {len(df_valid)}")
    print(f"   Inválidas (outras emoções): {len(df) - len(df_valid)}")
    
    if len(df_valid) == 0:
        print("\n❌ Nenhuma predição válida!")
        return None
    
    # Converter para numérico usando CLASS_MAPPING
    y_true = df_valid['true_class'].map(CLASS_MAPPING)
    y_pred = df_valid['predicted_class'].map(CLASS_MAPPING)
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
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
    """
    Salva o gráfico da matriz de confusão
    
    PARÂMETROS:
    -----------
    df : pd.DataFrame
        DataFrame com resultados
    sim_number : int
        Número da simulação
    """
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # Filtrar válidos
        df_valid = df[df['predicted_class'].isin(['happy', 'sad'])].copy()
        
        if len(df_valid) == 0:
            return
        
        y_true = df_valid['true_class'].map(CLASS_MAPPING)
        y_pred = df_valid['predicted_class'].map(CLASS_MAPPING)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Sad', 'Happy'],
                    yticklabels=['Sad', 'Happy'])
        plt.title(f'Matriz de Confusão - SIM{sim_number:02d}')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        
        # Salvar na pasta de resultados
        output_path = PATHS['results_roboflow'] / f"confusion_matrix_sim{sim_number:02d}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"\n💾 Matriz de confusão salva: {output_path.name}")
        
    except ImportError:
        print("\n⚠️  matplotlib/seaborn não instalados - gráfico não gerado")


def save_results(df, metrics, sim_number):
    """
    Salva os resultados da simulação
    
    PARÂMETROS:
    -----------
    df : pd.DataFrame
        DataFrame com resultados detalhados
    metrics : dict
        Métricas calculadas
    sim_number : int
        Número da simulação
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "SALVANDO RESULTADOS")
    print("=" * 80)
    
    # Criar pastas
    create_directories(['results_roboflow', 'roboflow_sims'])
    
    # Salvar CSV detalhado
    csv_path = PATHS['roboflow_sims'] / f"sim{sim_number:02d}_detalhado.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Resultados detalhados: {csv_path.name}")
    
    # Salvar métricas JSON
    metrics_data = {
        'simulation': f'SIM{sim_number:02d}',
        'simulation_number': sim_number,
        **metrics
    }
    
    json_path = PATHS['roboflow_sims'] / f"sim{sim_number:02d}_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"💾 Métricas: {json_path.name}")


def verify_existing_results(sim_number):
    """
    Verifica se já existem resultados para uma simulação
    
    PARÂMETROS:
    -----------
    sim_number : int
        Número da simulação
    
    RETORNA:
    --------
    bool : True se resultados existem e estão OK
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICANDO RESULTADOS EXISTENTES")
    print("=" * 80)
    
    csv_path = PATHS['roboflow_sims'] / f"sim{sim_number:02d}_detalhado.csv"
    json_path = PATHS['roboflow_sims'] / f"sim{sim_number:02d}_metrics.json"
    
    print(f"\n📁 Verificando SIM{sim_number:02d}:")
    
    # Verificar CSV
    csv_ok = csv_path.exists()
    print(f"   {'✅' if csv_ok else '❌'} CSV detalhado: {csv_path.name}")
    
    # Verificar JSON
    json_ok = json_path.exists()
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
        
        # Verificar CSV
        df = pd.read_csv(csv_path)
        print(f"\n📋 CSV tem {len(df)} registros")
        
        if len(df) != IMAGES_PER_CLASS * len(CLASSES):
            print(f"   ⚠️  Esperado: {IMAGES_PER_CLASS * len(CLASSES)} registros")
        
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
    1. Processar SIM01 (carregar modelo → processar → calcular métricas → salvar)
    2. Apenas verificar resultados existentes da SIM01
    3. Cancelar
    """
    
    print("\n" + "🔬 " * 25)
    print(" " * 15 + "ETAPA 3: PROCESSAR SIMULAÇÃO")
    print(" " * 25 + "YOLOv8 + Roboflow")
    print("🔬 " * 25 + "\n")
    
    SIM_NUMBER = 1  # Processamos SIM01 nesta etapa
    
    print(f"📋 CONFIGURAÇÃO:")
    print("-" * 80)
    print(f"   Simulação: SIM{SIM_NUMBER:02d}")
    print(f"   Classes: {CLASSES}")
    print(f"   Imagens por classe: {IMAGES_PER_CLASS}")
    print(f"   Total de imagens: {IMAGES_PER_CLASS * len(CLASSES)}")
    print("-" * 80)
    
    try:
        # Menu
        print("\n📋 OPÇÕES:")
        print("   1. Processar SIM01 (carregar modelo → inferência → métricas)")
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
            
            # 1. Carregar configuração
            print("\n[1/5] Carregando configuração...")
            config = load_model_config()
            if config is None:
                return False
            
            # 2. Carregar modelo
            print("\n[2/5] Carregando modelo...")
            model = connect_and_load_model(config)
            if model is None:
                return False
            
            # 3. Processar simulação
            print("\n[3/5] Processando simulação...")
            df = process_simulation(
                model, 
                SIM_NUMBER, 
                config.get('confidence_threshold', 40)
            )
            if df is None:
                return False
            
            # 4. Calcular métricas
            print("\n[4/5] Calculando métricas...")
            metrics = calculate_metrics(df)
            if metrics is None:
                return False
            
            # 5. Salvar resultados
            print("\n[5/5] Salvando resultados...")
            save_results(df, metrics, SIM_NUMBER)
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
   1. Carregamos o modelo YOLOv8 do Roboflow
   2. Processamos {IMAGES_PER_CLASS * len(CLASSES)} imagens da SIM{SIM_NUMBER:02d}
   3. Calculamos Accuracy, Precision, Recall, F1-Score
   4. Geramos Matriz de Confusão
   5. Salvamos resultados em CSV e JSON

📁 ARQUIVOS GERADOS:
   • {PATHS['roboflow_sims'].name}/sim{SIM_NUMBER:02d}_detalhado.csv
   • {PATHS['roboflow_sims'].name}/sim{SIM_NUMBER:02d}_metrics.json
   • confusion_matrix_sim{SIM_NUMBER:02d}.png

🎯 PRÓXIMA ETAPA:
   Etapa 4: Processar TODAS as 30 simulações
   
   Execute: python src/roboflow_yolo8/04_batch_processing.py
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