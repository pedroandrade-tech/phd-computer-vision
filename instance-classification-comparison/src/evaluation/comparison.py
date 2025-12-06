"""
comparison.py - Comparação dos Modelos YOLOv8 vs Gemini
=======================================================

ETAPA 5: Comparação estatística e visual dos dois modelos

O QUE FAZ:
- Carrega métricas dos 2 modelos (YOLOv8 e Gemini)
- Calcula estatísticas descritivas
- Gera BoxPlots para cada métrica
- Gera Gráficos de Linha (evolução por simulação)
- Realiza Teste Pareado de Wilcoxon (α = 0.05)
- Gera relatório completo de comparação

MÉTRICAS ANALISADAS:
1. Accuracy (Acurácia)
2. Precision (Precisão)
3. Recall (Revocação)
4. F1-Score

TESTE ESTATÍSTICO:
- Wilcoxon Signed-Rank Test
- Nível de confiança: 95% (α = 0.05)
- Teste pareado (mesmas simulações para ambos modelos)

USO:
python src/evaluation/comparison.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Adicionar raiz do projeto ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PATHS,
    NUM_SIMULATIONS,
    METRICS,
    create_directories
)

# ============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# ============================================================================

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Nomes amigáveis para as métricas
METRIC_NAMES = {
    'accuracy': 'Acurácia',
    'precision': 'Precisão',
    'recall': 'Recall',
    'f1_score': 'F1-Score'
}

# ============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ============================================================================

def load_metrics_data():
    """
    Carrega os dados de métricas dos dois modelos
    
    RETORNA:
    --------
    tuple : (df_yolo, df_gemini) ou (None, None) se erro
    """
    
    print("=" * 80)
    print(" " * 15 + "CARREGANDO DADOS DOS MODELOS")
    print("=" * 80)
    
    yolo_path = PATHS['roboflow_metrics']
    gemini_path = PATHS['gemini_metrics']
    
    print(f"\n📁 ARQUIVOS:")
    print(f"   YOLOv8: {yolo_path}")
    print(f"   Gemini: {gemini_path}")
    
    # Verificar se existem
    if not yolo_path.exists():
        print(f"\n❌ ERRO: {yolo_path} não encontrado!")
        print("   Execute: python src/roboflow_yolo8/04_batch_processing.py")
        return None, None
    
    if not gemini_path.exists():
        print(f"\n❌ ERRO: {gemini_path} não encontrado!")
        print("   Execute: python src/gemini/04_batch_processing.py")
        return None, None
    
    print("\n✅ Arquivos encontrados!")
    
    # Carregar CSVs
    df_yolo = pd.read_csv(yolo_path)
    df_gemini = pd.read_csv(gemini_path)
    
    print(f"\n📊 DADOS CARREGADOS:")
    print(f"   YOLOv8: {len(df_yolo)} simulações")
    print(f"   Gemini: {len(df_gemini)} simulações")
    
    # Verificar se têm o mesmo número de simulações
    if len(df_yolo) != len(df_gemini):
        print(f"\n⚠️  AVISO: Número diferente de simulações!")
        
        # Usar apenas simulações em comum
        common_sims = set(df_yolo['simulation_number']).intersection(
            set(df_gemini['simulation_number'])
        )
        df_yolo = df_yolo[df_yolo['simulation_number'].isin(common_sims)]
        df_gemini = df_gemini[df_gemini['simulation_number'].isin(common_sims)]
        
        print(f"   Usando simulações em comum: {len(common_sims)}")
    
    # Ordenar por simulation_number
    df_yolo = df_yolo.sort_values('simulation_number').reset_index(drop=True)
    df_gemini = df_gemini.sort_values('simulation_number').reset_index(drop=True)
    
    print(f"\n✅ Dados preparados: {len(df_yolo)} simulações")
    
    return df_yolo, df_gemini


def print_descriptive_statistics(df_yolo, df_gemini):
    """Exibe estatísticas descritivas dos dois modelos"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "ESTATÍSTICAS DESCRITIVAS")
    print("=" * 80)
    
    for metric in METRICS:
        name = METRIC_NAMES.get(metric, metric)
        
        print(f"\n📈 {name.upper()}:")
        print("-" * 80)
        
        yolo_values = df_yolo[metric]
        gemini_values = df_gemini[metric]
        
        print(f"{'Estatística':<20} {'YOLOv8':<15} {'Gemini':<15} {'Diferença':<15}")
        print("-" * 80)
        
        # Média
        yolo_mean = yolo_values.mean()
        gemini_mean = gemini_values.mean()
        diff_mean = yolo_mean - gemini_mean
        print(f"{'Média':<20} {yolo_mean:<15.4f} {gemini_mean:<15.4f} {diff_mean:+.4f}")
        
        # Mediana
        yolo_median = yolo_values.median()
        gemini_median = gemini_values.median()
        diff_median = yolo_median - gemini_median
        print(f"{'Mediana':<20} {yolo_median:<15.4f} {gemini_median:<15.4f} {diff_median:+.4f}")
        
        # Desvio Padrão
        yolo_std = yolo_values.std()
        gemini_std = gemini_values.std()
        diff_std = yolo_std - gemini_std
        print(f"{'Desvio Padrão':<20} {yolo_std:<15.4f} {gemini_std:<15.4f} {diff_std:+.4f}")
        
        # Mínimo e Máximo
        print(f"{'Mínimo':<20} {yolo_values.min():<15.4f} {gemini_values.min():<15.4f}")
        print(f"{'Máximo':<20} {yolo_values.max():<15.4f} {gemini_values.max():<15.4f}")

# ============================================================================
# FUNÇÕES DE GERAÇÃO DE GRÁFICOS
# ============================================================================

def generate_individual_boxplots(df_yolo, df_gemini):
    """Gera um BoxPlot para cada métrica"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "GERANDO BOXPLOTS INDIVIDUAIS")
    print("=" * 80)
    
    plots_path = PATHS['comparison_plots']
    
    for metric in METRICS:
        name = METRIC_NAMES.get(metric, metric)
        
        # Preparar dados
        data_to_plot = {
            'YOLOv8': df_yolo[metric],
            'Gemini': df_gemini[metric]
        }
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Criar BoxPlot
        bp = ax.boxplot(
            [data_to_plot['YOLOv8'], data_to_plot['Gemini']],
            labels=['YOLOv8', 'Gemini'],
            patch_artist=True,
            notch=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red',
                          markeredgecolor='red', markersize=8)
        )
        
        # Colorir boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Adicionar pontos individuais
        for i, (model_name, values) in enumerate(data_to_plot.items(), 1):
            x = np.random.normal(i, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.4, s=30, color='navy')
        
        # Estatísticas no gráfico
        yolo_mean = data_to_plot['YOLOv8'].mean()
        gemini_mean = data_to_plot['Gemini'].mean()
        
        text_str = f'YOLOv8: μ={yolo_mean:.4f}\nGemini: μ={gemini_mean:.4f}'
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Configurações
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'Comparação de {name} - YOLOv8 vs Gemini\n({len(df_yolo)} Simulações)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legenda
        legend_elements = [
            plt.Line2D([0], [0], marker='D', color='w',
                      markerfacecolor='red', markersize=10, label='Média')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # Salvar
        filename = f'boxplot_{metric}.png'
        filepath = plots_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ {filename}")
    
    print(f"\n💾 BoxPlots salvos em: {plots_path}")


def generate_combined_boxplot(df_yolo, df_gemini):
    """Gera um BoxPlot comparativo com todas as métricas"""
    
    print("\n📊 Criando BoxPlot comparativo (todas as métricas)...")
    
    plots_path = PATHS['comparison_plots']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Preparar posições
    positions_yolo = [1, 3, 5, 7]
    positions_gemini = [1.8, 3.8, 5.8, 7.8]
    
    data_yolo = [df_yolo[m] for m in METRICS]
    data_gemini = [df_gemini[m] for m in METRICS]
    
    # BoxPlot YOLOv8
    bp1 = ax.boxplot(data_yolo, positions=positions_yolo, widths=0.6,
                     patch_artist=True, notch=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red',
                                   markeredgecolor='red', markersize=8))
    
    # BoxPlot Gemini
    bp2 = ax.boxplot(data_gemini, positions=positions_gemini, widths=0.6,
                     patch_artist=True, notch=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red',
                                   markeredgecolor='red', markersize=8))
    
    # Colorir
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    
    # Configurações
    metric_labels = [METRIC_NAMES.get(m, m) for m in METRICS]
    ax.set_xticks([1.4, 3.4, 5.4, 7.4])
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Valor da Métrica', fontsize=12)
    ax.set_title(f'Comparação de Todas as Métricas - YOLOv8 vs Gemini\n({len(df_yolo)} Simulações)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legenda
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='lightblue', alpha=0.7, label='YOLOv8'),
        plt.Rectangle((0, 0), 1, 1, fc='lightgreen', alpha=0.7, label='Gemini')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    filename = 'boxplot_all_metrics_comparison.png'
    filepath = plots_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ {filename}")


def generate_line_plots(df_yolo, df_gemini):
    """Gera gráficos de linha para comparação por simulação"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "GERANDO GRÁFICOS DE LINHA")
    print("=" * 80)
    
    plots_path = PATHS['comparison_plots']
    simulations = df_yolo['simulation_number']
    
    # ========================================================================
    # Gráfico 1: Accuracy e F1-Score
    # ========================================================================
    
    print("\n📈 Criando gráfico de linha (Accuracy e F1-Score)...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Accuracy
    ax1.plot(simulations, df_yolo['accuracy'], marker='o', linewidth=2,
            label='YOLOv8', color='blue', markersize=6, alpha=0.7)
    ax1.plot(simulations, df_gemini['accuracy'], marker='s', linewidth=2,
            label='Gemini', color='green', markersize=6, alpha=0.7)
    
    yolo_acc_mean = df_yolo['accuracy'].mean()
    gemini_acc_mean = df_gemini['accuracy'].mean()
    
    ax1.axhline(y=yolo_acc_mean, color='blue', linestyle='--', linewidth=1.5,
               alpha=0.5, label=f'Média YOLOv8: {yolo_acc_mean:.4f}')
    ax1.axhline(y=gemini_acc_mean, color='green', linestyle='--', linewidth=1.5,
               alpha=0.5, label=f'Média Gemini: {gemini_acc_mean:.4f}')
    
    ax1.set_xlabel('Simulação', fontsize=12)
    ax1.set_ylabel('Accuracy (Acurácia)', fontsize=12)
    ax1.set_title('Comparação de Accuracy por Simulação', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(simulations[::2])
    
    # F1-Score
    ax2.plot(simulations, df_yolo['f1_score'], marker='o', linewidth=2,
            label='YOLOv8', color='blue', markersize=6, alpha=0.7)
    ax2.plot(simulations, df_gemini['f1_score'], marker='s', linewidth=2,
            label='Gemini', color='green', markersize=6, alpha=0.7)
    
    yolo_f1_mean = df_yolo['f1_score'].mean()
    gemini_f1_mean = df_gemini['f1_score'].mean()
    
    ax2.axhline(y=yolo_f1_mean, color='blue', linestyle='--', linewidth=1.5,
               alpha=0.5, label=f'Média YOLOv8: {yolo_f1_mean:.4f}')
    ax2.axhline(y=gemini_f1_mean, color='green', linestyle='--', linewidth=1.5,
               alpha=0.5, label=f'Média Gemini: {gemini_f1_mean:.4f}')
    
    ax2.set_xlabel('Simulação', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('Comparação de F1-Score por Simulação', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(simulations[::2])
    
    plt.tight_layout()
    
    filename = 'line_accuracy_f1score_comparison.png'
    filepath = plots_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ {filename}")
    
    # ========================================================================
    # Gráfico 2: Todas as métricas
    # ========================================================================
    
    print("\n📈 Criando gráfico de linha (todas as métricas)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        name = METRIC_NAMES.get(metric, metric)
        
        ax.plot(simulations, df_yolo[metric], marker='o', linewidth=2,
               label='YOLOv8', color='blue', markersize=5, alpha=0.7)
        ax.plot(simulations, df_gemini[metric], marker='s', linewidth=2,
               label='Gemini', color='green', markersize=5, alpha=0.7)
        
        yolo_mean = df_yolo[metric].mean()
        gemini_mean = df_gemini[metric].mean()
        
        ax.axhline(y=yolo_mean, color='blue', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=gemini_mean, color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Simulação', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} por Simulação', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(simulations[::3])
    
    plt.suptitle('Comparação de Todas as Métricas - YOLOv8 vs Gemini',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    filename = 'line_all_metrics_comparison.png'
    filepath = plots_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ {filename}")

# ============================================================================
# TESTE ESTATÍSTICO DE WILCOXON
# ============================================================================

def run_wilcoxon_test(df_yolo, df_gemini):
    """
    Executa o Teste Pareado de Wilcoxon para todas as métricas
    
    RETORNA:
    --------
    dict : Resultados do teste para cada métrica
    """
    
    print("\n" + "=" * 80)
    print(" " * 15 + "TESTE PAREADO DE WILCOXON")
    print("=" * 80)
    
    print("\n📊 TESTE ESTATÍSTICO:")
    print("   Teste: Wilcoxon Signed-Rank Test (pareado)")
    print("   Nível de significância: α = 0.05")
    print("   Confiança: 95%")
    print("   H0: Não há diferença entre os modelos")
    print("   H1: Há diferença significativa")
    
    print("\n" + "=" * 80)
    print("RESULTADOS DO TESTE DE WILCOXON:")
    print("=" * 80)
    
    wilcoxon_results = {}
    
    for metric in METRICS:
        name = METRIC_NAMES.get(metric, metric)
        
        print(f"\n📈 {name.upper()}:")
        print("-" * 80)
        
        yolo_values = df_yolo[metric].values
        gemini_values = df_gemini[metric].values
        
        # Realizar teste de Wilcoxon
        statistic, p_value = stats.wilcoxon(yolo_values, gemini_values)
        
        # Interpretação
        is_significant = p_value < 0.05
        
        # Calcular diferenças
        differences = yolo_values - gemini_values
        mean_diff = differences.mean()
        
        print(f"   Estatística W: {statistic:.4f}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Diferença média (YOLOv8 - Gemini): {mean_diff:+.4f}")
        
        if is_significant:
            print(f"   ✅ SIGNIFICATIVO (p < 0.05)")
            if mean_diff > 0:
                print(f"   🏆 YOLOv8 é significativamente MELHOR")
            else:
                print(f"   🏆 Gemini é significativamente MELHOR")
        else:
            print(f"   ❌ NÃO SIGNIFICATIVO (p ≥ 0.05)")
            print(f"   Conclusão: Desempenho estatisticamente similar")
        
        # Guardar resultados
        wilcoxon_results[metric] = {
            'metric_name': name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_significant': bool(is_significant),
            'significance_level': 0.05,
            'mean_difference': float(mean_diff),
            'yolo_mean': float(yolo_values.mean()),
            'gemini_mean': float(gemini_values.mean())
        }
    
    return wilcoxon_results


def generate_wilcoxon_plot(wilcoxon_results):
    """Gera visualização dos resultados do teste de Wilcoxon"""
    
    print("\n📊 Criando visualização do Teste de Wilcoxon...")
    
    plots_path = PATHS['comparison_plots']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Preparar dados
    metric_labels = [METRIC_NAMES.get(m, m) for m in METRICS]
    p_values = [wilcoxon_results[m]['p_value'] for m in METRICS]
    mean_diffs = [wilcoxon_results[m]['mean_difference'] for m in METRICS]
    
    # Criar gráfico de barras
    x_pos = np.arange(len(metric_labels))
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    
    bars = ax.bar(x_pos, mean_diffs, color=colors, alpha=0.7, edgecolor='black')
    
    # Linha de referência
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Adicionar p-values nas barras
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'p={p_val:.4f}',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')
    
    # Configurações
    ax.set_xlabel('Métricas', fontsize=12)
    ax.set_ylabel('Diferença Média (YOLOv8 - Gemini)', fontsize=12)
    ax.set_title('Teste de Wilcoxon - Diferenças entre Modelos\n(Barras vermelhas = diferença significativa, p < 0.05)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legenda
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='Significativo (p < 0.05)'),
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.7, label='Não significativo (p ≥ 0.05)')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    filename = 'wilcoxon_test_results.png'
    filepath = plots_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ {filename}")

# ============================================================================
# FUNÇÕES DE SALVAMENTO
# ============================================================================

def save_results(df_yolo, df_gemini, wilcoxon_results):
    """Salva os resultados em JSON e relatório de texto"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "SALVANDO RESULTADOS")
    print("=" * 80)
    
    plots_path = PATHS['comparison_plots']
    
    # Salvar resultados do Wilcoxon em JSON
    wilcoxon_json = plots_path / "wilcoxon_test_results.json"
    with open(wilcoxon_json, 'w') as f:
        json.dump({
            'test': 'Wilcoxon Signed-Rank Test',
            'paired': True,
            'significance_level': 0.05,
            'num_simulations': len(df_yolo),
            'timestamp': datetime.now().isoformat(),
            'results': wilcoxon_results
        }, f, indent=2)
    
    print(f"\n💾 Resultados do teste: {wilcoxon_json.name}")
    
    # Criar relatório em texto
    report_file = plots_path / "comparison_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "RELATÓRIO DE COMPARAÇÃO\n")
        f.write(" " * 15 + "YOLOv8 vs Gemini Flash\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Simulações: {len(df_yolo)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("ESTATÍSTICAS DESCRITIVAS\n")
        f.write("=" * 80 + "\n\n")
        
        for metric in METRICS:
            name = METRIC_NAMES.get(metric, metric)
            f.write(f"{name.upper()}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  YOLOv8:  μ={df_yolo[metric].mean():.4f} ± {df_yolo[metric].std():.4f}\n")
            f.write(f"  Gemini:  μ={df_gemini[metric].mean():.4f} ± {df_gemini[metric].std():.4f}\n")
            f.write(f"  Diferença: {df_yolo[metric].mean() - df_gemini[metric].mean():+.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("TESTE DE WILCOXON (α = 0.05)\n")
        f.write("=" * 80 + "\n\n")
        
        for metric in METRICS:
            name = METRIC_NAMES.get(metric, metric)
            result = wilcoxon_results[metric]
            f.write(f"{name.upper()}:\n")
            f.write(f"  Estatística W: {result['statistic']:.4f}\n")
            f.write(f"  P-value: {result['p_value']:.6f}\n")
            f.write(f"  Significativo: {'Sim' if result['is_significant'] else 'Não'}\n")
            
            if result['is_significant']:
                winner = "YOLOv8" if result['mean_difference'] > 0 else "Gemini"
                f.write(f"  Vencedor: {winner}\n")
            else:
                f.write(f"  Conclusão: Desempenho similar\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONCLUSÃO GERAL\n")
        f.write("=" * 80 + "\n\n")
        
        # Contar vitórias
        yolo_wins = sum(1 for m in METRICS if wilcoxon_results[m]['is_significant']
                       and wilcoxon_results[m]['mean_difference'] > 0)
        gemini_wins = sum(1 for m in METRICS if wilcoxon_results[m]['is_significant']
                         and wilcoxon_results[m]['mean_difference'] < 0)
        ties = len(METRICS) - yolo_wins - gemini_wins
        
        f.write(f"Métricas com diferença significativa:\n")
        f.write(f"  YOLOv8 melhor: {yolo_wins}/{len(METRICS)}\n")
        f.write(f"  Gemini melhor: {gemini_wins}/{len(METRICS)}\n")
        f.write(f"  Sem diferença: {ties}/{len(METRICS)}\n\n")
        
        if yolo_wins > gemini_wins:
            f.write("VENCEDOR GERAL: YOLOv8\n")
        elif gemini_wins > yolo_wins:
            f.write("VENCEDOR GERAL: Gemini\n")
        else:
            f.write("RESULTADO: Empate técnico - modelos com desempenho similar\n")
    
    print(f"💾 Relatório: {report_file.name}")

# ============================================================================
# FUNÇÃO DE VERIFICAÇÃO
# ============================================================================

def verify_existing_results():
    """Verifica se já existem resultados de comparação"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICANDO RESULTADOS EXISTENTES")
    print("=" * 80)
    
    plots_path = PATHS['comparison_plots']
    
    # Verificar pasta
    if not plots_path.exists():
        print(f"\n❌ Pasta de comparação não encontrada: {plots_path}")
        return False
    
    print(f"\n✅ Pasta encontrada: {plots_path}")
    
    # Verificar arquivos esperados
    expected_files = [
        'boxplot_accuracy.png',
        'boxplot_precision.png',
        'boxplot_recall.png',
        'boxplot_f1_score.png',
        'boxplot_all_metrics_comparison.png',
        'line_accuracy_f1score_comparison.png',
        'line_all_metrics_comparison.png',
        'wilcoxon_test_results.png',
        'wilcoxon_test_results.json',
        'comparison_report.txt'
    ]
    
    print(f"\n📁 ARQUIVOS ESPERADOS:")
    
    files_ok = 0
    for filename in expected_files:
        filepath = plots_path / filename
        exists = filepath.exists()
        print(f"   {'✅' if exists else '❌'} {filename}")
        if exists:
            files_ok += 1
    
    print(f"\n📊 Arquivos encontrados: {files_ok}/{len(expected_files)}")
    
    # Se tiver o JSON, mostrar resultados
    wilcoxon_json = plots_path / "wilcoxon_test_results.json"
    if wilcoxon_json.exists():
        print("\n" + "-" * 80)
        print("📈 RESULTADOS DO TESTE DE WILCOXON:")
        
        with open(wilcoxon_json, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', {})
        for metric in METRICS:
            if metric in results:
                r = results[metric]
                sig = "✅ SIG" if r['is_significant'] else "❌ NS"
                print(f"   {METRIC_NAMES.get(metric, metric):12s}: p={r['p_value']:.4f} {sig}")
    
    all_ok = files_ok == len(expected_files)
    
    print("\n" + "=" * 80)
    if all_ok:
        print("✅ TODOS OS RESULTADOS VERIFICADOS!")
    else:
        print("⚠️  Alguns arquivos estão faltando. Execute a opção 1.")
    
    return all_ok

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal - Menu interativo
    
    OPÇÕES:
    1. Executar comparação completa
    2. Apenas verificar resultados existentes
    3. Cancelar
    """
    
    print("\n" + "🏆 " * 25)
    print(" " * 10 + "ETAPA 5: COMPARAÇÃO DE MODELOS")
    print(" " * 20 + "YOLOv8 vs Gemini Flash")
    print("🏆 " * 25 + "\n")
    
    print("📋 CONFIGURAÇÃO:")
    print("-" * 80)
    print(f"   Simulações: {NUM_SIMULATIONS}")
    print(f"   Métricas: {METRICS}")
    print(f"   Teste estatístico: Wilcoxon (α = 0.05)")
    print("-" * 80)
    
    try:
        # Menu
        print("\n📋 OPÇÕES:")
        print("   1. Executar comparação completa")
        print("   2. Apenas verificar resultados existentes")
        print("   3. Cancelar")
        
        choice = input("\n❓ Escolha uma opção (1/2/3): ").strip()
        
        if choice == '3':
            print("\n❌ Operação cancelada.")
            return False
        
        elif choice == '2':
            # ================================================================
            # MODO: APENAS VERIFICAÇÃO
            # ================================================================
            return verify_existing_results()
        
        elif choice == '1':
            # ================================================================
            # MODO: COMPARAÇÃO COMPLETA
            # ================================================================
            
            # 1. Carregar dados
            print("\n[1/6] Carregando dados...")
            df_yolo, df_gemini = load_metrics_data()
            if df_yolo is None or df_gemini is None:
                return False
            
            # 2. Estatísticas descritivas
            print("\n[2/6] Calculando estatísticas...")
            print_descriptive_statistics(df_yolo, df_gemini)
            
            # 3. Criar pasta de plots
            print("\n[3/6] Preparando ambiente...")
            create_directories(['comparison_plots'])
            
            # 4. Gerar gráficos
            print("\n[4/6] Gerando gráficos...")
            generate_individual_boxplots(df_yolo, df_gemini)
            generate_combined_boxplot(df_yolo, df_gemini)
            generate_line_plots(df_yolo, df_gemini)
            
            # 5. Teste de Wilcoxon
            print("\n[5/6] Executando teste estatístico...")
            wilcoxon_results = run_wilcoxon_test(df_yolo, df_gemini)
            generate_wilcoxon_plot(wilcoxon_results)
            
            # 6. Salvar resultados
            print("\n[6/6] Salvando resultados...")
            save_results(df_yolo, df_gemini, wilcoxon_results)
        
        else:
            print("\n❌ Opção inválida.")
            return False
        
        # ====================================================================
        # SUCESSO
        # ====================================================================
        print("\n" + "=" * 80)
        print(" " * 25 + "🎉 ETAPA 5 CONCLUÍDA!")
        print("=" * 80)
        
        print(f"""
✅ O que fizemos:
   1. Carregamos métricas dos 2 modelos
   2. Calculamos estatísticas descritivas
   3. Geramos BoxPlots individuais e comparativo
   4. Geramos Gráficos de Linha
   5. Executamos Teste de Wilcoxon (α = 0.05)
   6. Salvamos relatório completo

📁 ARQUIVOS GERADOS:
   {PATHS['comparison_plots'].name}/
   ├── boxplot_accuracy.png
   ├── boxplot_precision.png
   ├── boxplot_recall.png
   ├── boxplot_f1_score.png
   ├── boxplot_all_metrics_comparison.png
   ├── line_accuracy_f1score_comparison.png
   ├── line_all_metrics_comparison.png
   ├── wilcoxon_test_results.png
   ├── wilcoxon_test_results.json
   └── comparison_report.txt

📊 COMO INTERPRETAR:
   • BoxPlot: Distribuição e mediana
   • Linha: Evolução por simulação
   • Wilcoxon: p < 0.05 = diferença significativa

🏆 CONCLUSÃO:
   Veja o relatório comparison_report.txt!
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