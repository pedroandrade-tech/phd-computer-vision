"""
02_connector.py - Conectar ao Gemini e Testar o Modelo
======================================================

ETAPA 2: Conexão com a API e teste do modelo Gemini Flash

O QUE FAZ:
- Conecta à API do Google Gemini usando a chave do .env
- Cria a classe GeminiClassifier
- Faz um teste com 1 imagem
- Compara com YOLOv8 (se disponível)
- Salva configuração do modelo

MODELO:
- Nome: Gemini 2.0 Flash
- Model ID: gemini-2.0-flash
- Tipo: Multimodal (texto + imagem)
- Rate Limit: 15 req/min (grátis)

USO:
python src/gemini/02_connector.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Adicionar raiz do projeto ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PATHS,
    CLASSES,
    GEMINI_API_KEY,
    CLASS_MAPPING,
    get_simulation_path,
    create_directories
)

# ============================================================================
# CLASSE GEMINI CLASSIFIER
# ============================================================================

class GeminiClassifier:
    """
    Classificador de emoções usando Google Gemini Flash
    
    COMO FUNCIONA:
    1. Recebe uma imagem
    2. Redimensiona para 224x224 (melhora qualidade)
    3. Envia para Gemini com prompt
    4. Interpreta resposta em linguagem natural
    5. Retorna classe (happy ou sad)
    
    DIFERENÇA DO YOLOV8:
    - YOLOv8: modelo especializado, retorna JSON estruturado
    - Gemini: modelo geral, usa prompts e retorna texto natural
    """
    
    def __init__(self, api_key, model_id="gemini-2.0-flash"):
        """
        Inicializa o classificador
        
        PARÂMETROS:
        -----------
        api_key : str
            Chave de API do Google Gemini
        model_id : str
            ID do modelo (default: gemini-2.0-flash)
        """
        
        import google.generativeai as genai
        import PIL.Image
        
        # Guardar referências
        self.genai = genai
        self.PIL = PIL
        
        # Configurar API
        genai.configure(api_key=api_key)
        
        # Carregar modelo
        self.model = genai.GenerativeModel(model_id)
        self.model_id = model_id
        
        print(f"   ✅ Modelo {model_id} carregado")
    
    def predict(self, image_path):
        """
        Faz predição de emoção em uma imagem
        
        PARÂMETROS:
        -----------
        image_path : str
            Caminho para a imagem
            
        RETORNA:
        --------
        dict com:
            - 'predicted_class': 'happy', 'sad' ou None
            - 'confidence': None (Gemini não retorna confiança)
            - 'detected': True se classificou, False caso contrário
            - 'error': mensagem de erro (ou None)
            - 'raw_response': resposta bruta do modelo
        """
        
        try:
            # Abrir imagem
            img = self.PIL.Image.open(image_path)
            
            # Redimensionar para 224x224 (melhora análise)
            img = img.resize((224, 224), self.PIL.Image.LANCZOS)
            
            # Prompt otimizado
            prompt = """Look at this face carefully. 
Classify the emotion as either 'Happy' or 'Sad'. 
Answer with ONLY ONE WORD: either 'Happy' or 'Sad'.
Do not add any explanation."""
            
            # Enviar para o modelo
            response = self.model.generate_content([prompt, img])
            
            # Verificar resposta
            if not response.text:
                return {
                    'predicted_class': None,
                    'confidence': None,
                    'detected': False,
                    'error': 'Resposta vazia do modelo',
                    'raw_response': None
                }
            
            # Processar resposta
            result_text = response.text.strip().lower()
            
            # Interpretar (evitar ambiguidade)
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
                'confidence': None,  # Gemini não retorna confiança
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
    """
    Carrega a configuração do modelo salva na Etapa 1
    
    RETORNA:
    --------
    dict : Configuração do modelo, ou None se erro
    """
    
    config_path = PATHS['gemini_config']
    
    if not config_path.exists():
        print(f"❌ Configuração não encontrada: {config_path}")
        print("   Execute primeiro: python src/gemini/01_config.py")
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)


def connect_to_gemini():
    """
    Conecta à API do Gemini e cria o classificador
    
    RETORNA:
    --------
    GeminiClassifier : Classificador pronto, ou None se erro
    """
    
    print("=" * 80)
    print(" " * 20 + "CONECTANDO AO GEMINI")
    print("=" * 80)
    
    # Verificar API key
    if not GEMINI_API_KEY:
        print("\n❌ GEMINI_API_KEY não configurada!")
        print("\n📝 SOLUÇÃO:")
        print("   1. Crie/edite o arquivo .env na raiz do projeto")
        print("   2. Adicione: GEMINI_API_KEY=sua_chave_aqui")
        print("   3. Obtenha em: https://aistudio.google.com/app/apikey")
        return None
    
    print(f"\n🔑 API Key: {GEMINI_API_KEY[:10]}***")
    
    try:
        # Carregar config para pegar model_id
        config = load_gemini_config()
        model_id = config.get('model_id', 'gemini-2.0-flash') if config else 'gemini-2.0-flash'
        
        print(f"\n🔌 Conectando ao modelo {model_id}...")
        
        classifier = GeminiClassifier(
            api_key=GEMINI_API_KEY,
            model_id=model_id
        )
        
        print("✅ Conectado com sucesso!")
        
        return classifier
        
    except ImportError:
        print("\n❌ Biblioteca 'google-generativeai' não instalada!")
        print("   Execute: pip install google-generativeai")
        return None
        
    except Exception as e:
        print(f"\n❌ Erro ao conectar: {e}")
        return None


def test_with_image(classifier):
    """
    Testa o modelo com uma imagem da SIM01
    
    PARÂMETROS:
    -----------
    classifier : GeminiClassifier
        Classificador conectado
    
    RETORNA:
    --------
    dict : Resultado do teste
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "TESTE COM UMA IMAGEM")
    print("=" * 80)
    
    # Pegar uma imagem da SIM01/happy
    sim01 = get_simulation_path(1)
    test_folder = sim01 / "happy"
    
    if not test_folder.exists():
        print(f"\n❌ Pasta de teste não encontrada: {test_folder}")
        print("   Execute primeiro: python src/data/data_prep.py")
        return None
    
    # Pegar primeira imagem
    image_files = list(test_folder.glob("*.jpg")) + \
                  list(test_folder.glob("*.png")) + \
                  list(test_folder.glob("*.jpeg"))
    
    if len(image_files) == 0:
        print(f"\n❌ Nenhuma imagem encontrada em: {test_folder}")
        return None
    
    test_image = image_files[0]
    print(f"\n📸 Imagem de teste: {test_image.name}")
    print(f"   Classe real: happy")
    
    # Fazer predição
    print("\n🔄 Fazendo predição...")
    print("   ⏳ Aguarde (pode levar alguns segundos)...")
    
    start_time = time.time()
    result = classifier.predict(str(test_image))
    elapsed_time = time.time() - start_time
    
    print(f"   ✅ Predição concluída em {elapsed_time:.2f}s")
    
    # Mostrar resultado
    print(f"\n📊 RESULTADO:")
    print(f"   Detectou: {'✅ Sim' if result['detected'] else '❌ Não'}")
    
    if result['detected']:
        print(f"   Classe predita: {result['predicted_class']}")
        print(f"   Resposta bruta: '{result['raw_response']}'")
        
        if result['predicted_class'] == 'happy':
            print(f"\n   ✅ PREDIÇÃO CORRETA!")
        else:
            print(f"\n   ❌ PREDIÇÃO INCORRETA (esperado: happy)")
    else:
        print(f"   Erro: {result['error']}")
    
    return result


def test_multiple_images(classifier, num_images=5):
    """
    Testa o modelo com múltiplas imagens
    
    PARÂMETROS:
    -----------
    classifier : GeminiClassifier
        Classificador conectado
    num_images : int
        Número de imagens para testar
    
    RETORNA:
    --------
    float : Acurácia do teste
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "TESTE COM MÚLTIPLAS IMAGENS")
    print("=" * 80)
    
    sim01 = get_simulation_path(1)
    
    # Pegar imagens de cada classe
    test_images = []
    
    for class_name in CLASSES:
        class_folder = sim01 / class_name
        if class_folder.exists():
            images = list(class_folder.glob("*.jpg"))[:num_images // 2 + 1]
            test_images.extend([(img, class_name) for img in images])
    
    test_images = test_images[:num_images]
    
    print(f"\n🧪 Testando com {len(test_images)} imagens...")
    print(f"\n{'Imagem':<30} {'Real':<10} {'Predito':<10} {'Correto':<10} {'Tempo':<10}")
    print("-" * 70)
    
    correct_count = 0
    total_time = 0
    
    for img_path, true_class in test_images:
        img_name = img_path.name
        
        start = time.time()
        result = classifier.predict(str(img_path))
        elapsed = time.time() - start
        total_time += elapsed
        
        pred_class = result['predicted_class'] if result['detected'] else 'N/A'
        is_correct = "✅" if pred_class == true_class else "❌"
        
        if pred_class == true_class:
            correct_count += 1
        
        print(f"{img_name:<30} {true_class:<10} {pred_class:<10} {is_correct:<10} {elapsed:.2f}s")
        
        # Pausa para respeitar rate limit
        time.sleep(1)
    
    accuracy = correct_count / len(test_images) if test_images else 0
    avg_time = total_time / len(test_images) if test_images else 0
    
    print(f"\n📈 RESUMO:")
    print(f"   Acurácia: {correct_count}/{len(test_images)} ({accuracy:.1%})")
    print(f"   Tempo médio: {avg_time:.2f}s por imagem")
    
    return accuracy


def verify_existing_config():
    """
    Verifica se já existe configuração salva
    
    RETORNA:
    --------
    bool : True se config existe e está OK
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICANDO CONFIGURAÇÃO")
    print("=" * 80)
    
    config_path = PATHS['gemini_config']
    
    if not config_path.exists():
        print(f"\n❌ Configuração não encontrada: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\n✅ Configuração encontrada")
        print(f"   Modelo: {config.get('model_name', 'N/A')}")
        print(f"   Model ID: {config.get('model_id', 'N/A')}")
        print(f"   Rate Limit: {config.get('rate_limit', 'N/A')} req/min")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro ao verificar: {e}")
        return False


def verify_connection():
    """
    Verifica se consegue conectar ao Gemini (sem fazer teste)
    
    RETORNA:
    --------
    bool : True se conexão OK
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "TESTANDO CONEXÃO")
    print("=" * 80)
    
    if not GEMINI_API_KEY:
        print("\n❌ GEMINI_API_KEY não configurada!")
        return False
    
    try:
        import google.generativeai as genai
        
        print(f"\n🔌 Testando conexão...")
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Tentar listar modelos para verificar conexão
        models = genai.list_models()
        model_names = [m.name for m in models if 'gemini' in m.name.lower()]
        
        print(f"✅ Conexão OK!")
        print(f"   Modelos disponíveis: {len(model_names)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro de conexão: {e}")
        return False

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal - Menu interativo
    
    OPÇÕES:
    1. Conectar, testar e validar
    2. Apenas verificar (conexão + config existente)
    3. Cancelar
    """
    
    print("\n" + "🔌 " * 25)
    print(" " * 15 + "ETAPA 2: CONEXÃO E TESTE DO MODELO")
    print(" " * 25 + "Gemini Flash")
    print("🔌 " * 25 + "\n")
    
    try:
        # Menu
        print("📋 OPÇÕES:")
        print("   1. Conectar, testar com imagens e validar")
        print("   2. Apenas verificar (conexão + config existente)")
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
            
            # Verificar config
            config_ok = verify_existing_config()
            
            # Verificar conexão
            connection_ok = verify_connection()
            
            # Resumo
            print("\n" + "=" * 80)
            print(" " * 25 + "RESUMO DA VERIFICAÇÃO")
            print("=" * 80)
            print(f"\n   Configuração: {'✅ OK' if config_ok else '❌ PROBLEMA'}")
            print(f"   Conexão API:  {'✅ OK' if connection_ok else '❌ PROBLEMA'}")
            
            if config_ok and connection_ok:
                print("\n✅ Tudo verificado e pronto!")
                return True
            else:
                print("\n⚠️  Há problemas. Execute a opção 1 para configurar.")
                return False
        
        elif choice == '1':
            # ================================================================
            # MODO: COMPLETO
            # ================================================================
            
            # 1. Conectar
            classifier = connect_to_gemini()
            if classifier is None:
                return False
            
            # 2. Testar com 1 imagem
            result = test_with_image(classifier)
            if result is None:
                return False
            
            # 3. Perguntar se quer teste adicional
            print("\n" + "-" * 80)
            extra_test = input("❓ Deseja testar com mais imagens? (s/n): ").strip().lower()
            
            if extra_test == 's':
                test_multiple_images(classifier, num_images=6)
        
        else:
            print("\n❌ Opção inválida.")
            return False
        
        # ====================================================================
        # SUCESSO
        # ====================================================================
        print("\n" + "=" * 80)
        print(" " * 25 + "🎉 ETAPA 2 CONCLUÍDA!")
        print("=" * 80)
        
        print("""
✅ O que fizemos:
   1. Conectamos ao Gemini Flash com API Key
   2. Criamos a classe GeminiClassifier
   3. Testamos com imagem(ns) real(is)
   4. Verificamos funcionamento

📝 CARACTERÍSTICAS DO GEMINI:
   • Usa prompts em linguagem natural
   • Não retorna confiança numérica
   • Rate limit: 15 req/min (grátis)
   • Tempo médio: ~2-4s por imagem

🎯 PRÓXIMA ETAPA:
   Etapa 3: Processar uma simulação (SIM01)
   
   Execute: python src/gemini/03_inference.py
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