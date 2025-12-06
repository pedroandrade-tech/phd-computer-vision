"""
02_connector.py - Conectar ao Roboflow e Carregar Modelo
========================================================

ETAPA 2: Conexão com a API e carregamento do modelo YOLOv8

O QUE FAZ:
- Conecta à API do Roboflow usando a chave do .env
- Acessa o projeto "human-face-emotions" versão 28
- Carrega o modelo pré-treinado
- Faz um teste com 1 imagem (opcional)
- Salva configuração do modelo em JSON

MODELO:
- Workspace: emotions-dectection
- Projeto: human-face-emotions
- Versão: 28
- Tipo: YOLOv8 Object Detection
- Classes alvo: happy, sad

USO:
python src/roboflow_yolo8/02_connector.py
"""

import os
import sys
import json
from pathlib import Path

# Adicionar raiz do projeto ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PATHS,
    CLASSES,
    ROBOFLOW_API_KEY,
    CLASS_MAPPING,
    get_simulation_path,
    create_directories
)

# ============================================================================
# INFORMAÇÕES DO MODELO
# ============================================================================

MODEL_INFO = {
    'workspace': 'emotions-dectection',
    'project': 'human-face-emotions',
    'version': 28,
    'model_classes': {
        0: 'anger',
        1: 'content',
        2: 'disgust',
        3: 'fear',
        4: 'happy',      # ← Vamos usar
        5: 'neutral',
        6: 'sad',        # ← Vamos usar
        7: 'surprise'
    },
    'target_classes': ['happy', 'sad'],
    'confidence_threshold': 40
}

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def connect_to_roboflow():
    """
    Conecta à API do Roboflow
    
    RETORNA:
    --------
    Roboflow : Cliente conectado, ou None se erro
    """
    
    print("=" * 80)
    print(" " * 20 + "CONECTANDO AO ROBOFLOW")
    print("=" * 80)
    
    # Verificar API key
    if not ROBOFLOW_API_KEY:
        print("\n❌ ROBOFLOW_API_KEY não configurada!")
        print("\n📝 SOLUÇÃO:")
        print("   1. Crie/edite o arquivo .env na raiz do projeto")
        print("   2. Adicione: ROBOFLOW_API_KEY=sua_chave_aqui")
        print("   3. Obtenha em: https://app.roboflow.com/ → Settings → API")
        return None
    
    print(f"\n🔑 API Key: {ROBOFLOW_API_KEY[:10]}***")
    
    try:
        from roboflow import Roboflow
        
        print("\n🔌 Conectando...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        print("✅ Conectado com sucesso!")
        
        return rf
        
    except ImportError:
        print("\n❌ Biblioteca 'roboflow' não instalada!")
        print("   Execute: pip install roboflow")
        return None
        
    except Exception as e:
        print(f"\n❌ Erro ao conectar: {e}")
        print("\n💡 Verifique:")
        print("   • Sua API key está correta")
        print("   • Você tem conexão com internet")
        return None


def load_model(rf):
    """
    Carrega o modelo do Roboflow
    
    PARÂMETROS:
    -----------
    rf : Roboflow
        Cliente Roboflow conectado
    
    RETORNA:
    --------
    tuple : (model, version) ou (None, None) se erro
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "CARREGANDO MODELO")
    print("=" * 80)
    
    try:
        # Acessar workspace
        print(f"\n📂 Acessando workspace: {MODEL_INFO['workspace']}")
        workspace = rf.workspace(MODEL_INFO['workspace'])
        print(f"✅ Workspace acessado")
        
        # Acessar projeto
        print(f"\n📂 Acessando projeto: {MODEL_INFO['project']}")
        project = workspace.project(MODEL_INFO['project'])
        print(f"✅ Projeto acessado")
        
        # Acessar versão
        print(f"\n📦 Acessando versão: {MODEL_INFO['version']}")
        version = project.version(MODEL_INFO['version'])
        print(f"✅ Versão acessada")
        
        # Carregar modelo
        print("\n🧠 Carregando modelo neural...")
        model = version.model
        print("✅ Modelo carregado com sucesso!")
        
        return model, version
        
    except Exception as e:
        print(f"\n❌ Erro ao carregar modelo: {e}")
        print("\n💡 Possíveis problemas:")
        print("   • Nome do workspace ou projeto incorreto")
        print("   • Versão não existe")
        print("   • Você não tem permissão para acessar")
        return None, None


def show_model_classes():
    """
    Exibe as classes do modelo e o mapeamento binário
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "CLASSES DO MODELO")
    print("=" * 80)
    
    print("\n🎯 CLASSES QUE O MODELO DETECTA:")
    for class_id, emotion in MODEL_INFO['model_classes'].items():
        marker = "✓" if emotion in MODEL_INFO['target_classes'] else "•"
        print(f"   {marker} {class_id}: {emotion}")
    
    print("\n📌 USAREMOS APENAS:")
    print("   ✓ happy (feliz)")
    print("   ✓ sad (triste)")
    
    print("\n🔢 MAPEAMENTO BINÁRIO:")
    print(f"   sad → {CLASS_MAPPING['sad']} (classe negativa)")
    print(f"   happy → {CLASS_MAPPING['happy']} (classe positiva)")


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
        - 'predicted_class': classe detectada ('happy', 'sad', etc)
        - 'confidence': confiança da predição (0.0 a 1.0)
        - 'detected': True se detectou rosto, False caso contrário
        - 'error': mensagem de erro (se houver)
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


def test_with_image(model):
    """
    Testa o modelo com uma imagem da SIM01
    
    PARÂMETROS:
    -----------
    model : roboflow.Model
        Modelo carregado
    
    RETORNA:
    --------
    bool : True se teste OK, False caso contrário
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
        return False
    
    # Pegar primeira imagem
    image_files = list(test_folder.glob("*.jpg")) + \
                  list(test_folder.glob("*.png")) + \
                  list(test_folder.glob("*.jpeg"))
    
    if len(image_files) == 0:
        print(f"\n❌ Nenhuma imagem encontrada em: {test_folder}")
        return False
    
    test_image = image_files[0]
    print(f"\n📸 Imagem de teste: {test_image.name}")
    print(f"   Classe real: happy")
    
    # Fazer predição
    print("\n🔄 Fazendo predição...")
    result = predict_emotion(model, str(test_image))
    
    print(f"\n📊 RESULTADO:")
    print(f"   Detectou rosto: {'✅ Sim' if result['detected'] else '❌ Não'}")
    
    if result['detected']:
        print(f"   Classe predita: {result['predicted_class']}")
        print(f"   Confiança: {result['confidence']:.2%}")
        
        if result['predicted_class'] == 'happy':
            print(f"   ✅ PREDIÇÃO CORRETA!")
        else:
            print(f"   ❌ PREDIÇÃO INCORRETA (esperado: happy)")
            print(f"   💡 Isso pode acontecer - o modelo não é 100% perfeito")
    else:
        print(f"   Erro: {result['error']}")
    
    return result['detected']


def save_model_config():
    """
    Salva a configuração do modelo em JSON
    
    O arquivo é salvo em: models/roboflow_yolo8/roboflow_config.json
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "SALVANDO CONFIGURAÇÃO")
    print("=" * 80)
    
    # Criar pasta se não existir
    create_directories(['models_roboflow'])
    
    config = {
        'workspace': MODEL_INFO['workspace'],
        'project': MODEL_INFO['project'],
        'version': MODEL_INFO['version'],
        'target_classes': MODEL_INFO['target_classes'],
        'binary_mapping': CLASS_MAPPING,
        'confidence_threshold': MODEL_INFO['confidence_threshold']
    }
    
    config_path = PATHS['roboflow_config']
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuração salva em: {config_path}")
    print("\n📋 Conteúdo:")
    print(json.dumps(config, indent=2))


def verify_existing_config():
    """
    Verifica se já existe configuração salva e se está correta
    
    RETORNA:
    --------
    bool : True se config existe e está OK, False caso contrário
    """
    
    print("\n" + "=" * 80)
    print(" " * 20 + "VERIFICANDO CONFIGURAÇÃO EXISTENTE")
    print("=" * 80)
    
    config_path = PATHS['roboflow_config']
    
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
        required_fields = ['workspace', 'project', 'version', 'target_classes']
        missing = [f for f in required_fields if f not in config]
        
        if missing:
            print(f"\n⚠️  Campos faltando: {missing}")
            return False
        
        # Verificar se valores batem
        if config['workspace'] != MODEL_INFO['workspace']:
            print(f"\n⚠️  Workspace diferente: {config['workspace']} vs {MODEL_INFO['workspace']}")
            return False
        
        if config['project'] != MODEL_INFO['project']:
            print(f"\n⚠️  Projeto diferente: {config['project']} vs {MODEL_INFO['project']}")
            return False
        
        if config['version'] != MODEL_INFO['version']:
            print(f"\n⚠️  Versão diferente: {config['version']} vs {MODEL_INFO['version']}")
            return False
        
        print("\n✅ Configuração válida!")
        return True
        
    except json.JSONDecodeError:
        print(f"\n❌ Erro ao ler JSON: arquivo corrompido")
        return False
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return False


def verify_connection_and_model():
    """
    Verifica se consegue conectar ao Roboflow e carregar o modelo
    (sem fazer teste com imagem)
    
    RETORNA:
    --------
    bool : True se tudo OK, False caso contrário
    """
    
    # Conectar
    rf = connect_to_roboflow()
    if rf is None:
        return False
    
    # Carregar modelo
    model, version = load_model(rf)
    if model is None:
        return False
    
    # Mostrar classes
    show_model_classes()
    
    print("\n✅ Conexão e modelo verificados com sucesso!")
    return True

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal - Menu interativo
    
    OPÇÕES:
    1. Conectar, carregar modelo, testar e salvar config
    2. Apenas verificar (conexão + config existente)
    3. Cancelar
    """
    
    print("\n" + "🔌 " * 25)
    print(" " * 15 + "ETAPA 2: CONEXÃO E CARREGAMENTO DO MODELO")
    print(" " * 25 + "YOLOv8 + Roboflow")
    print("🔌 " * 25 + "\n")
    
    # Mostrar info do modelo
    print("📊 MODELO A SER CARREGADO:")
    print("-" * 80)
    print(f"   Workspace: {MODEL_INFO['workspace']}")
    print(f"   Projeto:   {MODEL_INFO['project']}")
    print(f"   Versão:    {MODEL_INFO['version']}")
    print(f"   Classes:   {MODEL_INFO['target_classes']}")
    print("-" * 80)
    
    try:
        # Menu
        print("\n📋 OPÇÕES:")
        print("   1. Conectar, carregar modelo, testar e salvar config")
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
            
            # Verificar config existente
            config_ok = verify_existing_config()
            
            # Verificar conexão
            print("\n[Testando conexão com Roboflow...]")
            connection_ok = verify_connection_and_model()
            
            # Resumo
            print("\n" + "=" * 80)
            print(" " * 25 + "RESUMO DA VERIFICAÇÃO")
            print("=" * 80)
            print(f"\n   Configuração salva: {'✅ OK' if config_ok else '❌ PROBLEMA'}")
            print(f"   Conexão Roboflow:   {'✅ OK' if connection_ok else '❌ PROBLEMA'}")
            
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
            
            # 1. Conectar ao Roboflow
            rf = connect_to_roboflow()
            if rf is None:
                return False
            
            # 2. Carregar modelo
            model, version = load_model(rf)
            if model is None:
                return False
            
            # 3. Mostrar classes
            show_model_classes()
            
            # 4. Testar com imagem
            test_with_image(model)
            
            # 5. Salvar config
            save_model_config()
        
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
   1. Conectamos ao Roboflow com a API key
   2. Carregamos o modelo YOLOv8 (versão 28)
   3. Verificamos as classes do modelo
   4. Testamos com uma imagem
   5. Salvamos configuração

🎯 PRÓXIMA ETAPA:
   Etapa 3: Processar uma simulação (SIM01)
   
   Execute: python src/roboflow_yolo8/03_inference.py
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