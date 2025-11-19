import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import sys
import importlib.util


def importar_modulo(caminho_arquivo, nome_modulo):
    try:
        spec = importlib.util.spec_from_file_location(nome_modulo, caminho_arquivo)
        if spec and spec.loader:
            modulo = importlib.util.module_from_spec(spec)
            sys.modules[nome_modulo] = modulo
            spec.loader.exec_module(modulo)
            return modulo
        return None
    except Exception as e:
        print(f"Erro ao importar {nome_modulo}: {e}")
        return None


agente_clima = None
CLIMA_DISPONIVEL = False
for nome_possivel in ['agente_clima.py']:
    caminho = os.path.join(os.path.dirname(__file__), nome_possivel)
    if os.path.exists(caminho):
        agente_clima = importar_modulo(caminho, 'agente_clima')
        if agente_clima and hasattr(agente_clima, 'buscar_clima_atual'):
            CLIMA_DISPONIVEL = True
            print(f"Módulo de clima carregado: {nome_possivel}")
            break

agente_rag = None
RAG_DISPONIVEL = False
for nome_possivel in ['agente_geral.py']:
    caminho = os.path.join(os.path.dirname(__file__), nome_possivel)
    if os.path.exists(caminho):
        agente_rag = importar_modulo(caminho, 'agente_geral')
        if agente_rag and hasattr(agente_rag, 'inicializar_vectorstore'):
            RAG_DISPONIVEL = True
            print(f"Módulo RAG carregado: {nome_possivel}")
            break

agente_trilhas = None
TRILHAS_DISPONIVEL = False
for nome_possivel in ['agente_trilhas.py']:
    caminho = os.path.join(os.path.dirname(__file__), nome_possivel)
    if os.path.exists(caminho):
        agente_trilhas = importar_modulo(caminho, 'agente_trilhas')
        if agente_trilhas and hasattr(agente_trilhas, 'inicializar_vectorstores'):
            TRILHAS_DISPONIVEL = True
            print(f"Módulo de trilhas carregado: {nome_possivel}")
            break

print("\nStatus dos agentes:")
print(f"   Clima: {'Disponível' if CLIMA_DISPONIVEL else 'Indisponível'}")
print(f"   Informações Gerais: {'Disponível' if RAG_DISPONIVEL else 'Indisponível'}")
print(f"   Trilhas e Mapas: {'Disponível' if TRILHAS_DISPONIVEL else 'Indisponível'}")
print()

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("\nA variável GROQ_API_KEY não foi encontrada. Verifique o arquivo .env.")
    exit(1)

llm_classificador = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=100
)


class OrquestradorAgentes:
    def __init__(self):
        self.chat_history = []
        self.agentes_inicializados = {}

        print("\nInicializando agentes especializados...\n")

        if RAG_DISPONIVEL:
            try:
                vectorstore = agente_rag.inicializar_vectorstore()
                self.agentes_inicializados['rag'] = agente_rag.criar_chain_rag(vectorstore)
                print("Agente de Informações Gerais (RAG) inicializado")
            except Exception as e:
                print(f"Erro ao inicializar agente RAG: {e}")

        if TRILHAS_DISPONIVEL:
            try:
                vectorstore_texto, vectorstore_imagens = agente_trilhas.inicializar_vectorstores()
                self.agentes_inicializados['trilhas'] = agente_trilhas.criar_chain_rag(
                    vectorstore_texto,
                    vectorstore_imagens
                )
                print("Agente de Trilhas e Mapas inicializado")
            except Exception as e:
                print(f"Erro ao inicializar agente de trilhas: {e}")

        if CLIMA_DISPONIVEL:
            self.agentes_inicializados['clima'] = True
            print("Agente de Clima inicializado")

        print()

    def classificar_pergunta(self, pergunta: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um classificador de perguntas sobre o Parque Nacional da Tijuca.

Analise a pergunta do usuário e classifique em UMA das categorias:

1. clima
   - Perguntas sobre tempo, temperatura, condições climáticas
   - Previsão do tempo, chuva, sol, vento

2. trilhas
   - Perguntas sobre trilhas específicas
   - Mapas, rotas, caminhos, distância, dificuldade
   - Pontos turísticos e mirantes

3. geral
   - Fauna, flora, história do parque
   - Regras, normas e informações gerais

Responda apenas com uma palavra: clima, trilhas ou geral"""),
            ("human", "{pergunta}")
        ])

        try:
            chain = prompt | llm_classificador
            resposta = chain.invoke({"pergunta": pergunta})
            categoria = resposta.content.strip().lower()

            if categoria not in ['clima', 'trilhas', 'geral']:
                print(f"Categoria inválida '{categoria}', usando 'geral' como padrão")
                categoria = 'geral'

            return categoria

        except Exception as e:
            print(f"Erro na classificação: {e}. Usando 'geral' como padrão")
            return 'geral'

    def processar_pergunta(self, pergunta: str):
        print(f"\n{'=' * 70}")
        print(f"Pergunta: {pergunta}")
        print(f"{'=' * 70}\n")

        print("Analisando a pergunta...\n")
        categoria = self.classificar_pergunta(pergunta)

        emoji_categoria = {
            'clima': 'Clima e Previsão',
            'trilhas': 'Trilhas e Mapas',
            'geral': 'Informações Gerais'
        }

        print(f"Direcionando para: {emoji_categoria.get(categoria, 'Informações Gerais')}\n")
        print(f"{'=' * 70}\n")

        try:
            if categoria == 'clima' and 'clima' in self.agentes_inicializados:
                self._processar_clima(pergunta)

            elif categoria == 'trilhas' and 'trilhas' in self.agentes_inicializados:
                self._processar_trilhas(pergunta)

            elif categoria == 'geral' and 'rag' in self.agentes_inicializados:
                self._processar_geral(pergunta)

            else:
                print(f"Agente para '{categoria}' não está disponível no momento.")
                print("Tente outra pergunta ou verifique a configuração dos agentes.\n")

        except Exception as e:
            print(f"Erro ao processar pergunta: {e}")
            import traceback
            traceback.print_exc()

    def _processar_clima(self, pergunta: str):
        try:
            resultado = agente_clima.responder_clima(pergunta)
            print("\nRESPOSTA:\n")
            print(resultado)
            print()
        except Exception as e:
            print(f"Erro ao buscar clima: {e}\n")

    def _processar_trilhas(self, pergunta: str):
        chain_tuple = self.agentes_inicializados['trilhas']

        try:
            resposta, docs, mapas, self.chat_history = agente_trilhas.processar_pergunta_com_mapas(
                chain_tuple,
                pergunta,
                self.chat_history
            )
        except Exception as e:
            print(f"Erro no agente de trilhas: {e}\n")

    def _processar_geral(self, pergunta: str):
        chain_tuple = self.agentes_inicializados['rag']

        try:
            resposta, docs, self.chat_history = agente_rag.processar_pergunta_langchain(
                chain_tuple,
                pergunta,
                self.chat_history
            )
        except Exception as e:
            print(f"Erro no agente geral: {e}\n")

    def limpar_historico(self):
        self.chat_history = []
        print("\nHistórico de conversa limpo.\n")


def modo_interativo():
    print("\n" + "=" * 70)
    print("ASSISTENTE DO PARQUE NACIONAL DA TIJUCA")
    print("=" * 70)
    print("\nPosso ajudar com:")
    print("  • Clima e previsão do tempo")
    print("  • Trilhas, mapas e rotas")
    print("  • Fauna, flora e informações gerais")

    orquestrador = OrquestradorAgentes()

    print("\n" + "=" * 70)
    print("COMANDOS:")
    print("  • Digite sua pergunta normalmente")
    print("  • 'sair'  - encerrar o programa")
    print("  • 'limpar' - limpar histórico")
    print("  • 'ajuda' - ver exemplos de perguntas")
    print("=" * 70 + "\n")

    while True:
        try:
            entrada = input("Pergunta: ").strip()

            if not entrada:
                continue

            if entrada.lower() in ['sair', 'exit', 'quit']:
                print("\n" + "=" * 70)
                print("Encerrando o assistente. Boa visita ao parque!")
                print("=" * 70 + "\n")
                break

            if entrada.lower() in ['limpar', 'clear', 'reset']:
                orquestrador.limpar_historico()
                continue

            if entrada.lower() in ['ajuda', 'help', 'exemplos']:
                print("\n" + "=" * 70)
                print("EXEMPLOS DE PERGUNTAS:")
                print("=" * 70)
                print("\nCLIMA:")
                print("  • Como está o tempo agora?")
                print("  • Vai chover hoje?")
                print("  • Qual a previsão para os próximos dias?")
                print("\nTRILHAS:")
                print("  • Como faço para chegar no Pico da Tijuca?")
                print("  • Qual a dificuldade da trilha Cascatinha?")
                print("  • Mostre o mapa da trilha do Horto")
                print("\nINFORMAÇÕES GERAIS:")
                print("  • Quais animais posso ver no parque?")
                print("  • O que é permitido fazer?")
                print("  • Conte sobre a história do parque")
                print("=" * 70 + "\n")
                continue

            orquestrador.processar_pergunta(entrada)

        except KeyboardInterrupt:
            print("\n\nEncerrando. Até logo!\n")
            break
        except Exception as e:
            print(f"\nErro: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    modo_interativo()
