import chromadb
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

DB_FOLDER = r"C:\chroma\banco"

COLLECTION_NAME = "PlanoManejo_Tijuca"
TOP_K = 5  

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("\n⚠️  AVISO: Verifique se a GROQ_API_KEY está configurada corretamente")
    exit(1)

# inicializa llm do groq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2000
)

# inicializa os embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


def inicializar_vectorstore():
    # usando langchain concecta o banco de dados
    if not os.path.exists(DB_FOLDER):
        raise FileNotFoundError(
            f"Banco de dados não encontrado em: {DB_FOLDER}\n"
            "Execute primeiro o script de processamento dos PDFs."
        )

    try:
        # concecta o chromadb
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_FOLDER
        )

        # verifica a quantidade de documentos que tem na coleção
        collection = vectorstore._collection
        _ = collection.count()  # só para validar que a coleção responde

        return vectorstore

    except Exception as e:
        raise Exception(f"Erro ao acessar coleção: {e}")


def criar_prompt_template():
    # template do agente com histórico

    template = """Você é um guia experiente do Parque Nacional da Tijuca, no Rio de Janeiro.

Use uma linguagem clara, organizada e acolhedora, mas sem exageros e sem vícios de linguagem.
Evite expressões como "olha só", "sabe o que é incrível?", "cara, isso é demais", "vem comigo que eu te conto"
ou qualquer outra muleta de linguagem repetitiva.

ESTILO DA RESPOSTA:
- Explique as informações de forma direta, mas simpática.
- Use parágrafos curtos, bem organizados, evitando repetir a mesma ideia várias vezes.
- Procure responder em até 2 ou 3 parágrafos, com no máximo 8 frases no total.
- Se usar emojis, use no máximo 2 por resposta, apenas ao final de frases ou parágrafos, nunca no meio da frase.
- Nunca comece a resposta com emoji; primeiro o texto, depois o emoji, se fizer sentido.

VOCÊ PODE FALAR SOBRE:
- Fauna (animais, comportamentos, onde é mais comum avistar)
- Flora (espécies nativas, árvores marcantes, importância ecológica)
- Trilhas e pontos turísticos (dificuldade, tempo médio, cuidados)
- História do parque (reflorestamento, contexto histórico)
- Dicas práticas (horários, segurança, o que levar, regras gerais)

REGRAS IMPORTANTES:
1. Baseie TUDO no contexto dos documentos fornecidos – você é um guia responsável.
2. Se não souber algo, responda de forma honesta: por exemplo,
   "Não encontrei essa informação nos documentos do parque, mas posso te explicar sobre outro aspecto relacionado."
3. Considere o histórico da conversa – você lembra do que já conversaram.
4. Quando fizer referência às fontes, fale de forma natural, como
   "De acordo com o plano de manejo do parque..." ou "Nos documentos oficiais do parque é citado que...".
5. Explique termos técnicos de forma simples, evitando jargões sem explicação.
6. NUNCA invente informações – preserve a credibilidade do guia.

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA DO VISITANTE: {question}"""

    return ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])


def criar_chain_rag(vectorstore):
    # cria a chain RAG

    from langchain_core.output_parsers import StrOutputParser

    # retriver
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    # prompt
    prompt = criar_prompt_template()

    # formatação de documentos
    def format_docs(docs):
        return "\n\n".join([
            f"[Fonte: {doc.metadata.get('arquivo', 'Desconhecido')} - Parte {doc.metadata.get('parte', '?')}]\n{doc.page_content}"
            for doc in docs
        ])

    # chain(LCEL) com histórico
    retrieval_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retrieval_chain, retriever


def processar_pergunta_langchain(chain_tuple, pergunta, chat_history=None):
    # usa a chain pra processar a pergunta

    chain, retriever = chain_tuple

    if chat_history is None:
        chat_history = []

    print(f"Pergunta: {pergunta}\n")

    try:
        # busca documentos relevantes
        documentos = retriever.invoke(pergunta)

        # executa a chain para gerar a resposta com histórico
        resposta = chain.invoke({
            "question": pergunta,
            "chat_history": chat_history
        })

        # resposta no terminal
        print("Resposta:\n")
        print(resposta)
        print()

        # atualização do histórico
        chat_history.append(HumanMessage(content=pergunta))
        chat_history.append(AIMessage(content=resposta))

        return resposta, documentos, chat_history

    except Exception as e:
        print(f"❌ Erro ao processar pergunta: {e}")
        import traceback
        traceback.print_exc()
        return None, [], chat_history


def modo_interativo():
    # modo interativo para fazer diferentes perguntas
    print("=" * 70)
    print("🌿 Guia virtual do Parque Nacional da Tijuca")
    print("=" * 70)
    print("Posso te ajudar com:\n")
    print("  • Informações detalhadas sobre fauna e flora")
    print("  • Informações detalhadas sobre regras do parque")

    try:
        vectorstore = inicializar_vectorstore()
        chain_tuple = criar_chain_rag(vectorstore)
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        return

    chat_history = []

    print("\n" + "=" * 70)
    print(" COMANDOS DISPONÍVEIS:")
    print("  • Digite sua pergunta normalmente")
    print("  • 'sair' - encerrar o programa")
    print("  • 'limpar' - resetar histórico da conversa")
    print("=" * 70 + "\n")

    while True:
        try:
            pergunta = input("🌿 Sua pergunta: ").strip()

            if not pergunta:
                continue

            if pergunta.lower() in ["sair", "Sair"]:
                print("\n" + "=" * 70)
                print("👋 Obrigado por usar o Guia do Parque Nacional da Tijuca!")
                print("   Aproveite sua aventura na natureza! 🌿🏞️")
                print("=" * 70 + "\n")
                break

            if pergunta.lower() in ["limpar", "Limpar"]:
                chat_history = []
                print("\n🗑️ Histórico de conversa limpo!\n")
                continue

            resposta, docs, chat_history = processar_pergunta_langchain(
                chain_tuple,
                pergunta,
                chat_history
            )

        except KeyboardInterrupt:
            print("\n\n👋 Até logo!\n")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    modo_interativo()
