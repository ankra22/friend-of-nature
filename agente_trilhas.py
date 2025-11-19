import chromadb
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from datetime import datetime

load_dotenv()

DB_FOLDER_TEXTO = r"C:\chroma\banco"
DB_FOLDER_IMAGENS = os.path.join(os.path.dirname(__file__), "Banco de dados imagens PDF")
COLLECTION_NAME_TEXTO = "PlanoManejo_Tijuca"
COLLECTION_NAME_IMAGENS = "Imagens_PDF_Collection"
TOP_K_TEXTO = 5
TOP_K_IMAGENS = 3

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("\nA variável GROQ_API_KEY não foi encontrada no arquivo .env.")
    exit(1)

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2000
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


def inicializar_vectorstores():
    if not os.path.exists(DB_FOLDER_TEXTO):
        raise FileNotFoundError(
            f"Banco de dados de texto não encontrado em: {DB_FOLDER_TEXTO}\n"
            "Execute primeiro o processamento dos PDFs."
        )

    vectorstore_texto = Chroma(
        collection_name=COLLECTION_NAME_TEXTO,
        embedding_function=embeddings,
        persist_directory=DB_FOLDER_TEXTO
    )

    vectorstore_imagens = None
    if os.path.exists(DB_FOLDER_IMAGENS):
        try:
            client = chromadb.PersistentClient(path=DB_FOLDER_IMAGENS)
            collection_imagens = client.get_collection(name=COLLECTION_NAME_IMAGENS)
            vectorstore_imagens = collection_imagens
            print(f"Banco de imagens carregado: {collection_imagens.count()} imagens disponíveis\n")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar o banco de imagens: {e}\n")
    else:
        print(f"Aviso: Banco de imagens não encontrado em: {DB_FOLDER_IMAGENS}\n")

    return vectorstore_texto, vectorstore_imagens


def buscar_mapas_relevantes(vectorstore_imagens, query_text, top_k=TOP_K_IMAGENS):
    if vectorstore_imagens is None:
        return []

    try:
        keywords = query_text.lower().split()

        trilha_keywords = [
            'cascatinha', 'taunay', 'pico', 'tijuca', 'mirante',
            'mayrink', 'excelsior', 'imperador', 'conde', 'estrada',
            'caminho', 'trilha', 'vale', 'floresta', 'cachoeira'
        ]

        results = vectorstore_imagens.get(include=['metadatas', 'documents'])

        if not results['ids']:
            return []

        mapas_pontuados = []

        for doc_id, metadata, document in zip(
            results['ids'],
            results['metadatas'],
            results['documents']
        ):
            arquivo = metadata.get('arquivo_pdf', '').lower()
            caminho = metadata.get('caminho_relativo', '').lower()
            doc_text = document.lower() if document else ''

            score = 0

            for keyword in keywords:
                if len(keyword) > 2:
                    if keyword in arquivo:
                        score += 3
                    if keyword in caminho:
                        score += 2
                    if keyword in doc_text:
                        score += 1

            for trilha_kw in trilha_keywords:
                if trilha_kw in arquivo or trilha_kw in caminho:
                    score += 1

            if score > 0:
                mapas_pontuados.append({
                    'id': doc_id,
                    'arquivo': metadata.get('arquivo_pdf', 'Desconhecido'),
                    'pagina': metadata.get('pagina', '?'),
                    'caminho': metadata.get('caminho_relativo', ''),
                    'dimensoes': metadata.get('dimensoes', ''),
                    'relevancia': score,
                    'metodo': metadata.get('metodo_extracao', 'desconhecido')
                })

        mapas_pontuados.sort(key=lambda x: x['relevancia'], reverse=True)

        return mapas_pontuados[:top_k]

    except Exception as e:
        print(f"Erro ao buscar mapas: {e}")
        import traceback
        traceback.print_exc()
        return []


def recuperar_imagem_do_banco(vectorstore_imagens, doc_id):
    try:
        result = vectorstore_imagens.get(ids=[doc_id], include=['embeddings', 'metadatas'])

        if result is None or 'embeddings' not in result:
            return None, None

        embeddings_list = result['embeddings']
        if not embeddings_list:
            return None, None

        embedding = embeddings_list[0]
        if embedding is None:
            return None, None

        embedding_array = np.array(embedding)
        if embedding_array.size == 0:
            return None, None

        metadata = result['metadatas'][0] if result.get('metadatas') else {}

        if embedding_array.size == 602112:
            img_array = embedding_array.reshape(448, 448, 3)
        elif embedding_array.size == 150528:
            img_array = embedding_array.reshape(224, 224, 3)
        else:
            total_pixels = embedding_array.size // 3
            lado = int(np.sqrt(total_pixels))
            if lado * lado * 3 == embedding_array.size:
                img_array = embedding_array.reshape(lado, lado, 3)
            else:
                return None, None

        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='RGB')

        return img, metadata

    except Exception as e:
        print(f"Erro ao recuperar imagem: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def exibir_mapa_do_banco(vectorstore_imagens, mapa_info):
    try:
        img, metadata = recuperar_imagem_do_banco(vectorstore_imagens, mapa_info['id'])

        if img is None:
            print("Não foi possível recuperar a imagem")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_base = mapa_info['arquivo'].replace('.pdf', '').replace(' ', '_')
        filename = f"mapa_{nome_base}_p{mapa_info['pagina']}_{timestamp}.png"
        output_path = os.path.join(os.path.dirname(__file__), filename)

        if img.size[0] == 448:
            scale_factor = 3
        elif img.size[0] == 224:
            scale_factor = 6
        else:
            scale_factor = 3

        new_size = (img.size[0] * scale_factor, img.size[1] * scale_factor)
        img_display = img.resize(new_size, Image.Resampling.LANCZOS)
        img_display.save(output_path, quality=100, optimize=False)

        try:
            img_display.show()
        except:
            pass

        return output_path

    except Exception as e:
        print(f"Erro ao exibir mapa: {e}")
        import traceback
        traceback.print_exc()
        return None


def criar_prompt_template():
    template = """Você é um guia especializado em trilhas do Parque Nacional da Tijuca.

Use uma linguagem clara, direta e objetiva. Evite exageros ou repetições desnecessárias.

ESTILO:
- Comece resumindo o tipo de trilha, o esforço envolvido e o perfil ideal de visitante.
- Se houver dados nos documentos, apresente um pequeno bloco com distância, tempo médio, dificuldade e cuidados.
- Se alguma informação não estiver disponível, apenas diga isso de forma natural, sem inventar valores.
- Use apenas uma lista simples quando fizer sentido.
- Não utilize emojis.

TÓPICOS:
- Informações sobre trilhas específicas.
- Grau de dificuldade.
- Tempo médio.
- Cuidados e recomendações práticas.
- Riscos relevantes descritos nos documentos.

REGRAS:
1. Utilize apenas o que estiver nos documentos.
2. Se não houver informações suficientes, informe isso de maneira simples.
3. Leve em conta o histórico da conversa.
4. Cite documentos de modo natural quando necessário.
5. Explique termos técnicos de maneira acessível.

CONTEXTO:
{context}

PERGUNTA: {question}"""
    return ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])


def criar_chain_rag(vectorstore_texto, vectorstore_imagens):
    from langchain_core.output_parsers import StrOutputParser

    retriever = vectorstore_texto.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_TEXTO}
    )

    prompt = criar_prompt_template()

    def format_docs(docs):
        return "\n\n".join([
            f"[Fonte: {doc.metadata.get('arquivo', 'Desconhecido')} - Parte {doc.metadata.get('parte', '?')}]\n{doc.page_content}"
            for doc in docs
        ])

    def buscar_info_mapas(query):
        if vectorstore_imagens is None:
            return "Nenhum mapa disponível no momento."

        mapas = buscar_mapas_relevantes(vectorstore_imagens, query)
        if not mapas:
            return "Nenhum mapa encontrado para esta consulta."

        info = "Mapas disponíveis:\n"
        for i, mapa in enumerate(mapas, 1):
            info += f"{i}. {mapa['arquivo']} (Página {mapa['pagina']}) - Score: {mapa['relevancia']}\n"
        return info

    retrieval_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
            "mapas_info": lambda x: buscar_info_mapas(x["question"])
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retrieval_chain, retriever, vectorstore_imagens


def processar_pergunta_com_mapas(chain_tuple, pergunta, chat_history=None):
    chain, retriever, vectorstore_imagens = chain_tuple

    if chat_history is None:
        chat_history = []

    try:
        documentos = retriever.invoke(pergunta)
        mapas = buscar_mapas_relevantes(vectorstore_imagens, pergunta) if vectorstore_imagens else []

        if documentos:
            fontes = {doc.metadata['arquivo'] for doc in documentos if 'arquivo' in doc.metadata}
            if fontes:
                print(f"Fontes consultadas: {', '.join(fontes)}")

        if mapas:
            print(f"Mapas encontrados: {len(mapas)}")
            for i, mapa in enumerate(mapas, 1):
                print(f"{i}. {mapa['arquivo']} (Página {mapa['pagina']}) - Score {mapa['relevancia']}")

        resposta = chain.invoke({
            "question": pergunta,
            "chat_history": chat_history
        })

        print("\nResposta:")
        print(resposta)
        print()

        if mapas:
            print("Deseja visualizar algum mapa? (número ou 'não')")
            for i, mapa in enumerate(mapas, 1):
                print(f"[{i}] {mapa['arquivo']} - Página {mapa['pagina']}")

            escolha = input("Escolha: ").strip()
            if escolha.isdigit() and 1 <= int(escolha) <= len(mapas):
                mapa_escolhido = mapas[int(escolha) - 1]
                exibir_mapa_do_banco(vectorstore_imagens, mapa_escolhido)

        chat_history.append(HumanMessage(content=pergunta))
        chat_history.append(AIMessage(content=resposta))

        return resposta, documentos, mapas, chat_history

    except Exception as e:
        print(f"Erro ao processar pergunta: {e}")
        import traceback
        traceback.print_exc()
        return None, [], [], chat_history


def modo_interativo():
    print("\nGuia de Trilhas do Parque Nacional da Tijuca")
    print("Informações sobre trilhas, mapas, pontos de interesse e recomendações práticas.")

    try:
        vectorstore_texto, vectorstore_imagens = inicializar_vectorstores()
        chain_tuple = criar_chain_rag(vectorstore_texto, vectorstore_imagens)
    except Exception as e:
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return

    chat_history = []

    print("\nComandos:")
    print("  • Pergunte normalmente")
    print("  • 'sair' para encerrar")
    print("  • 'limpar' para apagar o histórico")

    while True:
        try:
            pergunta = input("Pergunta: ").strip()

            if not pergunta:
                continue

            if pergunta.lower() in ['sair', 'exit', 'quit']:
                print("\nEncerrando o guia. Boa caminhada!")
                break

            if pergunta.lower() in ['limpar', 'clear', 'reset']:
                chat_history = []
                print("Histórico limpo.\n")
                continue

            processar_pergunta_com_mapas(chain_tuple, pergunta, chat_history)

        except KeyboardInterrupt:
            print("\nAté logo.")
            break
        except Exception as e:
            print(f"\nErro: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    modo_interativo()
