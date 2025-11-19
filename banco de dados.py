import chromadb
import pdfplumber
import os
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from chromadb.utils import embedding_functions

pdf_folder = filedialog.askdirectory(
    title="Selecione a pasta com os arquivos PDF",
    initialdir=os.path.expanduser("~")
)

db_folder = os.path.join(os.path.dirname(__file__), "Banco de dados")
COLLECTION_NAME = "PlanoManejo_Tijuca"
CHUNK_SIZE = 1000
OVERLAP = 200


def validar_ambiente():
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"Pasta não encontrada: {pdf_folder}")

    pdfs = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    if not pdfs:
        raise FileNotFoundError(f"Nenhum PDF encontrado em: {pdf_folder}")

    print(f"Encontrados {len(pdfs)} PDFs para processar\n")
    return pdfs


def extrair_texto_pdf(caminho_pdf):
    texto = ""
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            print(f"Processando {len(pdf.pages)} páginas...")
            for i, page in enumerate(pdf.pages, 1):
                try:
                    t = page.extract_text()
                    if t:
                        texto += t + "\n"

                    if i % 10 == 0:
                        print(f"  → Página {i}/{len(pdf.pages)}")

                except Exception as e:
                    print(f"Erro na página {i}: {e}")
                    continue

    except Exception as e:
        print(f"Erro ao abrir PDF: {e}")
        return ""

    return texto


def criar_chunks_com_overlap(texto, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    if not texto or not texto.strip():
        return []

    texto = ' '.join(texto.split())
    chunks = []
    inicio = 0

    while inicio < len(texto):
        fim = inicio + chunk_size

        if fim < len(texto):
            ultimo_espaco = texto.rfind(' ', inicio, fim)
            if ultimo_espaco > inicio:
                fim = ultimo_espaco

        chunk = texto[inicio:fim].strip()

        if chunk:
            chunks.append(chunk)

        inicio = fim - overlap if fim < len(texto) else fim

    return chunks


def limpar_colecao_existente(collection):
    try:
        results = collection.get()
        if results['ids']:
            collection.delete(ids=results['ids'])
            print(f"Removidos {len(results['ids'])} documentos antigos\n")
    except Exception as e:
        print(f"Aviso ao limpar coleção: {e}\n")


def processar_pdfs():
    pdfs = validar_ambiente()
    os.makedirs(db_folder, exist_ok=True)

    print("Inicializando ChromaDB...")
    client = chromadb.PersistentClient(path=db_folder)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    limpar_colecao_existente(collection)

    total_chunks = 0
    pdfs_processados = 0
    pdfs_com_erro = 0

    inicio_geral = datetime.now()

    for idx, nome_arquivo in enumerate(pdfs, 1):
        print(f"\n[{idx}/{len(pdfs)}]  {nome_arquivo}")
        caminho_pdf = os.path.join(pdf_folder, nome_arquivo)

        try:
            inicio = datetime.now()
            texto = extrair_texto_pdf(caminho_pdf)

            if not texto.strip():
                print("Nenhum texto extraído — pulando arquivo")
                pdfs_com_erro += 1
                continue

            chunks = criar_chunks_com_overlap(texto)

            if not chunks:
                print("Nenhum chunk criado — pulando arquivo")
                pdfs_com_erro += 1
                continue

            collection.add(
                ids=[str(uuid.uuid4()) for _ in chunks],
                documents=chunks,
                metadatas=[
                    {
                        "arquivo": nome_arquivo,
                        "parte": i + 1,
                        "total_partes": len(chunks),
                        "tamanho_original": len(texto),
                        "data_processamento": datetime.now().isoformat()
                    }
                    for i in range(len(chunks))
                ]
            )

            tempo_decorrido = (datetime.now() - inicio).total_seconds()
            print(f"{len(chunks)} chunks criados em {tempo_decorrido:.1f}s")

            total_chunks += len(chunks)
            pdfs_processados += 1

        except Exception as e:
            print(f"Erro ao processar: {e}")
            pdfs_com_erro += 1
            continue

    tempo_total = (datetime.now() - inicio_geral).total_seconds()

    print("\nRESUMO DO PROCESSAMENTO")
    print(f"PDFs processados com sucesso: {pdfs_processados}")
    print(f"PDFs com erro: {pdfs_com_erro}")
    print(f"Total de chunks criados: {total_chunks}")
    print(f"Tempo total: {tempo_total:.1f}s")
    print(f"Banco salvo em: {db_folder}")


if __name__ == "__main__":
    try:
        processar_pdfs()
    except Exception as e:
        print(f"\nErro na leitura: {e}")
        import traceback
        traceback.print_exc()
