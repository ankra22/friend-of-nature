import chromadb
import os
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from PIL import Image
import numpy as np
from chromadb.utils import embedding_functions
import base64
from io import BytesIO
import fitz

pdf_folder = filedialog.askdirectory(
    title="Selecione a pasta com os PDFs (imagens)",
    initialdir=os.path.expanduser("~")
)

db_folder = os.path.join(os.path.dirname(__file__), "Banco de dados imagens trilhas")
COLLECTION_NAME = "Imagens_PDF_Collection"


def validar_ambiente():
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"Pasta não encontrada: {pdf_folder}")

    pdfs = []
    pastas_encontradas = 0

    for root, dirs, files in os.walk(pdf_folder):
        pastas_encontradas += 1
        for f in files:
            if f.lower().endswith('.pdf'):
                caminho = os.path.join(root, f)
                rel = os.path.relpath(caminho, pdf_folder)
                pdfs.append((caminho, rel))

    if not pdfs:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {pdf_folder} ou subpastas")

    print(f"Encontrados {len(pdfs)} PDFs em {pastas_encontradas} pasta(s)\n")
    return pdfs


def extrair_imagens_pdf(caminho_pdf):
    imagens = []
    try:
        doc = fitz.open(caminho_pdf)
        print(f"     {len(doc)} página(s)")

        total_embutidas = 0

        for pagina_num in range(len(doc)):
            pagina = doc[pagina_num]
            lista = pagina.get_images(full=True)

            for img_index, img_info in enumerate(lista):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]

                    img = Image.open(BytesIO(image_bytes))

                    if img.size[0] >= 100 and img.size[1] >= 100:
                        info = {
                            'pagina': pagina_num + 1,
                            'indice_imagem': img_index + 1,
                            'dimensoes_originais': img.size,
                            'modo': img.mode,
                            'formato': ext,
                            'tamanho_bytes': len(image_bytes),
                            'metodo': 'embutida'
                        }

                        imagens.append((img, info))
                        total_embutidas += 1
                except:
                    continue

        if total_embutidas > 0:
            print(f"       {total_embutidas} imagem(ns) embutida(s) extraída(s)")

        print("      Renderizando páginas...")

        for pagina_num in range(len(doc)):
            pagina = doc[pagina_num]
            matriz = fitz.Matrix(300 / 72, 300 / 72)
            pix = pagina.get_pixmap(matrix=matriz, alpha=False)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            info = {
                'pagina': pagina_num + 1,
                'indice_imagem': 1,
                'dimensoes_originais': img.size,
                'modo': img.mode,
                'formato': 'rendered_page',
                'tamanho_bytes': len(pix.samples),
                'metodo': 'renderizada',
                'dpi': 300
            }

            imagens.append((img, info))
            print(f"      → Página {pagina_num + 1}: {img.size[0]}x{img.size[1]} pixels")

        doc.close()
        return imagens

    except Exception as e:
        print(f"    Erro ao processar PDF: {e}")
        return []


def extrair_features_imagem(img):
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized).astype('float32') / 255.0
        embedding = img_array.flatten().tolist()

        return embedding

    except Exception as e:
        print(f"    Erro ao processar imagem: {e}")
        return None


def imagem_para_base64(img, max_size=(800, 800)):
    try:
        copia = img.copy()
        copia.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        copia.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

    except:
        return None


def limpar_colecao_existente(collection):
    try:
        results = collection.get()
        if results['ids']:
            collection.delete(ids=results['ids'])
            print(f"    Removidos {len(results['ids'])} itens antigos\n")
    except Exception as e:
        print(f"    Aviso ao limpar coleção: {e}\n")


def processar_pdfs():
    pdfs = validar_ambiente()

    os.makedirs(db_folder, exist_ok=True)

    print("Inicializando ChromaDB...")
    client = chromadb.PersistentClient(path=db_folder)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Imagens extraídas de PDFs"}
    )

    limpar_colecao_existente(collection)

    total_imagens = 0
    pdfs_processados = 0
    pdfs_com_erro = 0
    inicio_geral = datetime.now()

    for idx, (caminho_pdf, caminho_relativo) in enumerate(pdfs, 1):
        nome_pdf = os.path.basename(caminho_pdf)
        pasta_rel = os.path.dirname(caminho_relativo)

        print(f"\n[{idx}/{len(pdfs)}]  {caminho_relativo}")
        if pasta_rel:
            print(f"     Pasta: {pasta_rel}")

        try:
            inicio = datetime.now()
            imagens_extraidas = extrair_imagens_pdf(caminho_pdf)

            if not imagens_extraidas:
                print("    Nenhuma imagem encontrada — pulando arquivo")
                pdfs_com_erro += 1
                continue

            imagens_ok = 0

            for img, info in imagens_extraidas:
                try:
                    embedding = extrair_features_imagem(img)
                    if embedding is None:
                        continue

                    img_base64 = imagem_para_base64(img)

                    metadata = {
                        "arquivo_pdf": nome_pdf,
                        "caminho_relativo": caminho_relativo,
                        "pasta": pasta_rel if pasta_rel else "raiz",
                        "pagina": info['pagina'],
                        "indice_imagem": info['indice_imagem'],
                        "dimensoes": f"{info['dimensoes_originais'][0]}x{info['dimensoes_originais'][1]}",
                        "formato": info['formato'],
                        "tamanho_kb": round(info['tamanho_bytes'] / 1024, 2),
                        "data_processamento": datetime.now().isoformat(),
                    }

                    if img_base64:
                        metadata["preview_base64"] = img_base64[:1000]

                    doc_id = f"{caminho_relativo}_p{info['pagina']}_i{info['indice_imagem']}"

                    collection.add(
                        ids=[str(uuid.uuid4())],
                        embeddings=[embedding],
                        documents=[doc_id],
                        metadatas=[metadata]
                    )

                    imagens_ok += 1
                    total_imagens += 1

                except Exception as e:
                    print(f"        Erro ao processar imagem: {e}")
                    continue

            tempo_decorrido = (datetime.now() - inicio).total_seconds()
            print(f"     {imagens_ok} imagem(ns) processada(s) em {tempo_decorrido:.2f}s")

            pdfs_processados += 1

        except Exception as e:
            print(f"    Erro ao processar PDF: {e}")
            pdfs_com_erro += 1
            continue

    tempo_total = (datetime.now() - inicio_geral).total_seconds()

    print("\n" + "=" * 60)
    print("RESUMO DO PROCESSAMENTO")
    print("=" * 60)
    print(f" PDFs processados: {pdfs_processados}")
    print(f" PDFs com erro: {pdfs_com_erro}")
    print(f" Total de imagens extraídas: {total_imagens}")
    print(f" Tempo total: {tempo_total:.1f}s")
    print(f" Banco salvo em: {db_folder}")
    print(f" Coleção: {COLLECTION_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        processar_pdfs()
    except Exception as e:
        print(f"\nErro na execução: {e}")
        import traceback
        traceback.print_exc()
