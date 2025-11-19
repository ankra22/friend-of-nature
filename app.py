import io
from contextlib import redirect_stdout

import streamlit as st
from agente_orquestrador import OrquestradorAgentes


def extrair_resposta(saida: str) -> str:
    if not saida:
        return ""

    linhas = saida.splitlines()
    idx_resp = None

    for i, linha in enumerate(linhas):
        if "resposta" in linha.lower():
            idx_resp = i
            break

    if idx_resp is None:
        return saida.strip()

    resto = linhas[idx_resp + 1:]

    while resto and not resto[0].strip():
        resto = resto[1:]

    corte_idx = None
    for j, linha in enumerate(resto):
        l = linha.lower().strip()

        if "buscando clima atual" in l or "buscando previsão do tempo" in l:
            corte_idx = j
            break

        if "deseja visualizar algum mapa" in l:
            corte_idx = j
            break

        if l and set(l) == {"="} and len(l) >= 3:
            corte_idx = j
            break

    if corte_idx is not None:
        resto = resto[:corte_idx]

    return "\n".join(resto).strip()


st.set_page_config(
    page_title="Amigo da Natureza",
    page_icon="🌿",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    .stChatMessage p {
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 Amigo da Natureza")


def render_mensagem(conteudo: str):
    if not conteudo:
        return
    st.markdown(
        f"<div style='font-size:16px; line-height:1.6; white-space:pre-wrap;'>{conteudo}</div>",
        unsafe_allow_html=True,
    )


if "orquestrador" not in st.session_state:
    buf = io.StringIO()
    with redirect_stdout(buf):
        st.session_state["orquestrador"] = OrquestradorAgentes()
    st.session_state["init_log"] = buf.getvalue()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Olá! Eu sou o guia virtual do Parque Nacional da Tijuca.\n\n"
                "Pode perguntar sobre clima, trilhas, regras do parque ou curiosidades."
            ),
        }
    ]


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        render_mensagem(msg["content"])


pergunta = st.chat_input("Faça sua pergunta sobre o parque...")

if pergunta:
    st.session_state["messages"].append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        render_mensagem(pergunta)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            "<div style='font-size:16px; line-height:1.6;'>⌛ Consultando as informações do parque...</div>",
            unsafe_allow_html=True,
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            st.session_state["orquestrador"].processar_pergunta(pergunta)

        bruto = buf.getvalue().strip()
        resposta_texto = extrair_resposta(bruto)

        if not resposta_texto:
            resposta_texto = (
                "O sistema não retornou uma resposta clara.\n\n"
                "Verifique se os agentes foram inicializados corretamente."
            )

        placeholder.empty()
        render_mensagem(resposta_texto)

    st.session_state["messages"].append(
        {"role": "assistant", "content": resposta_texto}
    )
