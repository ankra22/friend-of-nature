#  Assistente Virtual do Parque Nacional da Tijuca

Um sistema inteligente de assistência para visitantes do Parque Nacional da Tijuca, utilizando RAG (Retrieval-Augmented Generation) e múltiplos agentes especializados.

##  Sobre o Projeto

Este assistente virtual combina processamento de linguagem natural, busca vetorial e APIs externas para fornecer informações precisas sobre:

- 🌦️ **Clima e Previsão do Tempo** - Condições meteorológicas em tempo real
- 🗺️ **Trilhas e Mapas** - Informações detalhadas sobre rotas, com visualização de mapas extraídos de PDFs
- 🌿 **Informações Gerais** - Fauna, flora, história e regras do parque

##  Arquitetura

O sistema utiliza uma arquitetura modular com orquestração inteligente:

```
                 ┌─────────────────────────────────────┐
                 │   Orquestrador Principal (LLM)     │
                 │   Classifica e roteia perguntas    │
                 └───────────────┬─────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
    ┌─────────┐          ┌──────────────┐      ┌──────────────┐
    │ Agente  │          │   Agente     │      │   Agente     │
    │ Clima   │          │   RAG Geral  │      │   Trilhas    │
    │ (API)   │          │  (ChromaDB)  │      │  (ChromaDB)  │
    └─────────┘          └──────────────┘      └──────────────┘
```

##  Funcionalidades

### Agente de Clima
- Consulta condições meteorológicas em tempo real via WeatherAPI
- Previsão do tempo para os próximos 3 dias
- Recomendações contextualizadas para atividades no parque

### Agente de Informações Gerais
- Base de conhecimento construída a partir do Plano de Manejo do parque
- Responde sobre fauna, flora, história e regras
- Mantém contexto de conversação

### Agente de Trilhas e Mapas
- Informações detalhadas sobre trilhas específicas
- Extração e visualização de mapas de PDFs
- Busca vetorial em textos e imagens
- Recomendações de segurança e dificuldade

##  Tecnologias Utilizadas

- **LangChain** - Framework para aplicações com LLMs
- **ChromaDB** - Banco de dados vetorial para RAG
- **Groq API** - Inferência de LLMs (Llama 3.3 70B)
- **HuggingFace Embeddings** - Geração de embeddings (all-MiniLM-L6-v2)
- **WeatherAPI** - Dados meteorológicos
- **PyMuPDF** - Extração de texto e imagens de PDFs
- **pdfplumber** - Processamento de PDFs
- **Streamlit** - Interface web interativa
- **PIL/Pillow** - Processamento de imagens

##  Instalação

### Pré-requisitos

- Python 3.8+

### Passo a Passo

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/assistente-parque-tijuca.git
cd assistente-parque-tijuca
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as bibliotecas:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:

Crie um arquivo `.env` na raiz do projeto:
```env
GROQ_API_KEY=sua_chave_groq_aqui
WEATHER_API_KEY=sua_chave_weatherapi_aqui
```

**Obtendo as chaves:**
- Groq API: https://console.groq.com
- Weather API: https://www.weatherapi.com

##  Preparação dos Dados

### 1. Processar PDFs de Informações Gerais

```bash
python processa_pdf.py
```

Selecione a pasta contendo os PDFs do Plano de Manejo do parque. O script irá:
- Extrair texto de cada PDF
- Dividir em chunks com overlap
- Gerar embeddings e armazenar no ChromaDB

### 2. Processar PDFs com Mapas de Trilhas

```bash
python processa_pdf_imagens.py
```

Selecione a pasta com PDFs contendo mapas. O script irá:
- Extrair imagens embutidas
- Renderizar páginas como imagens de alta resolução
- Gerar embeddings visuais e armazenar no ChromaDB

##  Uso

### Interface de Linha de Comando

```bash
python agente_orquestrador.py
```

Comandos disponíveis:
- Digite sua pergunta naturalmente
- `limpar` - Reseta o histórico de conversa
- `ajuda` - Mostra exemplos de perguntas
- `sair` - Encerra o programa

### Interface Web (Streamlit)

```bash
streamlit run interface_streamlit.py
```

Acesse em seu navegador: `http://localhost:8501`

## 💬 Exemplos de Uso

**Clima:**
```
Usuário: Como está o tempo agora?
Assistente: Neste momento, o Parque Nacional da Tijuca está com 
céu parcialmente nublado. A temperatura gira em torno de 24.5 °C...
```

**Trilhas:**
```
Usuário: Como faço para chegar no Pico da Tijuca?
Assistente: A Trilha do Pico da Tijuca é uma caminhada moderada a 
intensa, recomendada para pessoas com preparo físico razoável...
[Opção de visualizar mapa]
```

**Informações Gerais:**
```
Usuário: Quais animais posso ver no parque?
Assistente: O Parque Nacional da Tijuca abriga uma rica fauna, 
incluindo macacos-prego, preguiças, quatis...
```

## 📁 Estrutura do Projeto

```
assistente-parque-tijuca/
├── agente_orquestrador.py      # Orquestrador principal
├── agente_clima.py             # Agente de clima
├── agente_geral.py             # Agente RAG geral
├── agente_trilhas.py           # Agente de trilhas
├── processa_pdf.py             # Processa PDFs de texto
├── processa_pdf_imagens.py     # Processa PDFs com imagens
├── interface_streamlit.py      # Interface web
├── requirements.txt            # Dependências
├── .env.example                # Exemplo de variáveis de ambiente
├── Banco de dados/             # ChromaDB (textos)
├── Banco de dados imagens trilhas/  # ChromaDB (imagens)
└── README.md
```

## 🔧 Configuração Avançada

### Ajuste de Parâmetros RAG

Em `agente_geral.py` e `agente_trilhas.py`:

```python
TOP_K = 5              # Número de chunks recuperados
CHUNK_SIZE = 1000      # Tamanho dos chunks (caracteres)
OVERLAP = 200          # Overlap entre chunks
```

### Modelos LLM

Para alterar o modelo do Groq:

```python
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",  # Altere aqui
    temperature=0.3,
    max_tokens=2000
)
```

Modelos disponíveis:
- `llama-3.3-70b-versatile`
- `mixtral-8x7b-32768`
- `gemma-7b-it`

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🙏 Agradecimentos

- ICMBio - Instituto Chico Mendes de Conservação da Biodiversidade
- Parque Nacional da Tijuca
- Comunidade de desenvolvedores de IA e RAG
