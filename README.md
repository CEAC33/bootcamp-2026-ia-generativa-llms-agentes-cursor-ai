# bootcamp-2026-ia-generativa-llms-agentes-cursor-ai
Bootcamp 2026 IA Generativa, LLM Apps, Agentes IA, Cursor AI

### Arquitectura Básica - Toy Demo
- LLM
- Orchestration Framework
- Vector Database
- Toy UI (Streamlit)

### Arquitectura Avanzada - Profesional
- External APIs
- LLM
- AWS S3, cloud storage
- Backend Server - Render.com (fastAPI, orchestration frm, database)
- Frontend Server - Vercel.com (next.js)

### Componentes de la Técnica RAG
- Embeddings
- Vector Database

### Tips Avanzados RAG
- Splitters
- Retrieval QA Chain
- Vector Database
- Debugging

### RAG vs CAG

**Generación Aumentada por Recuperación (RAG)**

RAG combina las capacidades de recuperación de información y generación de texto para mejorar las respuestas de los modelos de lenguaje. Cuando se presenta una consulta, el sistema recupera documentos relevantes desde una base de datos externa y utiliza esa información para generar una respuesta más informada. Este enfoque permite a los modelos acceder a información actualizada y específica del dominio sin necesidad de ser reentrenados continuamente. En algunos casos, RAG puede introducir complejidades como mayor latencia debido a la recuperación en tiempo real y posibles errores en la selección de documentos.

**Generación Aumentada por Caché (CAG)**

La Generación Aumentada por Caché (CAG) todavía está en una etapa muy inicial. Algunos dicen que, en un futuro cercano, podrá abordar algunas de las limitaciones inherentes de RAG precargando el conocimiento relevante directamente en la ventana de contexto extendida del modelo de lenguaje. Este método implica curar un conjunto de datos estático, precargarlo en el contexto del modelo y utilizar el almacenamiento en caché del estado de inferencia para guardar estados intermedios. Al hacerlo, CAG elimina la necesidad de recuperación en tiempo real, reduciendo así la latencia y simplificando la arquitectura del sistema. Este enfoque es especialmente útil en escenarios donde la base de conocimiento es estable y puede caber dentro del contexto del modelo.

### ¿Cómo seleccionar la Orchestration Framework?
- **LangChain (LangChain, LangSmith, LangGraph) - más popular, inicialmente más para Toy Demos, se ha hecho más robusta**
- LlamaIndex - 2da herramienta más popular, rag profesionales, menos generalista; hacer menos cosas, pero hacerlas mejor
- API de Open AI - en caso de que solo se quiera usar las LLM de OpenAI, pero con mucha opacidad

LLMS: Poder Limitado

Muchas APIs:
- OpenAI
- Llama3
- Mistral
- Etc.

Problemas
- Limited Context Window
- Outdated Knowledge Base
- Data privacy
- Cost
- No connection with 3rd party tools
- Etc.

LLM  on Steroids
- Data In (txt, pdf, csv, sql, xls, html, png, etc)
- Tools (apps, rag, agents, APIs)
- Actions (format, manipulate, transform, store, memory modules, chains program, 3rd party tools, etc.)
- LLMs (one, many, closed source, open source, 3rd party tools)
- Data Out (tools, actions, memory)

Por qué elegimos:
- LangChain: Orchestration Framework
- LangSmith LLMOps
- Streamlit: POC Temporary UI
- LangServe, Next+FastAPI: Deployment
- LangGraph: Agents Framework
- CrewAI: Multi-Agents Framework

LCHAIN TEAM ANALYSIS
- Phase #1: knowledge-based tools
  - Answer questions, create content
- Phase #2: action-oriented agents
  - Take decisions, collaborate with humans and agents, and learn

### Poetry installation

```
brew install pipx
```

```
pipx install poetry==1.8.2
```

Check Versions
- Mira el archivo pyproject.toml

Python Version: 3.11.4

```
poetry install
```

```
poetry shell
```

```
jupyter lab
```

## LangChain Basics

- Data In (prompt, prompt template, chain)
- LLM (closed source, open source)
- Data Out (output parser)

### Connect with an LLM

Intro
- Input: the prompt we send to the LLM.
- Output: the response from the LLM.
- We can switch LLMs and use several different LLMs.

LangChain divides LLMs in two types
1.- LLM Model: text-completion model.
2.- Chat Model: converses with a sequence of messages and can have a particular role defined (system prompt). This type has become the most used in LangChain.

See the differences
- Even when sometimes the LangChain documentation can be confusing about it, the fact is that text-completion models and Chat models are both LLMs.
- But, as you can see in this playground, they have some significant differences. See that the chat models in LangChain have system messages, human messages (called "user messages" by OpenAI) and AI messages (called "Assitant Messages" by OpenAI).
- Since the launch of chatGPT, the Chat Model is the most popular LLM type and is used in most LLM apps.

Chat Models
- System - rol que queremos que juegue nuestro chatbot
- User/Human messages - mensajes que enviamos
- Assistant/AI messages - mensajes del chatbot

LLMs supported by Langchain: https://python.langchain.com/v0.1/docs/integrations/llms/

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import OpenAI

llmModel = OpenAI()

print("\n----------\n")

response = llmModel.invoke(
    "Tell me one fun fact about the Kennedy family."
)

print("Tell me one fun fact about the Kennedy family:")
print(response)

print("\n----------\n")

print("Streaming:")

for chunk in llmModel.stream(
    "Tell me one fun fact about the Kennedy family."
):
    print(chunk, end="", flush=True)
    
print("\n----------\n")

creativeLlmModel = OpenAI(temperature=0.9)

response = llmModel.invoke(
    "Write a short 5 line poem about JFK"
)

print("Write a short 5 line poem about JFK:")
print(response)

print("\n----------\n")

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

messages = [
    ("system", "You are an historian expert in the Kennedy family."),
    ("human", "Tell me one curious thing about JFK."),
]
response = chatModel.invoke(messages)

print("Tell me one curious thing about JFK:")
print(response.content)

print("\n----------\n")

print("Streaming:")

for chunk in chatModel.stream(messages):
    print(chunk.content, end="", flush=True)
    
print("\n----------\n")
```

```
response.content
```
"One curious thing about JFK is that he was a collector of unique and eclectic items. He had a fascination with history and art, and his collection included everything from ship models and scrimshaw to original manuscripts and rare books. JFK's love of collecting even extended to quirky items like coconut shells carved with faces that he displayed in the Oval Office. This hobby provided a glimpse into his personal interests and served as a reflection of his intellectual curiosity."


```
response.response_metadata
```
{'token_usage': {'completion_tokens': 87,
  'prompt_tokens': 29,
  'total_tokens': 116},
 'model_name': 'gpt-3.5-turbo-0125',
 'system_fingerprint': None,
 'finish_reason': 'stop',
 'logprobs': None}

```
response.schema()
```
{'title': 'AIMessage',
 'description': 'Message from an AI.\n\nAIMessage is returned from a chat model as a response to a prompt.\n\nThis message represents the output of the model and consists of both\nthe raw output as returned by the model together standardized fields\n(e.g., tool calls, usage metadata) added by the LangChain framework.',
 'type': 'object',
 'properties': {'content': {'title': 'Content',
   'anyOf': [{'type': 'string'},
    {'type': 'array',
     'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]},
  'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
  'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
  'type': {'title': 'Type', 'default': 'ai', 'enum': ['ai'], 'type': 'string'},
  'name': {'title': 'Name', 'type': 'string'},
  'id': {'title': 'Id', 'type': 'string'},
  'example': {'title': 'Example', 'default': False, 'type': 'boolean'},
  'tool_calls': {'title': 'Tool Calls',
   'default': [],
   'type': 'array',
   'items': {'$ref': '#/definitions/ToolCall'}},
  'invalid_tool_calls': {'title': 'Invalid Tool Calls',
   'default': [],
   'type': 'array',
   'items': {'$ref': '#/definitions/InvalidToolCall'}},
  'usage_metadata': {'$ref': '#/definitions/UsageMetadata'}},
 'required': ['content'],
 'definitions': {'ToolCall': {'title': 'ToolCall',
   'type': 'object',
   'properties': {'name': {'title': 'Name', 'type': 'string'},
    'args': {'title': 'Args', 'type': 'object'},
    'id': {'title': 'Id', 'type': 'string'},
    'type': {'title': 'Type', 'enum': ['tool_call'], 'type': 'string'}},
   'required': ['name', 'args', 'id']},
  'InvalidToolCall': {'title': 'InvalidToolCall',
   'type': 'object',
   'properties': {'name': {'title': 'Name', 'type': 'string'},
    'args': {'title': 'Args', 'type': 'string'},
    'id': {'title': 'Id', 'type': 'string'},
    'error': {'title': 'Error', 'type': 'string'},
    'type': {'title': 'Type',
     'enum': ['invalid_tool_call'],
     'type': 'string'}},
   'required': ['name', 'args', 'id', 'error']},
  'UsageMetadata': {'title': 'UsageMetadata',
   'type': 'object',
   'properties': {'input_tokens': {'title': 'Input Tokens', 'type': 'integer'},
    'output_tokens': {'title': 'Output Tokens', 'type': 'integer'},
    'total_tokens': {'title': 'Total Tokens', 'type': 'integer'}},
   'required': ['input_tokens', 'output_tokens', 'total_tokens']}}}




