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

Track operations
From now on, we can track the operations and the cost of this project from LangSmith:

https://smith.langchain.com/

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

Donde veas esto: 
```
llm = ChatOpenAI(model="gpt-3.5-turbo")
```

Escribe esto en su lugar: 
```
llm = ChatOpenAI(model="gpt-4o-mini")
```

Sustituir mixtral-8x7b-32768 con versiones posteriores como  mistral-saba-24b

### Connect with alternative LLMs

Intro to Groq

- Groq is an AI Startup company. It is not the same as Grok, the LLM from Elon Musk.
- It has developed a new chip call LPU (Language Processing Unit) which is specificly design to run LLMs faster and cheaper.
- It offers a Groq Cloud where you can try Open Source LLMs like Llama3 or Mixtral.
- It allows you to use Llama3 or Mixtral in your apps for free using a Groq API Key with some Rate Limits.

Groq Rate limits: https://console.groq.com/settings/limits


```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_groq import ChatGroq

llamaChatModel = ChatGroq(
    model="llama3-70b-8192"
)

mistralChatModel = ChatGroq(
    model="mixtral-8x7b-32768"
)

messages = [
    ("system", "You are an historian expert in the Kennedy family."),
    ("human", "How many members of the family died tragically?"),
]

print("\n----------\n")

print("How many members of the family died tragically? - LLama3 Response:")

print("\n----------\n")

llamaResponse = llamaChatModel.invoke(messages)

print(llamaResponse.content)

print("\n----------\n")

print("How many members of the family died tragically? - Mistral Response:")

print("\n----------\n")

mistralResponse = mistralChatModel.invoke(messages)

print(mistralResponse.content)
```

### DeepSeek

Documentación de LangChain para ChatDeepSeek:
https://python.langchain.com/api_reference/deepseek/chat_models/langchain_deepseek.chat_models.ChatDeepSeek.html

### Grok 4

### Kimi K2

### Modelo Open Source de OpenAI (GPT-OSS)

¿Qué es GPT‑OSS?
GPT‑OSS es la familia de modelos de lenguaje open source de OpenAI. Son modelos potentes, gratuitos y disponibles para todos bajo licencia Apache 2.0. Actualmente hay dos versiones disponibles:

gpt-oss-20b: 3.600 millones de parámetros activos, hasta 128.000 tokens de contexto

gpt-oss-120b: 5.100 millones de parámetros activos, hasta 128.000 tokens de contexto

Estos modelos ofrecen prácticamente las mismas capacidades que los modelos comerciales (como ChatGPT), sin coste de uso.

```python
from langchain.chat_models import ChatOpenAI
 
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="TU_API_KEY_DE_GROQ",  # Sustituye por tu clave
    model="gpt-oss-llama3-70b"     # O "mixtral-8x7b", según disponibilidad
)
 
response = llm.invoke("Explica qué es el modelo GPT‑OSS en términos sencillos.")
print(response.content)
```

GPT-4o ("Omni")
- Tipo: Multimodal (texto, audio, imagen)
- Ventana de contexto: 128.000 tokens
- Fortalezas: Respuesta rápida (320 ms), 50 % más barato que GPT-4 Turbo, interacción multimodal en tiempo real, capacidades multilingües avanzadas.
- Integración con LangChain y LangGraph: Sí (soporte completo).
- Acceso: Disponible para usuarios Free, Plus y Pro de ChatGPT y vía API.

Serie GPT-4.1 (GPT-4.1, 4.1 Mini, 4.1 Nano)
- Tipo: Texto y multimodal básico
- Ventana de contexto: Hasta 1 millón de tokens
- Fortalezas: Excelente en programación, seguimiento de instrucciones, manejo de documentos largos, menor latencia y costes reducidos.
- Integración con LangChain y LangGraph: Sí (parcial para Mini/Nano mediante adaptadores API).
- Acceso: Solo vía API.

GPT-4.5 (Nombre en clave: Orion)
- Tipo: Multimodal mejorado
- Ventana de contexto: No divulgada explícitamente
- Fortalezas: Flujo conversacional más natural, inteligencia emocional, menos alucinaciones.
- Integración con LangChain y LangGraph: Sí (acceso limitado en beta).
- Acceso: Suscripciones pagadas a ChatGPT.

Serie o (o1, o3, o4-mini)
- Tipo: Modelos especializados en razonamiento
- Ventana de contexto: Avanzada, pero no pública
- Fortalezas:
  - o1: Razonamiento científico y matemático.
  - o3: Resolución avanzada de problemas con integración de herramientas.
  - o4-mini: Gran desempeño en matemáticas, programación y tareas visuales.
- Integración con LangChain y LangGraph: Sí (se requiere acceso empresarial para algunos modelos).
- Acceso: Vía API o suscripción Pro.

GPT-5 (Próximamente)
- Previsto: Capacidades unificadas, aún más avanzadas en razonamiento y procesamiento multimodal.
- Integración con LangChain y LangGraph: Prevista (en preparación).
- Acceso: Más adelante en 2025.

Comparando los Modelos de OpenAI con los Mejores Modelos No-OpenAI en 2025
Claude 3.7 Sonnet (Anthropic)
- Tipo: Texto, datos estructurados, imágenes
- Ventana de contexto: Modo de Pensamiento Extendido
- Fortalezas: Razonamiento híbrido rápido/profundo, Claude Code para desarrolladores.
- Integración con LangChain y LangGraph: Sí (soporte nativo).
- Acceso: API de Claude e integraciones.

Gemini 2.0 Flash (Google DeepMind)
- Tipo: Multimodal (texto, vídeo, datos del mundo real)
- Ventana de contexto: 1 millón de tokens
- Fortalezas: Agentes de IA con capacidad de acción, respuestas en tiempo real, integración con el ecosistema de Google.
- Integración con LangChain y LangGraph: Sí (a través de plugins experimentales).
- Acceso: A través de la API de Gemini Pro.

Grok 3 (xAI / Elon Musk)
- Tipo: Multimodal
- Ventana de contexto: No especificada claramente
- Fortalezas: Integración con redes sociales (X), razonamiento mejorado.
- Integración con LangChain y LangGraph: No (requiere SDKs personalizados).
- Acceso: A través de la plataforma X y APIs.

DeepSeek R1 (DeepSeek)
- Tipo: Texto y razonamiento lógico
- Ventana de contexto: No divulgada
- Fortalezas: Modelo open-source centrado en tareas matemáticas y lógicas.
- Integración con LangChain y LangGraph: Sí (conectores desarrollados por la comunidad).
- Acceso: Licencia MIT (código abierto).

LLaMA 4 (Meta) – Scout & Maverick
- Tipo: Multimodal (texto, imagen)
- Ventana de contexto: Scout: 10 millones de tokens; Maverick: 1 millón
- Fortalezas: Arquitectura Mixture-of-Experts, contexto ultralargo, multilingüe.
- Integración con LangChain y LangGraph: Sí (Meta ofrece soporte oficial).
- Acceso: Pesos abiertos para uso comercial e investigación.

Qwen 2.5 Max (Alibaba)
- Tipo: Multimodal (texto, imagen, audio)
- Ventana de contexto: No especificada
- Fortalezas: Código abierto, integración con productos de Alibaba.
- Integración con LangChain y LangGraph: Parcial (soporte en crecimiento).
- Acceso: Open-source bajo Licencia Qwen.

Gemma (Google)
- Tipo: Generación de texto con enfoque académico
- Ventana de contexto: No especificada
- Fortalezas: Alto rendimiento en redacción técnica y académica.
- Integración con LangChain y LangGraph: Sí (experimental, en expansión).
- Acceso: Código abierto.

Mistral 7B (Mistral)
- Tipo: Modelo compacto de texto
- Ventana de contexto: Limitada
- Fortalezas: Alta eficiencia para dispositivos con recursos limitados.
- Integración con LangChain y LangGraph: Sí (soporte oficial).
- Acceso: Código abierto.

Phi-3 (Microsoft)
- Tipo: Modelo pequeño (SLM)
- Ventana de contexto: No especificada
- Fortalezas: Alta eficiencia, integración con productos de Microsoft.
- Integración con LangChain y LangGraph: Sí (disponible mediante puente API).
- Acceso: Licencia MIT.

Resumen Comparativo
Multimodalidad
La mayoría de los modelos punteros en 2025 son multimodales. Para tareas que implican procesamiento o generación de imágenes, audio o vídeo, los modelos de OpenAI, Google, Anthropic y Meta son los preferidos.

Modelos con capacidades multimodales (texto + imágenes/audio/video):

✅ Sí: GPT-4o, GPT-4.1, GPT-4.5, Serie o, Claude 3.7, Gemini 2.0, Grok 3, DeepSeek R1 (limitado), LLaMA 4, Qwen 2.5 Max

❌ No: Gemma, Mistral 7B, Phi-3

Ventana de Contexto
Para procesamiento de documentos largos o aplicaciones con gran memoria, LLaMA 4, Gemini 2.0 y GPT-4.1 ofrecen el mejor rendimiento.

Modelos con mayor capacidad de contexto:

🟢 Muy grande (≥1M tokens): GPT-4.1, Gemini 2.0, LLaMA 4

🟡 Grande (100K–500K): GPT-4o

🔴 No especificada o pequeña: GPT-4.5, Serie o, Claude 3.7, Grok 3, DeepSeek R1, Qwen 2.5 Max, Gemma, Phi-3, Mistral 7B



Código Abierto

Si el código abierto es esencial (para transparencia, autoalojamiento o bajo coste), destacan Meta, DeepSeek, Alibaba, Mistral y Microsoft.

Modelos disponibles para uso/modificación libre:

✅ Sí: DeepSeek R1, LLaMA 4, Qwen 2.5 Max, Gemma, Mistral 7B, Phi-3

❌ No: GPT-4o, GPT-4.1, GPT-4.5, Serie o, Claude 3.7, Gemini 2.0, Grok 3



Integración con LangChain / LangGraph

La mayoría de los modelos líderes se integran con LangChain/LangGraph, permitiendo flujos de trabajo complejos. Solo Grok 3 carece por ahora de integración pública.

Modelos compatibles con flujos de trabajo y agentes:

✅ Soporte oficial o completo: GPT-4o, GPT-4.1, GPT-4.5, Serie o, Claude 3.7, Gemini 2.0, DeepSeek R1, LLaMA 4, Mistral 7B, Phi-3

🟡 Soporte parcial o comunitario: Qwen 2.5 Max, Gemma

❌ No compatible: Grok 3

### Prompt Templates

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import OpenAI

llmModel = OpenAI()

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} story about {topic}."
)

llmModelPrompt = prompt_template.format(
    adjective="curious", 
    topic="the Kennedy family"
)

response = llmModel.invoke(llmModelPrompt)

print("Tell me one curious thing about the Kennedy family:")
print(response)

print("\n----------\n")

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an {profession} expert on {topic}."),
        ("human", "Hello, Mr. {profession}, can you please answer a question?"),
        ("ai", "Sure!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(
    profession="Historian",
    topic="The Kennedy family",
    user_input="How many grandchildren had Joseph P. Kennedy?"
)

response = chatModel.invoke(messages)

print("How many grandchildren had Joseph P. Kennedy?:")
print(response.content)

print("\n----------\n")

# Few Shot Prompting ------------------------------------

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
```

### Chains - Secuencias de instrucciones

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import OpenAI

llmModel = OpenAI()


from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")


from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

chain = final_prompt | chatModel

response = chain.invoke({"input": "Who was JFK?"})

print("\n----------\n")

print("Translate: Who was JFK?")
print(response.content)

print("\n----------\n")
```

### Output Parsers: Re-formatear las respuestas del LLM

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import OpenAI

llmModel = OpenAI()

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.prompts import PromptTemplate

from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

json_parser = SimpleJsonOutputParser()
# json_parser.get_format_instructions()
# 'Return a JSON Object.'

json_chain = json_prompt | llmModel | json_parser

response = json_chain.invoke({"question": "What is the biggest country?"})

print("What is the biggest country?")
print(response)
# {'answer': 'Russia'}

print("\n----------\n")


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
    
# Set up a parser
parser = JsonOutputParser(pydantic_object=Joke)

# Inject parser instructions into the prompt template.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create a chain with the prompt and the parser
chain = prompt | chatModel | parser

response = chain.invoke({"query": "Tell me a joke."})

print("Tell me a joke in custom format defined by Pydantic:")
print(response)

print("\n----------\n")
```

## LangChain Basic: Cómo trabajar con Datos. RAG Basics.

- Data Loaders
- Large data assets y RAG
- Splitters
- Embeddings
- Vector stores
- Retrievers, top k results, indexing

Basic Langchain
- Data in (txt, pdf, csv, sql, xls, html, png, etc.)
- Actions (RAG, splitters, embeddings, vector stores)
- LLMs
- Data out
- Actions (RAG, retrievers, top k results)

### Data Loaders: carga archivos de datos y pregunta al LLM sobre sus contenidos
- https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
- https://python.langchain.com/v0.1/docs/integrations/document_loaders/

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/be-good.txt")

loaded_data = loader.load()

print("\n----------\n")

print("Loaded TXT file:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('./data/Street_Tree_List.csv')

loaded_data = loader.load()

print("\n----------\n")

print("Loaded CSV file:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

from langchain_community.document_loaders import UnstructuredHTMLLoader

loader = UnstructuredHTMLLoader('./data/100-startups.html')

loaded_data = loader.load()

print("\n----------\n")

print("Loaded HTML page:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./data/5pages.pdf')

loaded_data = loader.load_and_split()

print("\n----------\n")

print("Loaded HTML page:")

print("\n----------\n")
#print(loaded_data[0].page_content)

print("\n----------\n")

from langchain_community.document_loaders import WikipediaLoader

name = "JFK"

loader = WikipediaLoader(query=name, load_max_docs=1)

loaded_data = loader.load()[0].page_content

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("human", "Answer this {question}, here is some extra {context}"),
    ]
)

messages = chat_template.format_messages(
    question="What was the full name of JFK?",
    context=loaded_data
)

response = chatModel.invoke(messages)

print("\n----------\n")

print("Respond from Wikipedia: What was the full name of JFK?")

print("\n----------\n")
#print(response.content)

print("\n----------\n")
```

### RAG: Carga grandes archivos de datos y pregunta al LLM sobre sus contenidos

What if the loaded data is too large? We will use RAG.
- When you load a document, you end up with strings. Sometimes the strings will be too large to fit into the context window. In those occassions we will use the RAG technique:
  - Split document in small chunks.
  - Transform text chunks in numeric chunks (embeddings).
  - Load embeddings to a vector database (aka vector store).
  - Load question and retrieve the most relevant embeddings to respond it.
  - Sent the embeddings to the LLM to format the response properly.
 
### Splitters: Divide grandes archivos de datos en pequeños fragmentos

Splitters / Document Transformers

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/be-good.txt")

loaded_data = loader.load()

print("\n----------\n")

print("TXT file loaded:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

print("Content of the first page loaded:")

print("\n----------\n")
#print(loaded_data[0].page_content)

print("\n----------\n")

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([loaded_data[0].page_content])

print("\n----------\n")

print("How many chunks of text were created by the splitter?")

print("\n----------\n")
#print(len(texts))

print("\n----------\n")

print("Print the first chunk of text")

print("\n----------\n")
#print(texts[0])

print("\n----------\n")

# Splitting with metadata
metadatas = [{"chunk": 0}, {"chunk": 1}]

documents = text_splitter.create_documents(
    [loaded_data[0].page_content, loaded_data[0].page_content], 
    metadatas=metadatas
)

print("\n----------\n")

print("Using a second splitter to create chunks of text with metadata, print the first chunk of text with metadata")

print("\n----------\n")
#print(documents[0])

print("\n----------\n")
```

Recursive Character Splitter
- This text splitter is the recommended one for generic text.
- It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""].
- This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

Simple Explanation:
- The "Recursive Character Splitter" is a method used to divide text into smaller, more manageable chunks, designed specifically to maintain the semantic integrity of the text.
- It operates by attempting to split the text using a list of characters in a specified order—beginning with the largest units like paragraphs, then moving to sentences, and finally to individual words if needed.
- The default sequence for splitting is ["\n\n", "\n", " ", ""], which means it first tries to split the text at double newline characters to separate paragraphs, then at single newlines for any remaining large blocks, followed by spaces to isolate sentences or phrases, and finally using an empty string if finer splitting is necessary.
- This method is particularly effective because it tries to keep text chunks as meaningful and complete as possible, ensuring that each chunk has a coherent piece of information.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=26,
    chunk_overlap=4
)

text1 = 'abcdefghijklmnopqrstuvwxyzabcdefg'

text2 = """
Data that Speak
LLM Applications are revolutionizing industries such as 
banking, healthcare, insurance, education, legal, tourism, 
construction, logistics, marketing, sales, customer service, 
and even public administration.

The aim of our programs is for students to learn how to 
create LLM Applications in the context of a business, 
which presents a set of challenges that are important 
to consider in advance.
"""

recursive_splitter.split_text(text1)

recursive_splitter.split_text(text2)

second_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

second_recursive_splitter.split_text(text2)
#['Data that Speak\nLLM Applications are revolutionizing industries such as \nbanking, healthcare, insurance, education, legal, tourism,',
# 'construction, logistics, marketing, sales, customer service, \nand even public administration.',
# 'The aim of our programs is for students to learn how to \ncreate LLM Applications in the context of a business,',
# 'which presents a set of challenges that are important \nto consider in advance.']
```

### Embeddings: Transforma los fragmentos de texto en números (vectores)

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/be-good.txt")

loaded_data = loader.load()

print("\n----------\n")

print("TXT file loaded:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

print("Content of the first page loaded:")

print("\n----------\n")
#print(loaded_data[0].page_content)

print("\n----------\n")

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([loaded_data[0].page_content])

print("\n----------\n")

print("How many chunks of text were created by the splitter?")

print("\n----------\n")
#print(len(texts))

print("\n----------\n")

print("Print the first chunk of text")

print("\n----------\n")
#print(texts[0])

print("\n----------\n")

metadatas = [{"chunk": 0}, {"chunk": 1}]

documents = text_splitter.create_documents(
    [loaded_data[0].page_content, loaded_data[0].page_content], 
    metadatas=metadatas
)

print("\n----------\n")

print("Using a second splitter to create chunks of thext with metadata, print the first chunk of text with metadata")

print("\n----------\n")
#print(documents[0])

print("\n----------\n")

from langchain_openai import OpenAIEmbeddings

# OpenAIEmbeddings is a lot of computational cost
# Use OpenAIEmbeddings with documents that are too big can be really expensive
# Do it with prudence, little by little and using LangSmith
embeddings_model = OpenAIEmbeddings()

chunks_of_text =     [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]

# Converting texts into numbers
embeddings = embeddings_model.embed_documents(chunks_of_text)

print("\n----------\n")

print("How many embeddings were created?")

print("\n----------\n")
#print(len(embeddings))

print("\n----------\n")

print("How long is the first embedding?")

print("\n----------\n")
#print(len(embeddings[0]))

print("\n----------\n")

print("Print the last 5 elements of the first embedding:")

print("\n----------\n")
#print(embeddings[0][:5])

print("\n----------\n")

# Converting the user question to numbers
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
```

### Vector Databases (vector stores): Almacena y gestiona embeddings
- https://python.langchain.com/v0.1/docs/integrations/vectorstores/
- https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loaded_document = TextLoader('./data/state_of_the_union.txt').load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings())

question = "What did the president say about the John Lewis Voting Rights Act?"

response = vector_db.similarity_search(question)

print("\n----------\n")

print("Ask the RAG App: What did the president say about the John Lewis Voting Rights Act?")

print("\n----------\n")
print(response[0].page_content)

print("\n----------\n")
```

### Retrievers: Encuentra los embeddings que mejor responden a la pregunta

Retriever: es como un repartidor

**Vector Stores vs. Retrievers**

1. Purpose and Functionality:

- Vector Stores: These are specialized databases designed to store information in the form of vectors (high-dimensional data points that represent text or other information). Vector stores are primarily used for quickly searching and retrieving similar vectors based on a query vector. They are focused on efficiently handling similarity comparisons between the stored vectors and any query vector.
- Retrievers: Retrievers are more general tools that use various methods, including vector stores, to find and return relevant documents or information in response to a query. A retriever doesn't necessarily store the information itself but knows how to access and retrieve it when needed.

2. Storage vs. Retrieval:

- Vector Stores: As the name implies, these are primarily concerned with storing data in a structured way that makes it fast and efficient to perform similarity searches.
- Retrievers: While they may utilize storage systems like vector stores, retrievers are more focused on the act of fetching the right information in response to a user's query. Their main job is to provide the end-user with the most relevant information or documents based on the input they receive.

3. Flexibility:

- Vector Stores: These are somewhat limited in their scope to handling tasks that involve similarity searches within the stored vectors. They are a specific tool for specific types of data retrieval tasks.
- Retrievers: They can be designed to use different back-end systems (like vector stores or other databases) and can be part of larger systems that may involve more complex data processing or response generation.
In summary, vector stores in LangChain are about how information is stored and efficiently accessed based on similarity, while retrievers are about using various methods (including vector stores) to actively fetch and return the right information in response to diverse queries.

**Differences .similarity_search vs. .as_retriever()**

Both methods involve finding the most relevant text based on a query, but they are structured differently and may offer different functionalities based on their implementation.

**.similarity_search**

This method directly performs a similarity search against a vector database, which in your first code snippet is managed by the Chroma class. The process includes:

- Embedding the input query using the same model that was used to embed the document chunks.
- Searching the vector database for the closest vectors to the query's embedding.
- Returning the most relevant chunks based on their semantic similarity to the query.
  
This method is straightforward and typically used when you need to quickly find and retrieve the text segments that best match the query.

**.as_retriever()**

This method involves a different approach:

1. Retriever Setup: In the second code snippet, **vector_db.as_retriever()** converts the vector database (managed by FAISS in this case) into a retriever object. This object abstracts the similarity search into a retriever model that can be used in more complex retrieval-augmented generation (RAG) tasks.
2. Invoke Method: The **invoke()** function on the retriever is then used to perform the query. This method can be part of a larger system where the retriever is integrated with other components (like a language model) to generate answers or further process the retrieved documents.
   
**Key Differences**

- **Flexibility:** **.as_retriever()** provides a more flexible interface that can be integrated into larger, more complex systems, potentially combining retrieval with generation (like in RAG setups). This method is beneficial in applications where the retrieved content might be used as input for further processing or answer generation.
- **Backend Implementation:** While **.similarity_search** directly accesses the vector database, **.as_retriever()** encapsulates this access within a retriever object, which might have additional functionalities or optimizations for specific retrieval tasks.
- **Use Cases:** The direct **.similarity_search** might be faster and more straightforward for simple query-to-document retrieval tasks. In contrast, **.as_retriever()** could be used in scenarios requiring additional steps after retrieval, like feeding the retrieved information into a language model for generating coherent and context-aware responses.
Both methods are useful, but their appropriateness depends on the specific requirements of your application, such as whether you need straightforward retrieval or a more complex retrieval-augmented generation process.

LCEL = LangChain Expression Language, LangChain de 2da Generación

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loaded_document = TextLoader('./data/state_of_the_union.txt').load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings())

question = "What did the president say about the John Lewis Voting Rights Act?"

response = vector_db.similarity_search(question)

print("\n----------\n")

print("Ask the RAG App: What did the president say about the John Lewis Voting Rights Act?")

print("\n----------\n")
#print(response[0].page_content)

print("\n----------\n")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/state_of_the_union.txt")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loaded_document = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

embeddings = OpenAIEmbeddings()

vector_db = FAISS.from_documents(chunks_of_text, embeddings)

retriever = vector_db.as_retriever()

response = retriever.invoke("what did he say about ketanji brown jackson?")

print("\n----------\n")

print("Ask the RAG App with Retriever: What did he say about ketanji brown jackson?")

print("\n----------\n")
print(response[0].page_content)

print("\n----------\n")
```

### Top K: decide cuántos resultados se utilizarán para responder a la pregunta

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loaded_document = TextLoader('./data/state_of_the_union.txt').load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings())

question = "What did the president say about the John Lewis Voting Rights Act?"

response = vector_db.similarity_search(question)

print("\n----------\n")

print("Ask the RAG App: What did the president say about the John Lewis Voting Rights Act?")

print("\n----------\n")
#print(response[0].page_content)

print("\n----------\n")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/state_of_the_union.txt")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loaded_document = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

embeddings = OpenAIEmbeddings()

vector_db = FAISS.from_documents(chunks_of_text, embeddings)

retriever = vector_db.as_retriever()

response = retriever.invoke("what did he say about ketanji brown jackson?")

print("\n----------\n")

print("Ask the RAG App with Retriever: What did he say about ketanji brown jackson?")

print("\n----------\n")
#print(response[0].page_content)

print("\n----------\n")

retriever = vector_db.as_retriever(search_kwargs={"k": 1})

response = retriever.invoke("what did he say about ketanji brown jackson?")

print("\n----------\n")

print("Ask the RAG App with top k=1: What did he say about ketanji brown jackson?")

print("\n----------\n")
print(response[0].page_content)

print("\n----------\n")
```

### RAG con LCEL: Primer vistazo a la solución compacta creada con LCEL

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loaded_document = TextLoader('./data/state_of_the_union.txt').load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings())

question = "What did the president say about the John Lewis Voting Rights Act?"

response = vector_db.similarity_search(question)

print("\n----------\n")

print("Ask the RAG App: What did the president say about the John Lewis Voting Rights Act?")

print("\n----------\n")
#print(response[0].page_content)

print("\n----------\n")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/state_of_the_union.txt")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loaded_document = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

embeddings = OpenAIEmbeddings()

vector_db = FAISS.from_documents(chunks_of_text, embeddings)

retriever = vector_db.as_retriever()

response = retriever.invoke("what did he say about ketanji brown jackson?")

print("\n----------\n")

print("Ask the RAG App with Retriever: What did he say about ketanji brown jackson?")

print("\n----------\n")
#print(response[0].page_content)

print("\n----------\n")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response = chain.invoke("what did he say about ketanji brown jackson?")

print("\n----------\n")

print("Ask the RAG App with LCEL: What did he say about ketanji brown jackson?")

print("\n----------\n")
print(response)

print("\n----------\n")
```

### Indexing: Método avanzado para gestionar documentos en vector stores

- https://python.langchain.com/v0.1/docs/modules/data_connection/indexing/

#### Indexing

- Advanced way to manage and search through many documents in a vector store.

**Indexing**

LangChain Indexing an **advanced technique** designed to efficiently integrate and synchronize documents from various sources into a vector store. This is particularly useful for tasks like semantic searches, where the aim is to find documents with similar meanings rather than those that match on specific keywords.

Core Features and Their Benefits
1. Avoiding Duplication: By preventing the same content from being written multiple times into the vector store, the system conserves storage space and reduces redundancy.
2. Change Detection: The API is designed to detect if a document has changed since the last index. If there are no changes, it avoids re-writing the document. This minimizes unnecessary write operations and saves computational resources.
3. Efficient Handling of Embeddings: Embeddings for unchanged content are not recomputed, thus saving processing time and further enhancing system efficiency.
   
**Technical Mechanics: Record Management**

The **RecordManager** is a pivotal component of the LangChain indexing system. It meticulously records each document's write activity into the vector store. Here's how it works:
- Document Hash: Every document is hashed. This hash includes both the content and metadata, providing a unique fingerprint for each document.
- Write Time and Source ID: Alongside the hash, the time the document was written and a source identifier are also stored. The source ID helps trace the document back to its origin, ensuring traceability and accountability.
These details are crucial for ensuring that only necessary data handling operations are carried out, thereby enhancing efficiency and reducing the workload on the system.

**Operational Efficiency and Cost Savings**

By integrating these features, LangChain indexing not only streamlines the management of document indices but also leads to significant cost savings. This is achieved by:

- Reducing the frequency and volume of data written to and read from the vector store.
- Decreasing the computational demand required for re-indexing and re-computing embeddings.
- Improving the overall speed and relevance of vector search results, which is vital for applications requiring rapid and accurate data retrieval.

**Conclusion**

The LangChain indexing API is a sophisticated tool that leverages modern database and hashing technologies to manage and search through large volumes of digital documents efficiently. It is especially valuable in environments where accuracy, efficiency, and speed of data retrieval are crucial, such as in academic research, business intelligence, and various fields of software development. This technology not only supports effective data management but also promotes cost-effectiveness by optimizing resource utilization.

The Indexing API from LangChain is an advanced feature suitable for very experienced developers primarily due to its complexity and the sophisticated understanding required to implement and manage it effectively. Here's why, broken down into simpler terms:

1. **Complex Integration:** The API is designed to handle documents from various sources and integrate them into a vector store for semantic searches. This requires understanding both the sources of the documents and the mechanics of vector stores, which deal with high-dimensional data representations.

2. **Efficiency Management:** It involves sophisticated features like avoiding duplication of content, detecting changes in documents, and efficiently managing embeddings (data representations). These processes require a deep understanding of data structures, hashing, and optimization techniques to ensure the system is efficient and does not waste resources.

3. Technical Operations:

- **Record Management:** The RecordManager component is crucial in tracking each document’s activity in the vector store, using detailed information such as document hashes, write times, and source IDs. This level of detail in record management necessitates familiarity with database operations, data integrity, and possibly cryptographic hashing.
- **Operational Efficiency and Cost Savings:** Implementing the indexing system effectively can lead to significant operational efficiencies and cost savings. However, this requires precise setup and tuning to reduce unnecessary computational demands and storage usage. Developers need to understand how to balance these factors to optimize performance and cost.
  
4. **Advanced Use Cases:** The API supports complex scenarios such as rapid and accurate data retrieval needed in fields like academic research, business intelligence, and software development. Each of these applications might require specialized knowledge to fully leverage the potential of the indexing API.

5. **Risk of Misimplementation:** Incorrect implementation can lead to inefficient data handling, increased operational costs, and slower retrieval times, which is why a high level of expertise is necessary to avoid potential pitfalls.

In conclusion, the LangChain Indexing API is an advanced tool that involves detailed and complex processes to manage large volumes of data efficiently. Its use is recommended for developers who are very experienced because it requires a strong understanding of database systems, data efficiency, and system integration. Proper utilization can greatly enhance the performance and cost-effectiveness of systems that rely on fast and accurate data retrieval.

**A Simple Example**

The LangChain Indexing API is a sophisticated tool that helps integrate and manage large sets of documents efficiently. To make it clearer, let's consider a simple example that illustrates how it could be used:

Example Scenario: Managing Research Papers in a University Database

**Context:** Imagine you are developing a system for a university's library to manage and search through thousands of research papers. The goal is to allow students and faculty to quickly find papers relevant to their interests based on content similarity, not just by keywords.

**Step-by-Step Use of LangChain Indexing API:**

1. **Gathering Documents:**
- Collect digital copies of all research papers to be included in the system.
- These might come from various departments or sources within the university.
  
2. **Integration into Vector Store:**
- Each research paper is converted into a "vector" using text embedding techniques. A vector is a numerical representation that captures the essence of the paper's content.
- These vectors are stored in a vector store, a specialized database for managing such data.
  
3. **Avoiding Duplication:**
- As new papers are added, the LangChain Indexing API checks if a similar paper already exists in the vector store.
- It uses a hash (a unique identifier generated from the paper’s content and metadata) to prevent the same paper from being stored multiple times, saving space and reducing clutter.

4. **Change Detection:**
- If a paper in the database is updated or revised, the API detects changes using the hash comparison.
- It updates the vector representation only if changes are detected, saving on unnecessary computational resources.

5. **Search and Retrieval:**
- When a student queries the system looking for papers on a specific topic, like "quantum computing applications," the API helps retrieve the most relevant papers.
- It does this by comparing the query's vector with those in the vector store and returning papers whose vectors are most similar in content, not just those that contain specific keywords.

6. **Operational Efficiency:**
- The system is optimized to handle large volumes of data efficiently, ensuring quick responses even when multiple users are accessing it simultaneously.
- This efficiency is crucial during exam periods or when new research is published and interest peaks.


**Conclusion**

By using the LangChain Indexing API, the university library can manage its research papers more effectively, making them easily accessible based on content relevance. This leads to better research outcomes for students and faculty and maximizes the use of the library’s resources.

This example demonstrates how the Indexing API not only simplifies the management of documents but also enhances the retrieval process, making it more aligned with the users' actual needs.
