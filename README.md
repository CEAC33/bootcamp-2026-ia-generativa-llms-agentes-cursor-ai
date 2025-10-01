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

https://github.com/AI-LLM-Bootcamp/b401

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

https://github.com/AI-LLM-Bootcamp/b402

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

## LangChain Basics: LCEL Chains y Runnables

https://github.com/AI-LLM-Bootcamp/b403

- Runnables
- LCEL Chains: sequence of runnables
- Runnable execution order
- Runnable execution alternatives
- Built-in runnables and functions
- Basic LCEL chain operations

Basic Lang Chain
- Data In
- Actions
  - runnables
  - chains
  - built-in runnables
  - built-in functions
  - combining chains
  - nesting chains
  - fallback for chains
- LLMs
- Data Out

### Una Chain LCEL Sencilla: Secuencia de Acciones, Secuencia de Runnables

Simple Chain
- Perform several actions in a particular order.

Intro
- Chains are sequences of actions with input or output data.
- Since the emmergence of LCEL, LangChain is favouring LCEL chains over Traditional (Legacy) Built-in Chains, but these are still maintained and frequently used.

  
LangChain documentation on Chains
- See the LCEL documentation page on Chains here. - https://python.langchain.com/v0.2/docs/how_to/sequence/
- See the Legacy documentation page on Chains here. In this page you can see a list of the main built-in legacy chains. - https://python.langchain.com/v0.1/docs/modules/chains/

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {politician}")

chain = prompt | chatModel | StrOutputParser()

response = chain.invoke({"politician": "JFK"})

print("\n----------\n")

print("Result from invoking the chain:")

print("\n----------\n")
print(response)

print("\n----------\n")
```

What does StrOutputParser do?

- The StrOutputParser is a specific tool within LangChain that simplifies the output from these language models. It takes the complex or structured output from the models and converts it into plain text (a string). This makes it easier to use this output in applications, like displaying it to users or processing it further for other purposes.
  
Specific Functions of StrOutputParser:

- **For LLM Outputs:** If a language model produces output that is already in text form, StrOutputParser doesn’t change it; it just passes it through.
- **For ChatModel Outputs:** If the output comes from a ChatModel (a type of language model designed for conversation), StrOutputParser extracts the main content from the structured output to make sure it ends up as plain text.

```
chain.invoke({"politician": "JFK"})
```
'One curious fact about JFK is that he was the first U.S. president to have been born in the 20th century. He was born on May 29, 1917, making him the first president born after the turn of the century.'

### La Chain Legacy de LangChain vs La Nueva Chain de LCEL

Intro to LCEL
Intro
- LCEL has become the backbone of the newest versions of LangChain.
- Traditional chains are still supported, but treated as "Legacy" and have less functionality than the new LCEL chains.
- Many students struggle with LCEL.


Main goals of LCEL
- Make it easy to build chains in a compact way.
- Support advanced LangChain functionality.

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LEGACY CHAIN
from langchain.chains import LLMChain

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

traditional_chain = LLMChain(
    llm=model,
    prompt=prompt
)

response = traditional_chain.predict(soccer_player="Maradona")

print("\n----------\n")

print("Legacy chain:")

print("\n----------\n")
print(response)

print("\n----------\n")

# New LCEL Chain
# The "pipe" operator | is the main element of the LCEL chains.
# The order (left to right) of the elements in a LCEL chain matters.
# An LCEL Chain is a Sequence of Runnables.

chain = prompt | model | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")
```

- All the components of the chain are Runnables.
- When we write chain.invoke() we are using invoke with all the componentes of the chain in an orderly manner:
  - First, we apply .invoke() to the prompt.
  - Then, with the previous output, we apply .invoke() to the model.
  - And finally, with the previous output, we apply .invoke() to the output parser.
 
### El concepto clave de LCEL que debes dominar: El Runnable Execution Order

LCEL Chain
- **An LCEL Chain is a Sequence of Runnables.**
- Almost any component in LangChain (prompts, models, output parsers, vector store retrievers, tools, etc) can be used as a Runnable.
- Runnables can be chained together using the pipe operator |. The resulting chains of runnables are also runnables themselves.
- The order (left to right) of the elements in a LCEL chain matters.


```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")

prompt.invoke({"soccer_player": "Ronaldo"})

from langchain_core.messages.human import HumanMessage

output_after_first_step = [HumanMessage(content='tell me a curious fact about Ronaldo')]

model.invoke(output_after_first_step)

from langchain_core.messages.ai import AIMessage

output_after_second_step = AIMessage(content='One curious fact about Cristiano Ronaldo is that he does not have any tattoos on his body. Despite the fact that many professional athletes have tattoos, Ronaldo has chosen to keep his body ink-free.', response_metadata={'token_usage': {'completion_tokens': 38, 'prompt_tokens': 14, 'total_tokens': 52}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-c9812511-043a-458a-bfb8-005bc0d057fb-0', usage_metadata={'input_tokens': 14, 'output_tokens': 38, 'total_tokens': 52})

response = output_parser.invoke(output_after_second_step)

print("\n----------\n")

print("After replicating the process without using a LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")
```

- All the components of the chain are Runnables.
- **When we write chain.invoke() we are using invoke with all the componentes of the chain in an orderly manner:**
  - First, we apply .invoke() with the user input to the prompt template.
  - Then, with the completed prompt, we apply .invoke() to the model.
  - And finally, we apply .invoke() to the output parser with the output of the model.
- IMPORTANT: the order of operations in a chain matters. If you try to execute the previous chain with the components in different order, the chain will fail.

<img width="889" height="407" alt="Screenshot 2025-09-29 at 10 04 44 p m" src="https://github.com/user-attachments/assets/cdef274c-1239-46f2-aafe-afc35428ec73" />

Ways to execute Runnables
- Remember:
  - An LCEL Chain is a Sequence of Runnables.
  - Almost any component in LangChain (prompts, models, output parsers, etc) can be used as a Runnable.
  - Runnables can be chained together using the pipe operator |. The resulting chains of runnables are also runnables themselves.
  - The order (left to right) of the elements in a LCEL chain matters.

### Formas alternativas de ejecutar Runnables LCEL: invoke, stream, batch

Alternative ways to execute runnables
- Invoke, Stream and Batch.

LCEL Chains/Runnables are used with:
- chain.invoke(): call the chain on an input.
- chain.stream(): call the chain on an input and stream back chunks of the response.
- chain.batch(): call the chain on a list of inputs.

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me one sentence about {politician}.")
chain = prompt | model

response = chain.invoke({"politician": "Churchill"})

print("\n----------\n")

print("Response with invoke:")

print("\n----------\n")
print(response.content)

print("\n----------\n")
    
print("\n----------\n")

print("Response with stream:")

print("\n----------\n")
for s in chain.stream({"politician": "F.D. Roosevelt"}):
    print(s.content, end="", flush=True)

print("\n----------\n")

response = chain.batch([{"politician": "Lenin"}, {"politician": "Stalin"}])

print("\n----------\n")

print("Response with batch:")

print("\n----------\n")
print(response)

print("\n----------\n")
```

The for loop explained in simple terms
This for loop is used to show responses piece by piece as they are received. Here's how it works in simple terms:

1. **Getting Responses in Parts:** The loop receives pieces of a response from the system one at a time. In this example, the system is responding with information about the politician "F.D. Roosevelt."

2. **Printing Out Responses Immediately:** Each time a new piece of the response is received, it immediately prints it out. The setup makes sure there are no new lines between parts, so it all looks like one continuous response.

3. **No Waiting:** By using this loop, you don't have to wait for the entire response to be ready before you start seeing parts of it. This makes it feel quicker and more like a conversation.

This way, the loop helps provide a smoother and more interactive way of displaying responses from the system as they are generated.

**The for loop explained in technical terms**

This for loop is used to handle streaming output from a language model response. Here’s a breakdown of its functionality and context:

1. **Iteration through Streamed Output:** The for loop iterates over the generator returned by chain.stream(...). The stream method of the chain object (which is a combination of a prompt template and a language model) is designed to yield responses incrementally. This is particularly useful when responses from a model are long or need to be displayed in real-time.

2. **Data Fetching:** In this loop, s represents each piece of content that is streamed back from the model as the response is being generated. The model in this case is set up to respond with information about the politician "F.D. Roosevelt".

3. **Output Handling:** Inside the loop, print(s.content, end="", flush=True) is called for each piece of streamed content. The print function is customized with:

  - end="": This parameter ensures that each piece of content is printed without adding a new line after each one, thus allowing the response to appear continuous on a single line.
  - flush=True: This parameter forces the output buffer to be flushed immediately after each print statement, ensuring that each piece of content is displayed to the user as soon as it is received without any delay.

By using this loop, the code is able to display each segment of the model's response as it becomes available, making the user interface more dynamic and responsive, especially for real-time applications where prompt feedback is beneficial.

LCEL Chains/Runnables can be also used asynchronously:
- chain.ainvoke(): call the chain on an input.
- chain.astream(): call the chain on an input and stream back chunks of the response.
- chain.abatch(): call the chain on a list of inputs.

### Built-in LCEL Runnables: passthrough, lambda, parallel, branch

Main built-in LCEL Runnables

Contents
- RunnablePassthrough.
- RunnableLambda.
- RunnableParallel.
  - itemgetter.
- RunnableBranch.

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

"""
RunnablePassthrough
It does not do anything to the input data.
Let's see it in a very simple example: a chain with just RunnablePassthrough() will output the original input without any modification.
"""

from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough()

response = chain.invoke("Abram")

print("\n----------\n")

print("Chain with RunnablePassthrough:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
RunnableLambda
To use a custom function inside a LCEL chain we need to wrap it up with RunnableLambda.
Let's define a very simple function to create Russian lastnames:
"""

def russian_lastname(name: str) -> str:
    return f"{name}ovich"

from langchain_core.runnables import RunnableLambda

chain = RunnablePassthrough() | RunnableLambda(russian_lastname)

response = chain.invoke("Abram")

print("\n----------\n")

print("Chain with RunnableLambda:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
RunnableParallel
- We will use RunnableParallel() for running tasks in parallel.
- This is probably the most important and most useful Runnable from LangChain.
- In the following chain, RunnableParallel is going to run these two tasks in parallel:
  - operation_a will use RunnablePassthrough.
  - operation_b will use RunnableLambda with the russian_lastname function.
"""

from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    {
        "operation_a": RunnablePassthrough(),
        "operation_b": RunnableLambda(russian_lastname)
    }
)

response = chain.invoke("Abram")

print("\n----------\n")

print("Chain with RunnableParallel:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
Instead of using RunnableLambda, now we are going to use a lambda function and we will invoke the chain with two inputs:
"""

chain = RunnableParallel(
    {
        "operation_a": RunnablePassthrough(),
        "soccer_player": lambda x: x["name"]+"ovich"
    }
)

response = chain.invoke({
    "name1": "Jordam",
    "name": "Abram"
})

"""
- See how the lambda function is taking the "name" input.
"""

print("\n----------\n")

print("Chain with RunnableParallel:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
We can add more Runnables to the chain
- In the following example, the prompt Runnable will take the output of the RunnableParallel:
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

def russian_lastname_from_dictionary(person):
    return person["name"] + "ovich"

chain = RunnableParallel(
    {
        "operation_a": RunnablePassthrough(),
        "soccer_player": RunnableLambda(russian_lastname_from_dictionary),
        "operation_c": RunnablePassthrough(),
    }
) | prompt | model | output_parser

response = chain.invoke({
    "name1": "Jordam",
    "name": "Abram"
})

"""
As you saw, the prompt Runnable took "Abramovich", the output of the RunnableParallel, as the value for the "soccer_player" variable.
"""

print("\n----------\n")

print("Chain with RunnableParallel:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    ["AI Accelera has trained more than 10,000 Alumni from all continents and top companies"], embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-3.5-turbo")

retrieval_chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

response = retrieval_chain.invoke("who are the Alumni of AI Accelera?")

"""
Important: the syntax of RunnableParallel can have several variations.
- When composing a RunnableParallel with another Runnable you do not need to wrap it up in the RunnableParallel class. Inside a chain, the next three syntaxs are equivalent:
  - RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
  - RunnableParallel(context=retriever, question=RunnablePassthrough())
  - {"context": retriever, "question": RunnablePassthrough()}
"""

print("\n----------\n")

print("Chain with Advanced Use of RunnableParallel:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
Using itemgetter with RunnableParallel
- When you are calling the LLM with several different input variables.
"""

from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

model = ChatOpenAI(model="gpt-3.5-turbo")

vectorstore = FAISS.from_texts(
    ["AI Accelera has trained more than 5,000 Enterprise Alumni."], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

response = chain.invoke({"question": "How many Enterprise Alumni has trained AI Accelera?", "language": "Pirate English"})

print("\n----------\n")

print("Chain with RunnableParallel and itemgetter:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
RunnableBranch: Router Chain
- A RunnableBranch is a special type of runnable that allows you to define a set of conditions and runnables to execute based on the input.
- A RunnableBranch is initialized with a list of (condition, runnable) pairs and a default runnable. It selects which branch by passing each condition the input it's invoked with. It selects the first condition to evaluate to True, and runs the corresponding runnable to that condition with the input.
- For advanced uses, a custom function (https://python.langchain.com/v0.1/docs/expression_language/how_to/routing/) may be a better alternative than RunnableBranch.
The following advanced example can classify and respond to user questions based on specific topics like rock, politics, history, sports, or general inquiries. It uses some new topics that we will explain in the following lesson. Here’s a simplified explanation of each part:

1. Prompt Templates: Each template is tailored for a specific topic:

- rock_template: Configured for rock and roll related questions.
- politics_template: Tailored to answer questions about politics.
- history_template: Designed for queries related to history.
- sports_template: Set up to address sports-related questions.
- general_prompt: A general template for queries that don't fit the specific categories.
Each template includes a placeholder {input} where the actual user question will be inserted.

2. RunnableBranch: This is a branching mechanism that selects which template to use based on the topic of the question. It evaluates conditions (like x["topic"] == "rock") to determine the topic and uses the appropriate prompt template.

3. Topic Classifier: A Pydantic class that classifies the topic of a user's question into one of the predefined categories (rock, politics, history, sports, or general).

4. Classifier Chain:
- Chain: Processes the user's input to predict the topic.
- Parser: Extracts the predicted topic from the classifier's output.

5. RunnablePassthrough: This component feeds the user's input and the classified topic into the RunnableBranch.

6. Final Chain:

- The user's input is first processed to classify its topic.
- The appropriate prompt is then selected based on the classified topic.
- The selected prompt is used to formulate a question which is then sent to a model (like ChatOpenAI).
- The model’s response is parsed as a string and returned.

7. Execution:

- The chain is invoked with a sample question, "Who was Napoleon Bonaparte?"
- Based on the classification, it selects the appropriate template, generates a query to the chat model, and processes the response.

The system effectively creates a dynamic response generator that adjusts the way it answers based on the topic of the inquiry, making use of specialized knowledge for different subjects.
"""

from langchain.prompts import PromptTemplate

rock_template = """You are a very smart rock and roll professor. \
You are great at answering questions about rock and roll in a concise\
and easy to understand manner.

Here is a question:
{input}"""

rock_prompt = PromptTemplate.from_template(rock_template)

politics_template = """You are a very good politics professor. \
You are great at answering politics questions..

Here is a question:
{input}"""

politics_prompt = PromptTemplate.from_template(politics_template)

history_template = """You are a very good history teacher. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods.

Here is a question:
{input}"""

history_prompt = PromptTemplate.from_template(history_template)

sports_template = """ You are a sports teacher.\
You are great at answering sports questions.

Here is a question:
{input}"""

sports_prompt = PromptTemplate.from_template(sports_template)

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

general_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question as accurately as you can.\n\n{input}"
)
prompt_branch = RunnableBranch(
  (lambda x: x["topic"] == "rock", rock_prompt),
  (lambda x: x["topic"] == "politics", politics_prompt),
  (lambda x: x["topic"] == "history", history_prompt),
  (lambda x: x["topic"] == "sports", sports_prompt),
  general_prompt
)

from typing import Literal

from langchain.pydantic_v1 import BaseModel
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function


class TopicClassifier(BaseModel):
    "Classify the topic of the user question"
    
    topic: Literal["rock", "politics", "history", "sports"]
    "The topic of the user question. One of 'rock', 'politics', 'history', 'sports' or 'general'."

classifier_function = convert_to_openai_function(TopicClassifier)

llm = ChatOpenAI().bind(functions=[classifier_function], function_call={"name": "TopicClassifier"}) 

parser = PydanticAttrOutputFunctionsParser(pydantic_schema=TopicClassifier, attr_name="topic")

classifier_chain = llm | parser

"""
The classifier_function classifies or categorizes the topic of a user's question into specific categories such as "rock," "politics," "history," or "sports." Here’s how it works in simple terms:

1. Conversion to Function: It converts the TopicClassifier Pydantic class, which is a predefined classification system, into a function that can be easily used with LangChain. This conversion process involves wrapping the class so that it can be integrated and executed within an OpenAI model.

2. Topic Detection: When you input a question, this function analyzes the content of the question to determine which category or topic it belongs to. It looks for keywords or patterns that match specific topics. For example, if the question is about a rock band, the classifier would identify the topic as "rock."

3. utput: The function outputs the identified topic as a simple label, like "rock" or "history." This label is then used by other parts of the LangChain to decide how to handle the question, such as choosing the right template for formulating a response.

In essence, the classifier_function acts as a smart filter that helps the system understand what kind of question is being asked so that it can respond more accurately and relevantly.
"""

from operator import itemgetter

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


final_chain = (
    RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain) 
    | prompt_branch 
    | ChatOpenAI()
    | StrOutputParser()
)

response = final_chain.invoke(
    {"input": "Who was Napoleon Bonaparte?"}
)

print("\n----------\n")

print("Chain with RunnableBranch:")

print("\n----------\n")
print(response)

print("\n----------\n")
```

### Built-in functions para LCEL Runnables: bind y assign

Main built-in LCEL functions for runnables

Contents
- .bind()
- .assign()

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

"""
- The "pipe" operator | is the main element of the LCEL chains.
- The order (left to right) of the elements in a LCEL chain matters.
- An LCEL Chain is a Sequence of Runnables.
"""

chain = prompt | model | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

"""
- All the components of the chain are Runnables.
- When we write chain.invoke() we are using invoke with all the componentes of the chain in an orderly manner:
  - First, we apply .invoke() to the prompt.
  - Then, with the previous output, we apply .invoke() to the model.
  - And finally, with the previous output, we apply .invoke() to the output parser.
"""

print("\n----------\n")

print("Basic LCEL chain:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
Use of .bind() to add arguments to a Runnable in a LCEL Chain
- For example, we can add an argument to stop the model response when it reaches the word "Ronaldo":
"""

chain = prompt | model.bind(stop=["Ronaldo"]) | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("Basic LCEL chain with .bind():")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
Use of .bind() to call an OpenAI Function in a LCEL Chain
"""

functions = [
    {
      "name": "soccerfacts",
      "description": "Curious facts about a soccer player",
      "parameters": {
        "type": "object",
        "properties": {
          "question": {
            "type": "string",
            "description": "The question for the curious facts about a soccer player"
          },
          "answer": {
            "type": "string",
            "description": "The answer to the question"
          }
        },
        "required": ["question", "answer"]
      }
    }
  ]

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "soccerfacts"}, functions= functions)
    | JsonOutputFunctionsParser()
)

response = chain.invoke({"soccer_player": "Mbappe"})

"""
Note: OpenAI API has deprecated functions in favor of tools. See here for more info.
https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_functions_agent/
"""

print("\n----------\n")

"""
Use of .bind() to attach OpenAI tools
Note: In the OpenAI Chat API, functions are now considered a legacy options that is deprecated in favor of tools. If you're creating agents using OpenAI LLM models, you should be using OpenAI Tools rather than OpenAI functions.

While you should generally use the .bind_tools() method for tool-calling models, you can also bind provider-specific args directly if you want lower level control:

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

model = ChatOpenAI(model="gpt-3.5-turbo").bind(tools=tools)
model.invoke("What's the weather in SF, NYC and LA?")
"""

print("Call OpenAI Function in LCEL chain with .bind():")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
The assign() function allows adding keys to a chain
Example: we will create a key name "operation_b" assigned to a custom function with a RunnableLambda.
We will start with a very basic chain with just RunnablePassthrough:
"""

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

chain = RunnableParallel({"original_input": RunnablePassthrough()})

chain.invoke("whatever")
# {'original_input': 'whatever'}
"""
- As you can see, right now this chain is only assigning the user input to the "original_input" variable.
- Let's now add the new key "uppercase" with the assign function.
- In the new "uppercase" key, we will use a RunnableLambda with the custom function named make_uppercase
"""

def make_uppercase(arg):
    return arg["original_input"].upper()

chain = RunnableParallel({"original_input": RunnablePassthrough()}).assign(uppercase=RunnableLambda(make_uppercase))

response = chain.invoke("whatever")
# {'original_input': 'whatever', 'uppercase': 'WHATEVER'}

"""
- As you can see, the output of the chain has now 2 keys: original_input and uppercase.
- In the uppercase key, we can see that the make_uppercase function has been applied to the user input.
"""

print("\n----------\n")

print("Basic LCEL chain with .assign():")

print("\n----------\n")
print(response)

print("\n----------\n")
```

### Combinando LCEL chains: coercion, nesting y fallback


```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Write one brief sentence about {politician}")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke({
    "politician": "JFK"
})

print("\n----------\n")

print("Basic LCEL chain with output parser:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

vectorstore = DocArrayInMemorySearch.from_texts(
    ["AI Accelera has provided Generative AI Training and Consulting Services in more than 100 countries", "Aceleradora AI is the branch of AI Accelera for the Spanish-Speaking market"],
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

get_question_and_retrieve_relevant_docs = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = get_question_and_retrieve_relevant_docs | prompt | model | output_parser

response = chain.invoke("In how many countries has AI Accelera provided services?")

print("\n----------\n")

print("Mid-level LCEL chain with retriever:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    ["AI Accelera has trained more than 10,000 Alumni from all continents and top companies"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-3.5-turbo")

retrieval_chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

response = retrieval_chain.invoke("who are the Alumni of AI Accelera?")

print("\n----------\n")

print("Mid-level LCEL chain with RunnableParallel:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

model = ChatOpenAI(model="gpt-3.5-turbo")

vectorstore = FAISS.from_texts(
    ["AI Accelera has trained more than 3,000 Enterprise Alumni."], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

response = chain.invoke({"question": "How many Enterprise Alumni has trained AI Accelera?", "language": "Pirate English"})

print("\n----------\n")

print("Mid-level LCEL chain with RunnableParallel and itemgetter:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    user_input = RunnablePassthrough(),
    transformed_output = lambda x: x["num"] + 1,
)

response = runnable.invoke({"num": 1})

print("\n----------\n")

print("Mid-level LCEL chain with RunnablePassthrough:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
Coercion: a chain inside another chain
- Remember: almost any component in LangChain (prompts, models, output parsers, etc) can be used as a Runnable.
- Runnables can be chained together using the pipe operator |. The resulting chains of runnables are also runnables themselves
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a sentence about {politician}")

model = ChatOpenAI(model="gpt-3.5-turbo")

chain = prompt | model | StrOutputParser()

response = chain.invoke("Chamberlain")
# 'Chamberlain was a British politician who is best known for his policy of appeasement towards Nazi Germany in the years leading up to World War II.'

print("\n----------\n")

print("Chaining Runnables:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
Coercion: combine a chain with other Runnables to create a new chain.
See how in the composed_chain we are including the previous chain:
"""

from langchain_core.output_parsers import StrOutputParser

historian_prompt = ChatPromptTemplate.from_template("Was {politician} positive for Humanity?")

composed_chain = {"politician": chain} | historian_prompt | model | StrOutputParser()

response = composed_chain.invoke({"politician": "Lincoln"})
# "Yes, Abraham Lincoln is considered to have had a positive impact on humanity. His leadership during the Civil War helped to preserve the unity of the United States and ultimately led to the end of slavery. The Emancipation Proclamation was a significant step towards equality and justice for all individuals, and Lincoln's efforts to abolish slavery have had a lasting impact on society. Additionally, Lincoln's leadership and dedication to upholding the principles of democracy and freedom have inspired generations of Americans and individuals around the world."

print("\n----------\n")

composed_chain.invoke({"politician": "Attila"})
# "Attila the Hun's conquests and actions were not necessarily positive for humanity. He was known for his brutal tactics, including pillaging and destroying cities, and his reign brought suffering and devastation to many people. While he was certainly a powerful and formidable leader, his legacy is more often associated with violence and destruction rather than positive contributions to humanity."

print("Coercion:")

print("\n----------\n")
#print(response)

print("\n----------\n")

composed_chain_with_lambda = (
    chain
    | (lambda input: {"politician": input})
    | historian_prompt
    | model
    | StrOutputParser()
)

response = composed_chain_with_lambda.invoke({"politician": "Robespierre"})

print("\n----------\n")

print("Chain with lambda function:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
Another example: a chain inside another chain
"""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt1 = ChatPromptTemplate.from_template("what is the country {politician} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what continent is the country {country} in? respond in {language}"
)

model = ChatOpenAI()

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"country": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

response = chain2.invoke({"politician": "Miterrand", "language": "French"})
# "Le continent où se trouve la France, dont François Mitterrand était originaire, est l'Europe."

print("\n----------\n")

print("Multiple Chains:")

print("\n----------\n")
#print(response)

print("\n----------\n")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

def russian_lastname_from_dictionary(person):
    return person["name"] + "ovich"

chain = RunnableParallel(
    {
        "soccer_player": RunnablePassthrough() 
        | RunnableLambda(russian_lastname_from_dictionary), 
        "operation_c": RunnablePassthrough()
    }
) | prompt | model | output_parser

response = chain.invoke({
    "name1": "Jordam",
    "name": "Abram"
})

print("\n----------\n")

print("Nested Chains:")

print("\n----------\n")
#print(response)

print("\n----------\n")

"""
Fallback for Chains
- When working with language models, you may often encounter issues from the underlying APIs, whether these be rate limiting or downtime. Therefore, as you go to move your LLM applications into production it becomes more and more important to safeguard against these. That's why LangChain introduced the concept of fallbacks.
- A fallback is an alternative plan that may be used in an emergency.
- Fallbacks can be applied not only on the LLM level but on the whole runnable level. This is important because often times different models require different prompts. So if your call to OpenAI fails, you don't just want to send the same prompt to Anthropic - you probably want to use a different prompt template and send a different version there.
- We can create fallbacks for LCEL chains. Here we do that with two different models: ChatOpenAI (with a bad model name to easily create a chain that will error) and then normal OpenAI (which does not use a chat model). Because OpenAI is NOT a chat model, you likely want a different prompt.
"""

# First let's create a chain with a ChatModel
# We add in a string output parser here so the outputs between the two are the same type
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a funny assistant who always includes a joke in your response",
        ),
        ("human", "Who is the best {sport} player worldwide?"),
    ]
)
# Here we're going to use a bad model name to easily create a chain that will error
chat_model = ChatOpenAI(model="gpt-fake")

bad_chain = chat_prompt | chat_model | StrOutputParser()

# Now lets create a chain with the normal OpenAI model
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

prompt_template = """Instructions: You're a funny assistant who always includes a joke in your response.

Question: Who is the best {sport} player worldwide?"""

prompt = PromptTemplate.from_template(prompt_template)

llm = OpenAI()

good_chain = prompt | llm

# We can now create a final chain which combines the two
chain = bad_chain.with_fallbacks([good_chain])

response = chain.invoke({"sport": "soccer"})
# "\n\nResponse: Well, it depends on who you ask. Some might say Messi, others might say Ronaldo. But I personally think it's my grandma, she can kick a ball farther than anyone I know!"

print("\n----------\n")

print("Fallback for Chains:")

print("\n----------\n")
#print(response)

print("\n----------\n")
```

### LCEL chains y RAG: Un estudio más detallado

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

"""
See below that the prompt we have imported from the hub has 2 variables: "context" and "question".

prompt
# ChatPromptTemplate(input_variables=['context', 'question'], metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])
"""

response = rag_chain.invoke("What is Task Decomposition?")
# 'Task Decomposition is the process of breaking down a task into smaller, more manageable subtasks in order to facilitate the completion of the overall task.'

"""
- This is how this chain works when we invoke it:
  - "What is Task Decomposition?" is passed as unique input.
  - context executes the retriever over the input.
  - format_docs executes the formatter function over the input.
  - The input is assigned to question.
  - the prompt is defined using the previous question and context variables.
  - the model is executed with the previous prompt.

  - the output parser is executed over the response of the model.
Note: what does the previos formatter function do?
The format_docs function takes a list of objects named docs. Each object in this list is expected to have an attribute named page_content, which stores textual content for each document.

The purpose of the function is to extract the page_content from each document in the docs list and then combine these contents into a single string. The contents of different documents are separated by two newline characters (\n\n), which means there will be an empty line between the content of each document in the final string. This formatting choice makes the combined content easier to read by clearly separating the content of different documents.

Here's a breakdown of how the function works:

1. The for doc in docs part iterates over each object in the docs list.
2. For each iteration, doc.page_content accesses the page_content attribute of the current document, which contains its textual content.
3. The join method then takes these pieces of text and concatenates them into a single string, inserting \n\n between each piece to ensure they are separated by a blank line in the final result.

The function ultimately returns this newly formatted single string containing all the document contents, neatly separated by blank lines.
"""

print("\n----------\n")

print("Chains in RAG:")

print("\n----------\n")
print(response)

print("\n----------\n")
```

## LangChain Basics: Memoria. ¿Pueden recordar los LLMs?

https://github.com/AI-LLM-Bootcamp/b405

- ¿Pueden recordar los LLMs?
- Temporary memory (Buffer memory)
- Permanent memory (Chat history)

### Memoria Temporal: buffer memory

Memory
- Save your conversation with the LLM.

Intro
- Ability to store information about past interactions.
- **Most of memory-related functionality in LangChain is marked as beta**. This is for two reasons:
  1. Most functionality are not production ready.
  2. Most functionality work with Legacy chains, not the newer LCEL syntax.
- **The main exception to this is the ChatMessageHistory functionality**. This functionality is largely production ready and does integrate with LCEL.

LangChain documentation on Memory
- See the LangChain documentation page on Memory here. - https://python.langchain.com/v0.1/docs/modules/memory/
- See the LangChain documentation page on how to use ChatMessageHistory with LCEL here. - https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/
- See the LangChain documentation page on the various ChatMessageHistory integrations here. - https://python.langchain.com/v0.1/docs/integrations/memory/

**Buffer Memory**
- ConversationBufferMemory keeps a list of chat messages in a buffer and passes those into the prompt template.
- A buffer refers to a **temporary storage area in memory** used to hold data. Here, the buffer specifically holds chat messages before they are processed or used in some way, such as being passed into a prompt template.
- In simple terms, a buffer is like a waiting room for data. It's a temporary holding spot where data can stay until it's ready to be used or processed. Imagine you're cooking and you chop up vegetables before you need to cook them. Instead of chopping each vegetable right before it goes in the pan, you chop them all at once and put them aside on a cutting board. That cutting board with the chopped vegetables is like a buffer — it holds everything ready for you until you're ready to use it. This way, when you need the vegetables, they are all prepared and ready to go, making the cooking process smoother and faster.
- As you can see in the following example, ConversationBufferMemory was used mostly in the initial versions of LangChain.

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
response = conversation({"question": "hi"})
# > Entering new LLMChain chain...
# Prompt after formatting:
# System: You are a nice chatbot having a conversation with a human.
# Human: hi

# > Finished chain.
# {'question': 'hi',
# 'chat_history': [HumanMessage(content='hi'),
#  AIMessage(content='Hello! How are you doing today?')],
# 'text': 'Hello! How are you doing today?'}

print("\n----------\n")

print("Result from invoking the chain:")

print("\n----------\n")
print(response)

print("\n----------\n")

conversation({"question": "My name is Julio and I have moved 33 times."})

"""
> Entering new LLMChain chain...
Prompt after formatting:
System: You are a nice chatbot having a conversation with a human.
Human: hi
AI: Hello! How are you doing today?
Human: My name is Julio and I have moved 33 times.
> Finished chain.
{'question': 'My name is Julio and I have moved 33 times.',
'chat_history': [HumanMessage(content='hi'),
 AIMessage(content='Hello! How are you doing today?'),
 HumanMessage(content='My name is Julio and I have moved 33 times.'),
 AIMessage(content="Nice to meet you, Julio! Moving 33 times must have given you a lot of interesting experiences. What's the story behind moving so many times?")],
 'text': "Nice to meet you, Julio! Moving 33 times must have given you a lot of interesting experiences. What's the story behind moving so many times?"}
"""

conversation({"question": "If my average moving distance was 100 miles, how many miles took all my moves?"})

"""
> Entering new LLMChain chain...
Prompt after formatting:
System: You are a nice chatbot having a conversation with a human.
Human: hi
AI: Hello! How are you doing today?
Human: My name is Julio and I have moved 33 times.
AI: Nice to meet you, Julio! Moving 33 times must have given you a lot of interesting experiences. What's the story behind moving so many times?
Human: If my average moving distance was 100 miles, how many miles took all my moves?

> Finished chain.
{'question': 'If my average moving distance was 100 miles, how many miles took all my moves?',
 'chat_history': [HumanMessage(content='hi'),
  AIMessage(content='Hello! How are you doing today?'),
  HumanMessage(content='My name is Julio and I have moved 33 times.'),
  AIMessage(content="Nice to meet you, Julio! Moving 33 times must have given you a lot of interesting experiences. What's the story behind moving so many times?"),
  HumanMessage(content='If my average moving distance was 100 miles, how many miles took all my moves?'),
  AIMessage(content="To calculate the total distance you have moved, you can multiply the number of moves (33) by the average distance of each move (100 miles):\n\n33 moves x 100 miles = 3300 miles\n\nSo, all your moves combined would cover a total distance of 3300 miles. That's quite a journey!")],
 'text': "To calculate the total distance you have moved, you can multiply the number of moves (33) by the average distance of each move (100 miles):\n\n33 moves x 100 miles = 3300 miles\n\nSo, all your moves combined would cover a total distance of 3300 miles. That's quite a journey!"}
"""

conversation({"question": "Do you remember my name?"})

"""
> Entering new LLMChain chain...
Prompt after formatting:
System: You are a nice chatbot having a conversation with a human.
Human: hi
AI: Hello! How are you doing today?
Human: My name is Julio and I have moved 33 times.
AI: Nice to meet you, Julio! Moving 33 times must have given you a lot of interesting experiences. What's the story behind moving so many times?
Human: If my average moving distance was 100 miles, how many miles took all my moves?
AI: To calculate the total distance you have moved, you can multiply the number of moves (33) by the average distance of each move (100 miles):

33 moves x 100 miles = 3300 miles

So, all your moves combined would cover a total distance of 3300 miles. That's quite a journey!
Human: Do you remember my name?

> Finished chain.
{'question': 'Do you remember my name?',
 'chat_history': [HumanMessage(content='hi'),
  AIMessage(content='Hello! How are you doing today?'),
  HumanMessage(content='My name is Julio and I have moved 33 times.'),
  AIMessage(content="Nice to meet you, Julio! Moving 33 times must have given you a lot of interesting experiences. What's the story behind moving so many times?"),
  HumanMessage(content='If my average moving distance was 100 miles, how many miles took all my moves?'),
  AIMessage(content="To calculate the total distance you have moved, you can multiply the number of moves (33) by the average distance of each move (100 miles):\n\n33 moves x 100 miles = 3300 miles\n\nSo, all your moves combined would cover a total distance of 3300 miles. That's quite a journey!"),
  HumanMessage(content='Do you remember my name?'),
  AIMessage(content='Yes, Julio! Is there anything else you would like to know or talk about?')],
 'text': 'Yes, Julio! Is there anything else you would like to know or talk about?'}
"""

print(memory.buffer)
"""
[HumanMessage(content='hi'), AIMessage(content='Hello! How are you doing today?'), HumanMessage(content='My name is Julio and I have moved 33 times.'), AIMessage(content="Nice to meet you, Julio! Moving 33 times must have given you a lot of interesting experiences. What's the story behind moving so many times?"), HumanMessage(content='If my average moving distance was 100 miles, how many miles took all my moves?'), AIMessage(content="To calculate the total distance you have moved, you can multiply the number of moves (33) by the average distance of each move (100 miles):\n\n33 moves x 100 miles = 3300 miles\n\nSo, all your moves combined would cover a total distance of 3300 miles. That's quite a journey!"), HumanMessage(content='Do you remember my name?'), AIMessage(content='Yes, Julio! Is there anything else you would like to know or talk about?')]
"""

memory.load_memory_variables({})
"""
{'chat_history': [HumanMessage(content='hi'),
  AIMessage(content='Hello! How are you doing today?'),
  HumanMessage(content='My name is Julio and I have moved 33 times.'),
  AIMessage(content="Nice to meet you, Julio! Moving 33 times must have given you a lot of interesting experiences. What's the story behind moving so many times?"),
  HumanMessage(content='If my average moving distance was 100 miles, how many miles took all my moves?'),
  AIMessage(content="To calculate the total distance you have moved, you can multiply the number of moves (33) by the average distance of each move (100 miles):\n\n33 moves x 100 miles = 3300 miles\n\nSo, all your moves combined would cover a total distance of 3300 miles. That's quite a journey!"),
  HumanMessage(content='Do you remember my name?'),
  AIMessage(content='Yes, Julio! Is there anything else you would like to know or talk about?')]}
"""
```

Conversation Buffer Window Memory

Similar to the previous one, but **you can limit the number of conversational exchanges stored in memory**. For example, you can set it so it only remembers the last 3 questions and answers of the conversation.

```
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
window_memory = ConversationBufferWindowMemory(k=3)
conversation_window = ConversationChain(
    llm=llm, 
    memory = window_memory,
    verbose=True
)
"""
/Users/juliocolomer/Library/Caches/pypoetry/virtualenvs/b405-Pby3ZuTj-py3.11/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:139: LangChainDeprecationWarning: The class `ConversationChain` was deprecated in LangChain 0.2.7 and will be removed in 1.0. Use RunnableWithMessageHistory: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html instead.
  warn_deprecated(
"""
conversation_window({"input": "Hi, my name is Julio"})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi, my name is Julio
AI:

> Finished chain.
{'input': 'Hi, my name is Julio',
 'history': '',
 'response': "Hello Julio! It's nice to meet you. How can I assist you today?"}
"""

conversation_window({"input": "My favorite color is blue"})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi, my name is Julio
AI: Hello Julio! It's nice to meet you. How can I assist you today?
Human: My favorite color is blue
AI:

> Finished chain.
{'input': 'My favorite color is blue',
 'history': "Human: Hi, my name is Julio\nAI: Hello Julio! It's nice to meet you. How can I assist you today?",
 'response': "Blue is a great choice! Did you know that blue is often associated with calmness and tranquility? It's also a popular color for clothing and interior design. Is there anything else you'd like to share or ask?"}
"""

conversation_window({"input": "My favorite animals are dogs"})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi, my name is Julio
AI: Hello Julio! It's nice to meet you. How can I assist you today?
Human: My favorite color is blue
AI: Blue is a great choice! Did you know that blue is often associated with calmness and tranquility? It's also a popular color for clothing and interior design. Is there anything else you'd like to share or ask?
Human: My favorite animals are dogs
AI:

> Finished chain.
{'input': 'My favorite animals are dogs',
 'history': "Human: Hi, my name is Julio\nAI: Hello Julio! It's nice to meet you. How can I assist you today?\nHuman: My favorite color is blue\nAI: Blue is a great choice! Did you know that blue is often associated with calmness and tranquility? It's also a popular color for clothing and interior design. Is there anything else you'd like to share or ask?",
 'response': 'Dogs are wonderful companions! They are known for their loyalty, playful nature, and ability to form strong bonds with their owners. There are so many different breeds of dogs, each with their own unique characteristics and personalities. Do you have a favorite breed of dog, Julio?'}
"""

conversation_window({"input": "I like to drive a vespa scooter in the city"})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi, my name is Julio
AI: Hello Julio! It's nice to meet you. How can I assist you today?
Human: My favorite color is blue
AI: Blue is a great choice! Did you know that blue is often associated with calmness and tranquility? It's also a popular color for clothing and interior design. Is there anything else you'd like to share or ask?
Human: My favorite animals are dogs
AI: Dogs are wonderful companions! They are known for their loyalty, playful nature, and ability to form strong bonds with their owners. There are so many different breeds of dogs, each with their own unique characteristics and personalities. Do you have a favorite breed of dog, Julio?
Human: I like to drive a vespa scooter in the city
AI:

> Finished chain.
{'input': 'I like to drive a vespa scooter in the city',
 'history': "Human: Hi, my name is Julio\nAI: Hello Julio! It's nice to meet you. How can I assist you today?\nHuman: My favorite color is blue\nAI: Blue is a great choice! Did you know that blue is often associated with calmness and tranquility? It's also a popular color for clothing and interior design. Is there anything else you'd like to share or ask?\nHuman: My favorite animals are dogs\nAI: Dogs are wonderful companions! They are known for their loyalty, playful nature, and ability to form strong bonds with their owners. There are so many different breeds of dogs, each with their own unique characteristics and personalities. Do you have a favorite breed of dog, Julio?",
 'response': 'Vespa scooters are a stylish and convenient way to get around in the city! They are known for their retro design, compact size, and ease of maneuverability in urban areas. Riding a Vespa scooter can be a fun and efficient way to navigate through traffic and explore the city. Have you customized your Vespa in any way, Julio?'}
"""

conversation_window({"input": "My favorite city is San Francisco"})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: My favorite color is blue
AI: Blue is a great choice! Did you know that blue is often associated with calmness and tranquility? It's also a popular color for clothing and interior design. Is there anything else you'd like to share or ask?
Human: My favorite animals are dogs
AI: Dogs are wonderful companions! They are known for their loyalty, playful nature, and ability to form strong bonds with their owners. There are so many different breeds of dogs, each with their own unique characteristics and personalities. Do you have a favorite breed of dog, Julio?
Human: I like to drive a vespa scooter in the city
AI: Vespa scooters are a stylish and convenient way to get around in the city! They are known for their retro design, compact size, and ease of maneuverability in urban areas. Riding a Vespa scooter can be a fun and efficient way to navigate through traffic and explore the city. Have you customized your Vespa in any way, Julio?
Human: My favorite city is San Francisco
AI:

> Finished chain.
{'input': 'My favorite city is San Francisco',
 'history': "Human: My favorite color is blue\nAI: Blue is a great choice! Did you know that blue is often associated with calmness and tranquility? It's also a popular color for clothing and interior design. Is there anything else you'd like to share or ask?\nHuman: My favorite animals are dogs\nAI: Dogs are wonderful companions! They are known for their loyalty, playful nature, and ability to form strong bonds with their owners. There are so many different breeds of dogs, each with their own unique characteristics and personalities. Do you have a favorite breed of dog, Julio?\nHuman: I like to drive a vespa scooter in the city\nAI: Vespa scooters are a stylish and convenient way to get around in the city! They are known for their retro design, compact size, and ease of maneuverability in urban areas. Riding a Vespa scooter can be a fun and efficient way to navigate through traffic and explore the city. Have you customized your Vespa in any way, Julio?",
 'response': "San Francisco is a beautiful city with so much to offer! It's known for its iconic landmarks like the Golden Gate Bridge and Alcatraz Island, as well as its diverse neighborhoods, vibrant food scene, and rich cultural history. The city's hilly terrain and stunning views of the bay make it a unique and charming place to visit or live. Have you had the chance to explore all the different neighborhoods in San Francisco, Julio?"}
"""

conversation_window({"input": "My favorite season is summer"})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: My favorite animals are dogs
AI: Dogs are wonderful companions! They are known for their loyalty, playful nature, and ability to form strong bonds with their owners. There are so many different breeds of dogs, each with their own unique characteristics and personalities. Do you have a favorite breed of dog, Julio?
Human: I like to drive a vespa scooter in the city
AI: Vespa scooters are a stylish and convenient way to get around in the city! They are known for their retro design, compact size, and ease of maneuverability in urban areas. Riding a Vespa scooter can be a fun and efficient way to navigate through traffic and explore the city. Have you customized your Vespa in any way, Julio?
Human: My favorite city is San Francisco
AI: San Francisco is a beautiful city with so much to offer! It's known for its iconic landmarks like the Golden Gate Bridge and Alcatraz Island, as well as its diverse neighborhoods, vibrant food scene, and rich cultural history. The city's hilly terrain and stunning views of the bay make it a unique and charming place to visit or live. Have you had the chance to explore all the different neighborhoods in San Francisco, Julio?
Human: My favorite season is summer
AI:

> Finished chain.
{'input': 'My favorite season is summer',
 'history': "Human: My favorite animals are dogs\nAI: Dogs are wonderful companions! They are known for their loyalty, playful nature, and ability to form strong bonds with their owners. There are so many different breeds of dogs, each with their own unique characteristics and personalities. Do you have a favorite breed of dog, Julio?\nHuman: I like to drive a vespa scooter in the city\nAI: Vespa scooters are a stylish and convenient way to get around in the city! They are known for their retro design, compact size, and ease of maneuverability in urban areas. Riding a Vespa scooter can be a fun and efficient way to navigate through traffic and explore the city. Have you customized your Vespa in any way, Julio?\nHuman: My favorite city is San Francisco\nAI: San Francisco is a beautiful city with so much to offer! It's known for its iconic landmarks like the Golden Gate Bridge and Alcatraz Island, as well as its diverse neighborhoods, vibrant food scene, and rich cultural history. The city's hilly terrain and stunning views of the bay make it a unique and charming place to visit or live. Have you had the chance to explore all the different neighborhoods in San Francisco, Julio?",
 'response': "Summer is a popular season for many people! It's known for warm weather, longer days, and opportunities for outdoor activities like swimming, hiking, and barbecues. The bright sunshine and clear skies of summer can create a cheerful and energetic atmosphere. Whether you enjoy beach days, summer festivals, or simply relaxing in the sun, there's something for everyone to love about the summer season. What are some of your favorite summer activities, Julio?"}
"""

conversation_window({"input": "What is my favorite color?"})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: I like to drive a vespa scooter in the city
AI: Vespa scooters are a stylish and convenient way to get around in the city! They are known for their retro design, compact size, and ease of maneuverability in urban areas. Riding a Vespa scooter can be a fun and efficient way to navigate through traffic and explore the city. Have you customized your Vespa in any way, Julio?
Human: My favorite city is San Francisco
AI: San Francisco is a beautiful city with so much to offer! It's known for its iconic landmarks like the Golden Gate Bridge and Alcatraz Island, as well as its diverse neighborhoods, vibrant food scene, and rich cultural history. The city's hilly terrain and stunning views of the bay make it a unique and charming place to visit or live. Have you had the chance to explore all the different neighborhoods in San Francisco, Julio?
Human: My favorite season is summer
AI: Summer is a popular season for many people! It's known for warm weather, longer days, and opportunities for outdoor activities like swimming, hiking, and barbecues. The bright sunshine and clear skies of summer can create a cheerful and energetic atmosphere. Whether you enjoy beach days, summer festivals, or simply relaxing in the sun, there's something for everyone to love about the summer season. What are some of your favorite summer activities, Julio?
Human: What is my favorite color?
AI:

> Finished chain.
{'input': 'What is my favorite color?',
 'history': "Human: I like to drive a vespa scooter in the city\nAI: Vespa scooters are a stylish and convenient way to get around in the city! They are known for their retro design, compact size, and ease of maneuverability in urban areas. Riding a Vespa scooter can be a fun and efficient way to navigate through traffic and explore the city. Have you customized your Vespa in any way, Julio?\nHuman: My favorite city is San Francisco\nAI: San Francisco is a beautiful city with so much to offer! It's known for its iconic landmarks like the Golden Gate Bridge and Alcatraz Island, as well as its diverse neighborhoods, vibrant food scene, and rich cultural history. The city's hilly terrain and stunning views of the bay make it a unique and charming place to visit or live. Have you had the chance to explore all the different neighborhoods in San Francisco, Julio?\nHuman: My favorite season is summer\nAI: Summer is a popular season for many people! It's known for warm weather, longer days, and opportunities for outdoor activities like swimming, hiking, and barbecues. The bright sunshine and clear skies of summer can create a cheerful and energetic atmosphere. Whether you enjoy beach days, summer festivals, or simply relaxing in the sun, there's something for everyone to love about the summer season. What are some of your favorite summer activities, Julio?",
 'response': "I'm sorry, Julio, but I do not have that information. Can you please tell me what your favorite color is?"}
"""

conversation_window({"input": "My favorite city is San Francisco"})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: My favorite city is San Francisco
AI: San Francisco is a beautiful city with so much to offer! It's known for its iconic landmarks like the Golden Gate Bridge and Alcatraz Island, as well as its diverse neighborhoods, vibrant food scene, and rich cultural history. The city's hilly terrain and stunning views of the bay make it a unique and charming place to visit or live. Have you had the chance to explore all the different neighborhoods in San Francisco, Julio?
Human: My favorite season is summer
AI: Summer is a popular season for many people! It's known for warm weather, longer days, and opportunities for outdoor activities like swimming, hiking, and barbecues. The bright sunshine and clear skies of summer can create a cheerful and energetic atmosphere. Whether you enjoy beach days, summer festivals, or simply relaxing in the sun, there's something for everyone to love about the summer season. What are some of your favorite summer activities, Julio?
Human: What is my favorite color?
AI: I'm sorry, Julio, but I do not have that information. Can you please tell me what your favorite color is?
Human: My favorite city is San Francisco
AI:

> Finished chain.
{'input': 'My favorite city is San Francisco',
 'history': "Human: My favorite city is San Francisco\nAI: San Francisco is a beautiful city with so much to offer! It's known for its iconic landmarks like the Golden Gate Bridge and Alcatraz Island, as well as its diverse neighborhoods, vibrant food scene, and rich cultural history. The city's hilly terrain and stunning views of the bay make it a unique and charming place to visit or live. Have you had the chance to explore all the different neighborhoods in San Francisco, Julio?\nHuman: My favorite season is summer\nAI: Summer is a popular season for many people! It's known for warm weather, longer days, and opportunities for outdoor activities like swimming, hiking, and barbecues. The bright sunshine and clear skies of summer can create a cheerful and energetic atmosphere. Whether you enjoy beach days, summer festivals, or simply relaxing in the sun, there's something for everyone to love about the summer season. What are some of your favorite summer activities, Julio?\nHuman: What is my favorite color?\nAI: I'm sorry, Julio, but I do not have that information. Can you please tell me what your favorite color is?",
 'response': "San Francisco is a beautiful city with so much to offer! It's known for its iconic landmarks like the Golden Gate Bridge and Alcatraz Island, as well as its diverse neighborhoods, vibrant food scene, and rich cultural history. The city's hilly terrain and stunning views of the bay make it a unique and charming place to visit or live. Have you had the chance to explore all the different neighborhoods in San Francisco, Julio?"}
"""
```

Conversation Token Buffer Memory

Similar to the previous one, but this time you can limit the number of tokens stored in memory.

If you are using the pre-loaded poetry shell, you do not need to install the following package because it is already pre-loaded for you:
```python
#!pip install tiktoken
from langchain.memory import ConversationTokenBufferMemory
token_memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=50
)
conversation_token = ConversationChain(
    llm=llm, 
    memory = token_memory,
    verbose=True
)
```

Conversation Summary Memory

Stores a summary of the previous conversational exchanges.

```python
from langchain.memory import ConversationSummaryBufferMemory
summary_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100
)
conversation_summary = ConversationChain(
    llm=llm, 
    memory = summary_memory,
    verbose=True
)
conversation_summary({"input": """Kurt Cobain dropped out of high school, 
then worked there as a janitor Even though he was by all accounts a slob, 
Kurt Cobain worked as a janitor at Weatherwax High School, not long after 
dropping out of that very school. The dancing janitor in the 
"Smells Like Teen Spirit" music video was an inside joke for 
those who knew of Cobain's old job.
"""})

"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Kurt Cobain dropped out of high school, 
then worked there as a janitor Even though he was by all accounts a slob, 
Kurt Cobain worked as a janitor at Weatherwax High School, not long after 
dropping out of that very school. The dancing janitor in the 
"Smells Like Teen Spirit" music video was an inside joke for 
those who knew of Cobain's old job.

AI:

> Finished chain.
{'input': 'Kurt Cobain dropped out of high school, \nthen worked there as a janitor Even though he was by all accounts a slob, \nKurt Cobain worked as a janitor at Weatherwax High School, not long after \ndropping out of that very school. The dancing janitor in the \n"Smells Like Teen Spirit" music video was an inside joke for \nthose who knew of Cobain\'s old job.\n',
 'history': '',
 'response': 'That\'s correct! Kurt Cobain did in fact drop out of high school and later worked as a janitor at Weatherwax High School. The dancing janitor in the "Smells Like Teen Spirit" music video was indeed a nod to his past job. It\'s interesting how these little details can add layers of meaning to his work, don\'t you think?'}
"""

conversation_summary({"input": """
There were at least five different drummers in the band 
before Dave Grohl. Cobain and Novoselic were always members 
of Nirvana—formerly known as Skid Row, Pen Cap Chew, Bliss, 
and Ted Ed Fred—but finding a permanent drummer proved to be 
even harder than coming up with a decent band name. In the 
beginning, there was trivia answer Aaron Burckhard, who pissed 
off Cobain by getting Kurt's car impounded after being arrested 
for fighting with a police officer. Then there was Melvins 
drummer Dale Crover, who pounded the skins for Cobain and 
Novoselic on their first demo tape before moving to San Francisco. 
Next came Dave Foster, who got arrested for assaulting the 
son of the mayor of Cosmopolis, Washington. Burckhard briefly 
returned before announcing he was too hungover to practice one day. 
Then a mutual friend introduced Cobain and Novoselic to Chad 
Channing, who hung around for two years before the group's 
co-founders decided he wasn't cutting it anymore. Mudhoney 
drummer Dan Peters played on the "Sliver" single.
"""})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
System: The human mentions that Kurt Cobain dropped out of high school and then worked as a janitor there. Despite being known as a slob, Cobain worked as a janitor at Weatherwax High School after dropping out. The dancing janitor in the "Smells Like Teen Spirit" music video was an inside joke for those aware of Cobain's previous job.
AI: That's correct! Kurt Cobain did in fact drop out of high school and later worked as a janitor at Weatherwax High School. The dancing janitor in the "Smells Like Teen Spirit" music video was indeed a nod to his past job. It's interesting how these little details can add layers of meaning to his work, don't you think?
Human: 
There were at least five different drummers in the band 
before Dave Grohl. Cobain and Novoselic were always members 
of Nirvana—formerly known as Skid Row, Pen Cap Chew, Bliss, 
and Ted Ed Fred—but finding a permanent drummer proved to be 
even harder than coming up with a decent band name. In the 
beginning, there was trivia answer Aaron Burckhard, who pissed 
off Cobain by getting Kurt's car impounded after being arrested 
for fighting with a police officer. Then there was Melvins 
drummer Dale Crover, who pounded the skins for Cobain and 
Novoselic on their first demo tape before moving to San Francisco. 
Next came Dave Foster, who got arrested for assaulting the 
son of the mayor of Cosmopolis, Washington. Burckhard briefly 
returned before announcing he was too hungover to practice one day. 
Then a mutual friend introduced Cobain and Novoselic to Chad 
Channing, who hung around for two years before the group's 
co-founders decided he wasn't cutting it anymore. Mudhoney 
drummer Dan Peters played on the "Sliver" single.

AI:

> Finished chain.
{'input': '\nThere were at least five different drummers in the band \nbefore Dave Grohl. Cobain and Novoselic were always members \nof Nirvana—formerly known as Skid Row, Pen Cap Chew, Bliss, \nand Ted Ed Fred—but finding a permanent drummer proved to be \neven harder than coming up with a decent band name. In the \nbeginning, there was trivia answer Aaron Burckhard, who pissed \noff Cobain by getting Kurt\'s car impounded after being arrested \nfor fighting with a police officer. Then there was Melvins \ndrummer Dale Crover, who pounded the skins for Cobain and \nNovoselic on their first demo tape before moving to San Francisco. \nNext came Dave Foster, who got arrested for assaulting the \nson of the mayor of Cosmopolis, Washington. Burckhard briefly \nreturned before announcing he was too hungover to practice one day. \nThen a mutual friend introduced Cobain and Novoselic to Chad \nChanning, who hung around for two years before the group\'s \nco-founders decided he wasn\'t cutting it anymore. Mudhoney \ndrummer Dan Peters played on the "Sliver" single.\n',
 'history': 'System: The human mentions that Kurt Cobain dropped out of high school and then worked as a janitor there. Despite being known as a slob, Cobain worked as a janitor at Weatherwax High School after dropping out. The dancing janitor in the "Smells Like Teen Spirit" music video was an inside joke for those aware of Cobain\'s previous job.\nAI: That\'s correct! Kurt Cobain did in fact drop out of high school and later worked as a janitor at Weatherwax High School. The dancing janitor in the "Smells Like Teen Spirit" music video was indeed a nod to his past job. It\'s interesting how these little details can add layers of meaning to his work, don\'t you think?',
 'response': 'Wow, that\'s a lot of drummers Nirvana went through before Dave Grohl joined the band! It must have been frustrating for Cobain and Novoselic to constantly have to find new drummers. It\'s interesting to hear about the different personalities and incidents that led to each drummer\'s departure. It\'s amazing how the right drummer can really make or break a band. And it\'s cool that Dan Peters from Mudhoney played on the "Sliver" single. It\'s fascinating to learn about the history and evolution of the band\'s lineup.'}
"""

conversation_summary({"input": """
Back in Washington, Crover performed with Cobain and Novoselic 
on a seven date tour with Sonic Youth in August 1990, before 
Dave Grohl's band Scream broke up and Melvins frontman Buzz 
Osbourne introduced Grohl to Cobain and Novoselic, ending the 
vicious cycle of rotating drummers.
"""})
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
System: The human mentions Kurt Cobain's journey with different drummers before Dave Grohl joined Nirvana. From Aaron Burckhard to Chad Channing and even a brief stint with Dan Peters from Mudhoney, the band went through a series of lineup changes. Each drummer brought their own unique story and challenges, ultimately shaping the band's sound and trajectory.
Human: 
Back in Washington, Crover performed with Cobain and Novoselic 
on a seven date tour with Sonic Youth in August 1990, before 
Dave Grohl's band Scream broke up and Melvins frontman Buzz 
Osbourne introduced Grohl to Cobain and Novoselic, ending the 
vicious cycle of rotating drummers.

AI:

> Finished chain.
{'input': "\nBack in Washington, Crover performed with Cobain and Novoselic \non a seven date tour with Sonic Youth in August 1990, before \nDave Grohl's band Scream broke up and Melvins frontman Buzz \nOsbourne introduced Grohl to Cobain and Novoselic, ending the \nvicious cycle of rotating drummers.\n",
 'history': "System: The human mentions Kurt Cobain's journey with different drummers before Dave Grohl joined Nirvana. From Aaron Burckhard to Chad Channing and even a brief stint with Dan Peters from Mudhoney, the band went through a series of lineup changes. Each drummer brought their own unique story and challenges, ultimately shaping the band's sound and trajectory.",
 'response': "Yes, that's correct! Dale Crover from the Melvins did indeed perform with Kurt Cobain and Krist Novoselic on that tour with Sonic Youth in August 1990. It's fascinating how the connections between different bands and musicians ultimately led to Dave Grohl joining Nirvana and solidifying the lineup. The music scene in Washington at that time was really interconnected and influential in shaping the sound of bands like Nirvana."}
"""

print(summary_memory.buffer)
"""
System: The human mentions Kurt Cobain's journey with different drummers before Dave Grohl joined Nirvana, including Aaron Burckhard, Chad Channing, and Dan Peters. Buzz Osbourne introduced Grohl to Cobain and Novoselic after a tour with Sonic Youth, ending the cycle of rotating drummers and solidifying the band's lineup.
AI: Yes, that's correct! Dale Crover from the Melvins did indeed perform with Kurt Cobain and Krist Novoselic on that tour with Sonic Youth in August 1990. It's fascinating how the connections between different bands and musicians ultimately led to Dave Grohl joining Nirvana and solidifying the lineup. The music scene in Washington at that time was really interconnected and influential in shaping the sound of bands like Nirvana.
"""

summary_memory.load_memory_variables({})
"""
{'history': "System: The human mentions Kurt Cobain's journey with different drummers before Dave Grohl joined Nirvana, including Aaron Burckhard, Chad Channing, and Dan Peters. Buzz Osbourne introduced Grohl to Cobain and Novoselic after a tour with Sonic Youth, ending the cycle of rotating drummers and solidifying the band's lineup.\nAI: Yes, that's correct! Dale Crover from the Melvins did indeed perform with Kurt Cobain and Krist Novoselic on that tour with Sonic Youth in August 1990. It's fascinating how the connections between different bands and musicians ultimately led to Dave Grohl joining Nirvana and solidifying the lineup. The music scene in Washington at that time was really interconnected and influential in shaping the sound of bands like Nirvana."}
"""
```

### Chat Message History

Chat Message History
- Save list of chat messages and then fetch them all.

Intro
- Ability to store information about past interactions.
- Most of memory-related functionality in LangChain is marked as beta. This is for two reasons:
  1. Most functionality are not production ready.
  2. Most functionality work with Legacy chains, not the newer LCEL syntax.
- The main exception to this is the ChatMessageHistory functionality. This functionality is largely production ready and does integrate with LCEL.

LangChain documentation on Memory
- See the LangChain documentation page on Memory here. - https://python.langchain.com/v0.1/docs/modules/memory/
- See the LangChain documentation page on how to use ChatMessageHistory with LCEL here. - https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/
- See the LangChain documentation page on the various ChatMessageHistory integrations here. - https://python.langchain.com/v0.1/docs/integrations/memory/

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# ChatMessageHistory
# ChatMessageHistory provides convenience methods for saving HumanMessages, AIMessages, and then fetching them all.
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")

my_chat_memory = history.messages
# [HumanMessage(content='hi!'), AIMessage(content='whats up?')]

print("\n----------\n")

print("Chat Memory:")

print("\n----------\n")
print(my_chat_memory)

print("\n----------\n")
```

## LangChain Basics: El ecosistema LangChain


