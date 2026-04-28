# LegalMind RAG — Naive RAG for Legal Document Q&A

> **Use Case**: A production-grade Naive RAG system for a law firm's internal knowledge base. Lawyers and paralegals ask natural-language questions against thousands of contracts, case files, and regulatory documents — and get cited, source-grounded answers in seconds.

---

## Use Case: LegalMind Q&A System

**Problem**: A mid-size law firm has 50,000+ documents (contracts, briefs, case files, regulatory memos). Associates spend hours manually searching for precedents. Senior partners need instant answers during client calls.

**Solution**: Naive RAG pipeline that ingests all legal documents, embeds them into a vector store, and answers questions with cited source passages — giving lawyers GPT-quality answers grounded in their own documents.

**Key Personas**:
- 🧑‍⚖️ **Associate Lawyer** — searches for precedents, contract clauses
- 👩‍💼 **Paralegal** — extracts facts from large case files
- 🏛️ **Partner** — quick Q&A during client calls

---

## System Architecture

### 1. Full Ecosystem Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '15px', 'fontFamily': 'arial'}}}%%
graph TB
    classDef clientStyle   fill:#2563EB,stroke:#000,stroke-width:3px,color:#fff
    classDef gatewayStyle  fill:#D97706,stroke:#000,stroke-width:3px,color:#fff
    classDef ragStyle      fill:#16A34A,stroke:#000,stroke-width:3px,color:#fff
    classDef storageStyle  fill:#7C3AED,stroke:#000,stroke-width:3px,color:#fff
    classDef obsStyle      fill:#DC2626,stroke:#000,stroke-width:3px,color:#fff
    classDef infraStyle    fill:#0891B2,stroke:#000,stroke-width:3px,color:#fff

    subgraph CLIENT["🖥️  Client Layer"]
        WEB["Web App<br/>(React + TypeScript)"]
        API_CLIENT["REST API Client<br/>(Axios)"]
        SLACK["Slack Bot<br/>(Optional Integration)"]
    end

    subgraph GATEWAY["🔀  API Gateway Layer"]
        NGINX["Nginx Reverse Proxy<br/>(Rate Limiting + SSL)"]
        FASTAPI["FastAPI Application<br/>(Python 3.11)"]
        AUTH["JWT Auth Middleware<br/>(Role-Based Access)"]
    end

    subgraph RAG_CORE["⚙️  RAG Core Engine"]
        INGEST["Document Ingestion<br/>Service"]
        CHUNK["Text Chunker<br/>(Recursive + Semantic)"]
        EMBED["Embedding Service<br/>(OpenAI Ada-002)"]
        RETRIEVE["Retrieval Engine<br/>(Cosine Similarity)"]
        GENERATE["Generation Service<br/>(GPT-4o)"]
        RERANK["Post-Retrieval<br/>Reranker (Optional)"]
    end

    subgraph STORAGE["🗄️  Storage Layer"]
        VECTORDB["Vector Store<br/>(ChromaDB / Pinecone)"]
        DOCDB["Document Store<br/>(PostgreSQL + pgvector)"]
        CACHE["Response Cache<br/>(Redis)"]
        S3["Object Storage<br/>(S3 / MinIO)"]
    end

    subgraph OBSERVABILITY["📊  Observability Stack"]
        LOGS["Structured Logging<br/>(Loguru + JSON)"]
        METRICS["Metrics<br/>(Prometheus + Grafana)"]
        TRACES["Tracing<br/>(OpenTelemetry)"]
    end

    subgraph INFRA["🏗️  Infrastructure"]
        DOCKER["Docker Compose<br/>(Local Dev)"]
        K8S["Kubernetes<br/>(Production)"]
        CI["CI/CD Pipeline<br/>(GitHub Actions)"]
    end

    WEB        --> NGINX
    API_CLIENT --> NGINX
    SLACK      --> NGINX
    NGINX      --> FASTAPI
    FASTAPI    --> AUTH
    AUTH       --> INGEST
    AUTH       --> RETRIEVE

    INGEST  --> CHUNK
    CHUNK   --> EMBED
    EMBED   --> VECTORDB
    INGEST  --> S3
    INGEST  --> DOCDB

    RETRIEVE --> VECTORDB
    RETRIEVE --> RERANK
    RERANK   --> GENERATE
    RETRIEVE --> CACHE
    GENERATE --> CACHE

    FASTAPI --> LOGS
    FASTAPI --> METRICS
    FASTAPI --> TRACES

    DOCKER --> FASTAPI
    K8S    --> FASTAPI
    CI     --> DOCKER

    class WEB,API_CLIENT,SLACK clientStyle
    class NGINX,FASTAPI,AUTH gatewayStyle
    class INGEST,CHUNK,EMBED,RETRIEVE,GENERATE,RERANK ragStyle
    class VECTORDB,DOCDB,CACHE,S3 storageStyle
    class LOGS,METRICS,TRACES obsStyle
    class DOCKER,K8S,CI infraStyle
```

---

### 2. Naive RAG Pipeline Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '15px', 'fontFamily': 'arial'}}}%%
flowchart TD
    classDef offlineStyle fill:#1D4ED8,stroke:#000,stroke-width:3px,color:#fff
    classDef onlineStyle  fill:#15803D,stroke:#000,stroke-width:3px,color:#fff
    classDef edgeStyle    fill:#B45309,stroke:#000,stroke-width:3px,color:#fff

    subgraph OFFLINE["📥  Offline Indexing Pipeline"]
        direction TB
        DOC["📄 Raw Documents<br/>.pdf  .docx  .txt"]
        LOAD["📂 Document Loader<br/>(LangChain Loaders)"]
        CLEAN["🧹 Text Cleaner<br/>(strip headers / footers)"]
        SPLITTER["✂️ Recursive Text Splitter<br/>chunk_size=512  overlap=64"]
        EMBEDDER["🔢 Embedding Model<br/>text-embedding-ada-002"]
        STORE["💾 Vector Store Upsert<br/>(ChromaDB)"]

        DOC --> LOAD --> CLEAN --> SPLITTER --> EMBEDDER --> STORE
    end

    subgraph ONLINE["🟢  Online Query Pipeline"]
        direction TB
        Q["💬 User Query<br/>'What is the penalty clause?'"]
        QE["🔢 Query Embedder<br/>same model · same dim"]
        VS["🔍 Vector Similarity Search<br/>top-k=5 · cosine distance"]
        CHUNKS["📑 Retrieved Chunks<br/>+ metadata + scores"]
        PROMPT["📝 Prompt Builder<br/>system + context + query"]
        LLM["🤖 LLM Generation<br/>GPT-4o with citations"]
        ANS["✅ Answer + Sources<br/>+ confidence score"]

        Q --> QE --> VS --> CHUNKS --> PROMPT --> LLM --> ANS
    end

    STORE -.->|persisted index| VS

    class DOC,LOAD,CLEAN,SPLITTER,EMBEDDER,STORE offlineStyle
    class Q,QE,VS,CHUNKS,PROMPT,LLM,ANS onlineStyle
```

---

### 3. RAG Data Flow Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '14px', 'fontFamily': 'arial', 'actorBkg': '#1E40AF', 'actorTextColor': '#ffffff', 'actorBorderColor': '#000000', 'noteBkgColor': '#FEF3C7', 'noteBorderColor': '#000000', 'noteTextColor': '#000000', 'activationBkgColor': '#DBEAFE', 'activationBorderColor': '#1D4ED8', 'sequenceNumberColor': '#000000'}}}%%
sequenceDiagram
    autonumber
    actor U  as 👤 User
    participant API   as 🔀 FastAPI
    participant Cache as ⚡ Redis Cache
    participant Embed as 🔢 Embedding Svc
    participant VS    as 🗄️ Vector Store
    participant LLM   as 🤖 GPT-4o

    U  ->> API:   POST /query {question, filters}
    API ->> Cache: GET cached_response(hash(question))

    alt ✅ Cache Hit
        Cache -->> API: cached answer
        API   -->> U:   Answer (served from cache)
    else ❌ Cache Miss
        API   ->>  Embed: embed(question)
        Embed -->> API:   query_vector [1536-dim]
        API   ->>  VS:    similarity_search(vector, top_k=5)
        VS    -->> API:   [(chunk_text, metadata, score)]
        Note over API,LLM: Build prompt with retrieved context
        API   ->>  LLM:   generate(system_prompt + context + question)
        LLM   -->> API:   answer + citations
        API   ->>  Cache: SET response (TTL=1 hr)
        API   -->> U:     Answer + Sources + Confidence
    end
```

---

### 4. Document Ingestion Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '15px', 'fontFamily': 'arial'}}}%%
flowchart LR
    classDef inputStyle   fill:#0369A1,stroke:#000,stroke-width:3px,color:#fff
    classDef processStyle fill:#7C3AED,stroke:#000,stroke-width:3px,color:#fff
    classDef outputStyle  fill:#15803D,stroke:#000,stroke-width:3px,color:#fff

    subgraph INPUT["📂  Input Sources"]
        PDF["📄 PDF Files"]
        DOCX["📝 Word Docs"]
        TXT["🔤 Plain Text"]
        URL["🌐 Web URLs"]
    end

    subgraph PROCESS["⚙️  Processing Steps"]
        direction TB
        P1["1️⃣  Load + Extract Text"]
        P2["2️⃣  Clean & Normalize"]
        P3["3️⃣  Detect Language"]
        P4["4️⃣  Split into Chunks<br/>512 tokens · 64 overlap"]
        P5["5️⃣  Generate Embeddings<br/>1536-dim float32"]
        P6["6️⃣  Attach Metadata<br/>{source, page, date}"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph OUTPUT["🗄️  Storage Destinations"]
        VS2["🔍 Vector Store<br/>(embeddings + chunk_id)"]
        DB["🐘 PostgreSQL<br/>(full text + metadata)"]
        OBJ["☁️ Object Store<br/>(original files)"]
    end

    PDF  --> P1
    DOCX --> P1
    TXT  --> P1
    URL  --> P1

    P6 --> VS2
    P6 --> DB
    P1 --> OBJ

    class PDF,DOCX,TXT,URL inputStyle
    class P1,P2,P3,P4,P5,P6 processStyle
    class VS2,DB,OBJ outputStyle
```

---

### 5. Component Dependency Graph

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '15px', 'fontFamily': 'arial'}}}%%
graph LR
    classDef ingStyle  fill:#B45309,stroke:#000,stroke-width:3px,color:#fff
    classDef retStyle  fill:#1D4ED8,stroke:#000,stroke-width:3px,color:#fff
    classDef genStyle  fill:#15803D,stroke:#000,stroke-width:3px,color:#fff
    classDef apiStyle  fill:#DC2626,stroke:#000,stroke-width:3px,color:#fff
    classDef utlStyle  fill:#6B7280,stroke:#000,stroke-width:3px,color:#fff

    subgraph CORE["📦  src/"]
        ING["📥 ingestion/<br/>loader.py<br/>cleaner.py<br/>chunker.py"]
        RET["🔍 retrieval/<br/>vector_store.py<br/>retriever.py<br/>reranker.py"]
        GEN["🤖 generation/<br/>prompt_builder.py<br/>llm_client.py<br/>citation.py"]
        API2["🔀 api/<br/>main.py<br/>routes.py<br/>schemas.py<br/>middleware.py"]
        UTL["🛠️ utils/<br/>config.py<br/>logging.py<br/>metrics.py<br/>cache.py"]
    end

    API2 --> RET
    API2 --> ING
    API2 --> GEN
    RET  --> UTL
    ING  --> UTL
    GEN  --> UTL
    RET  --> GEN

    class ING  ingStyle
    class RET  retStyle
    class GEN  genStyle
    class API2 apiStyle
    class UTL  utlStyle
```

---

## Project Structure

```
naive-rag-project/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py           # Multi-format document loading
│   │   ├── cleaner.py          # Text normalization
│   │   └── chunker.py          # Recursive text splitting
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py     # ChromaDB + Pinecone abstraction
│   │   ├── retriever.py        # Similarity search
│   │   └── reranker.py         # Optional cross-encoder rerank
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py   # Prompt template engine
│   │   ├── llm_client.py       # OpenAI client wrapper
│   │   └── citation.py         # Source citation extractor
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app + lifespan
│   │   ├── routes.py           # All API routes
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── middleware.py       # Auth, logging, rate-limit
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Settings (Pydantic BaseSettings)
│       ├── logging.py          # Loguru structured logging
│       ├── metrics.py          # Prometheus metrics
│       └── cache.py            # Redis cache wrapper
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_api.py
├── scripts/
│   ├── ingest_documents.py     # CLI ingestion entrypoint
│   └── evaluate_rag.py         # RAG evaluation script (RAGAS)
├── data/
│   ├── raw/                    # Raw uploaded documents
│   ├── processed/              # Cleaned text files
│   └── vectorstore/            # ChromaDB persistent store
├── config/
│   ├── settings.yaml
│   └── prompts.yaml
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourfirm/legalMind-rag
cd legalMind-rag
pip install -e ".[dev]"

# 2. Set environment variables
cp .env.example .env
# Fill in OPENAI_API_KEY, CHROMA_PATH, POSTGRES_URL

# 3. Start infrastructure
docker-compose up -d  # starts Redis + ChromaDB + PostgreSQL

# 4. Ingest your documents
python scripts/ingest_documents.py --source ./data/raw --recursive

# 5. Start the API
uvicorn src.api.main:app --reload --port 8000

# 6. Query it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the termination clauses in the Smith contract?"}'
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | required |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-ada-002` |
| `LLM_MODEL` | Generation model | `gpt-4o` |
| `CHROMA_PATH` | ChromaDB persist path | `./data/vectorstore` |
| `CHUNK_SIZE` | Token chunk size | `512` |
| `CHUNK_OVERLAP` | Token overlap | `64` |
| `TOP_K` | Retrieved chunks | `5` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379` |
| `POSTGRES_URL` | PostgreSQL connection | required |

---

## Evaluation

```bash
python scripts/evaluate_rag.py --dataset ./data/eval_set.json
```

Metrics tracked:
- **Faithfulness** — answer grounded in context
- **Answer Relevancy** — answer addresses the question
- **Context Recall** — relevant chunks retrieved
- **Context Precision** — retrieved chunks are relevant