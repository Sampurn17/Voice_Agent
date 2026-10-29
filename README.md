# 🎓 IIIT Una Voice Assistant — Vapi + RAG

A conversational voice agent for **IIIT Una** that answers questions about fees, academics, hostels, campus facilities, placements, and more — powered by [Vapi AI](https://vapi.ai), **Cohere embeddings**, and **Pinecone** vector search.

Built on FastAPI with a clean-architecture backend that also manages todos, reminders, calendar events, and outbound calls.

---

## ✨ Features

| Category | What it does |
|---|---|
| **🔍 RAG Knowledge Base** | Retrieves institute-specific context (fee structure, hostel rules, campus info, curricula) from Pinecone using Cohere `embed-english-v3.0` embeddings |
| **🎙️ Vapi Webhook** | Handles Vapi `tool-calls` and `apiRequest` payloads, returning RAG-retrieved answers in real-time during voice calls |
| **📝 Todo Management** | Create, list, complete, and delete todos via voice or API |
| **⏰ Reminders** | Set reminders with importance levels |
| **📅 Calendar Events** | Schedule events with start/end times |
| **📞 Outbound Calls** | Trigger calls to phone numbers via the Vapi telephony API |
| **🗄️ SQLite Persistence** | Todos, reminders, and events stored locally via SQLAlchemy ORM |

---

## 🏗️ Architecture

```
vapi_personal_assistant_voice_agent/
├── data/                          # Institute knowledge base (source documents)
│   ├── academics/                 # Curriculum PDFs (CSE, ECE)
│   ├── administration/            # Fee structure, campus info
│   └── hostels/                   # Hostel rules & regulations
├── src/
│   ├── api/
│   │   ├── dependencies.py        # DI: DB sessions, Vapi request validation
│   │   └── routes/
│   │       ├── todo.py            # CRUD endpoints for todos
│   │       ├── reminder.py        # CRUD endpoints for reminders
│   │       ├── calendar_event.py  # CRUD endpoints for calendar events
│   │       ├── call.py            # Outbound call initiation via Vapi API
│   │       └── vapi_webhook.py    # POST /vapi/webhook — handles RAG tool calls
│   ├── config/
│   │   └── settings.py            # Pydantic settings (reads .env)
│   ├── models/
│   │   ├── base.py                # SQLAlchemy engine & session factory
│   │   ├── database.py            # Table creation on startup
│   │   ├── domain/                # Pydantic request/response schemas
│   │   └── entities/              # SQLAlchemy ORM models
│   ├── repositories/              # Data access layer (generic CRUD)
│   ├── services/                  # Business logic (todo, reminder, call, calendar)
│   ├── scripts/
│   │   ├── ingest_iiit_una.py     # Seed hardcoded IIIT Una knowledge chunks
│   │   └── ingest_real_files.py   # Ingest PDFs & text files from data/
│   ├── tools/
│   │   └── rag.py                 # Cohere embed → Pinecone query → context retrieval
│   └── utils/
│       └── helpers.py             # JSON parsing & error handling utilities
├── ingest.py                      # Quick-start: create Pinecone index + seed sample docs
├── test_webhook.py                # Smoke test for the /vapi/webhook endpoint
├── requirements.txt
├── .env.example
└── .gitignore
```

### How the RAG Pipeline Works

```
User speaks a question via Vapi voice call
        │
        ▼
Vapi sends a tool-call webhook to POST /vapi/webhook
        │
        ▼
vapi_webhook.py extracts the query & department
        │
        ▼
rag.py embeds the query with Cohere (embed-english-v3.0)
        │
        ▼
Pinecone returns the top-3 most relevant document chunks
        │
        ▼
Context is returned to Vapi → assistant speaks the answer
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Cohere](https://dashboard.cohere.com/api-keys) API key
- A [Pinecone](https://app.pinecone.io) API key
- A [Vapi](https://vapi.ai) account (for voice call features)
- A tunneling tool like [ngrok](https://ngrok.com) to expose `localhost` to Vapi

### 1. Clone & Install

```bash
git clone https://github.com/Sampurn17/Voice_Agent.git
cd Voice_Agent
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your actual keys:

```env
COHERE_API_KEY=your_cohere_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 3. Seed the Knowledge Base

**Option A — Quick start with sample data:**

```bash
python ingest.py
```

This creates the `institute-knowledge-base` Pinecone index and upserts 3 sample document chunks.

**Option B — Ingest real files from `data/`:**

```bash
python -m src.scripts.ingest_real_files
```

Reads all `.txt`, `.md`, and `.pdf` files from `data/`, chunks them with LangChain's `RecursiveCharacterTextSplitter`, embeds via Cohere, and upserts to Pinecone.

**Option C — Ingest hardcoded IIIT Una knowledge:**

```bash
python -m src.scripts.ingest_iiit_una
```

Seeds structured knowledge about academics, ECE labs, administration, campus facilities, and the placement cell.

### 4. Run the Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Expose to Vapi

```bash
ngrok http 8000
```

Copy the HTTPS URL and set it as the **Server URL** or **API Request URL** in your Vapi assistant's tool configuration, pointing to:

```
https://<your-ngrok-url>/vapi/webhook
```

### 6. Test Locally

```bash
python test_webhook.py
```

Or visit the auto-generated docs at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📚 Knowledge Base Content

The `data/` directory contains real IIIT Una institutional data:

| File | Content |
|---|---|
| `administration/fee_structure.txt` | Complete semester-wise fee breakdown, payment modes, scholarships, refund policy |
| `administration/campus_info.txt` | Location, administrative blocks, contact info, anti-ragging policy |
| `hostels/hostel_rules.txt` | Gate timings, mess rules, electrical appliance policy, disciplinary actions |
| `academics/*.pdf` | CSE and ECE curriculum documents |

---

## 🔌 API Endpoints

### Vapi Webhook

| Method | Path | Description |
|---|---|---|
| `POST` | `/vapi/webhook` | Handles Vapi `tool-calls` and direct `apiRequest` payloads |

### Personal Assistant CRUD

| Method | Path | Description |
|---|---|---|
| `POST` | `/todos/` | Create a todo |
| `GET` | `/todos/` | List all todos |
| `POST` | `/reminders/` | Create a reminder |
| `GET` | `/reminders/` | List all reminders |
| `POST` | `/calendar-events/` | Create a calendar event |
| `GET` | `/calendar-events/` | List all events |
| `POST` | `/calls/` | Initiate an outbound call via Vapi |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **FastAPI** | Async web framework & auto-generated OpenAPI docs |
| **Cohere** | Text embedding (`embed-english-v3.0`, 1024-dim) |
| **Pinecone** | Serverless vector database for semantic search |
| **SQLAlchemy** | ORM for local SQLite persistence |
| **Pydantic** | Request/response validation & settings management |
| **LangChain** | Document loading (PDF, TXT) and text chunking |
| **Vapi AI** | Voice assistant platform with telephony integration |
| **Uvicorn** | ASGI server |

---

## 📂 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `COHERE_API_KEY` | ✅ | Cohere API key for generating embeddings |
| `PINECONE_API_KEY` | ✅ | Pinecone API key for vector storage & search |
| `VAPI_API_KEY` | Optional | Only needed for outbound call functionality |

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
