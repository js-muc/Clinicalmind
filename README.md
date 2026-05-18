# ClinicalMind

Scalable Clinical AI + RAG System

## Architecture
- Modular RAG
- Clinical reasoning
- Numeric reasoning
- Agent-ready architecture
- LangChain/LangGraph preparation

## Run

```bash
pip install -r requirements.txt
python app/main.py
```

<div align="center">

# 🏥 ClinicalMind
### AI-Powered Clinical Intelligence System for African Healthcare

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge)

> Upload patient records and query them in plain English.
> Built for African healthcare workers who deserve AI assistance too.

[🎥 Demo Video](https://youtu.be/Dqa3pkfvcDU) · [🐛 Report Bug](https://github.com/js-muc/Clinicalmind/issues) · [✨ Request Feature](https://github.com/js-muc/Clinicalmind/issues)

</div>

---

## 📸 Demo

![ClinicalMind Demo](demo.png)

---

## 🧠 What Makes ClinicalMind Different

Most RAG systems just retrieve text.
**ClinicalMind reasons.**

| Feature | Basic RAG | ClinicalMind |
|---------|-----------|--------------|
| Find specific patient | ✅ | ✅ |
| Count patients by condition | ❌ | ✅ |
| Apply numeric threshold logic | ❌ | ✅ |
| Semantic clinical reasoning | ❌ | ✅ |
| Detect anomalies | ❌ | ✅ |
| Drug interaction warnings | ❌ | ✅ |
| Risk scoring | ❌ | ✅ |

---

## ⚡ Example Queries

```python
# Aggregation Layer
"How many patients have pneumonia?"
→ Returns: 7 patients with names, IDs and pages

# Numeric Reasoning Layer
"Which patients had temperature above 39?"
→ Returns: Only matching patients with reasoning

# Exact Match Layer
"Find patient HSP0048"
→ Returns: Full patient record instantly

# Semantic Layer
"Show me patients with breathing difficulties"
→ Returns: Semantically relevant records
```

---

## 🏗️ Architecture

```
clinicalmind/
│
├── app/
│   ├── agents/
│   │   ├── tools/              # LangGraph agent tools
│   │   ├── workflows/          # Diagnosis, risk, triage graphs
│   │   └── memory/             # Conversation memory
│   │
│   ├── clinical/               # Clinical intelligence layer
│   │   ├── reasoning_engine.py
│   │   ├── aggregation_engine.py
│   │   ├── risk_engine.py
│   │   └── symptom_engine.py
│   │
│   ├── rag/                    # RAG pipeline
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   └── retrieval.py
│   │
│   ├── routing/                # Query routing
│   │   ├── intent_detection.py
│   │   └── query_router.py
│   │
│   └── storage/                # Data storage
│       ├── vector_store.py
│       └── document_store.py
│
├── requirements.txt
├── run.py
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/js-muc/Clinicalmind.git
cd Clinicalmind

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the application
python run.py

```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4 |
| RAG Framework | LangChain |
| Agent Framework | LangGraph |
| Embeddings | Sentence Transformers |
| Vector Store | FAISS |
| Interface | Gradio |
| Language | Python 3.11 |
| Deployment | Docker |

---

## 🌍 Why Africa Needs This

```
Enterprise clinical AI systems cost $500,000+
Small African clinics have zero AI assistance
Doctors spend hours manually reading records
Critical warning signs get missed daily

ClinicalMind changes this.
Affordable. Deployable in one hour.
Built specifically for African contexts.
```

---

## 🗺️ Roadmap

- [x] 3-layer reasoning engine
- [x] PDF document processing
- [x] Natural language clinical queries
- [x] Modular LangGraph architecture
- [ ] Drug interaction detection
- [ ] Patient risk scoring
- [ ] React professional dashboard
- [ ] CSV support
- [ ] Multi-document support
- [ ] REST API via FastAPI

---

## 👨‍💻 Built By

**Jesee Muchoki** — Full Stack Developer & AI Engineer
🇰🇪 Nairobi, Kenya

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jesee-muchoki-6870b9259)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/js-muc)
[![YouTube Demo](https://img.shields.io/badge/Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/Dqa3pkfvcDU)

> *"Everything I know I taught myself.*
> *Imagine what I can build for you."*

---

## 📄 License

MIT License — feel free to use and contribute.

