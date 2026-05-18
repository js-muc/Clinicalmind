<div align="center">

# 🏥 ClinicalMind

### Clinical AI + Retrieval-Augmented Reasoning System

AI-powered medical document intelligence system built for scalable healthcare record analysis and reasoning.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-FF6F00?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-System-blue?style=for-the-badge)

[🎥 Demo Video](https://youtu.be/Dqa3pkfvcDU) ·
[🐛 Issues](https://github.com/js-muc/Clinicalmind/issues) ·
[⭐ Star Repository](https://github.com/js-muc/Clinicalmind)

</div>

---

# 📌 Overview

ClinicalMind is a modular clinical reasoning system that combines:

- Retrieval-Augmented Generation (RAG)
- Semantic medical search
- Numeric reasoning
- Patient aggregation queries
- Structured clinical extraction

The platform allows healthcare documents to be queried using natural language.

Example:

```text
"Show patient HSP0007"

"Which patients had temperature above 39?"

"How many patients have pneumonia?"
```

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
| Structured clinical extraction | ❌ | ✅ |
| Patient lookup routing | ❌ | ✅ |
| Multi-query clinical search | ❌ | ✅ |

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

## 🏗️ Current Architecture

```text
clinicalmind/
│
├── app/
│   │
│   ├── clinical/
│   │   ├── extractors.py
│   │   ├── reasoning_engine.py
│   │   └── record_lookup.py
│   │
│   ├── handlers/
│   │   └── lookup_handler.py
│   │
│   ├── rag/
│   │   └── loader.py
│   │
│   ├── routing/
│   │   ├── classifier.py
│   │   └── query_parser.py
│   │
│   └── main.py
│
├── requirements.txt
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
python app/main.py

```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| UI | Gradio |
| Embeddings | SentenceTransformers |
| LLM | OpenAI GPT-4 |
| PDF Processing | PyPDF |
| Vector Similarity | NumPy |
| Environment | python-dotenv |
| Architecture Style | Modular RAG |

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
- [x] Modular clinical routing architecture
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

