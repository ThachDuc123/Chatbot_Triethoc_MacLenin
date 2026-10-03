#  Marxist-Leninist Philosophy Chatbot

> An AI-powered question-answering and learning system for **Marxist-Leninist Philosophy**, combining **Retrieval-Augmented Generation (RAG)**, semantic retrieval, re-ranking, LLMs, and educational question-answering.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![RAG](https://img.shields.io/badge/AI-RAG-purple)](#-system-architecture)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.x-green)](#-technology-stack)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)](#-technology-stack)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red)](#-technology-stack)

---

##  Overview

**Marxist-Leninist Philosophy Chatbot** is an AI-based learning assistant designed to answer questions and support learning in the field of **Marxist-Leninist Philosophy**.

Instead of relying only on an LLM's internal knowledge, the system combines document retrieval with language-model generation. Relevant philosophical materials can be retrieved from the project's knowledge base and used as context for generating answers.

The repository also contains components for:

* Semantic routing
* Vector embeddings
* Document processing
* Retrieval-Augmented Generation
* Re-ranking
* LLM inference
* Reflection
* Philosophy multiple-choice questions
* Retrieval and inference benchmarking

---

##  Project Goals

The project focuses on building a specialized AI assistant capable of:

*  Working with a domain-specific philosophy knowledge base
*  Retrieving relevant information for user questions
*  Generating answers using Large Language Models
*  Improving retrieval quality with re-ranking
*  Routing questions according to their semantic characteristics
*  Supporting philosophy multiple-choice question datasets
*  Evaluating retrieval and generation performance
*  Benchmarking inference and pipeline execution time

---

##  System Architecture

```text
                    ┌──────────────────┐
                    │      User        │
                    │    Question      │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │   Semantic Router    │
                 │  Question Routing    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     Embeddings       │
                 │ Semantic Encoding    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Retriever       │
                 │ Vector Search / RAG  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Re-Ranker       │
                 │ Result Refinement    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │        LLM           │
                 │ Answer Generation    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Reflection      │
                 │ Answer Processing    │
                 └──────────┬───────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │      Answer      │
                    └──────────────────┘
```

---

##  Key Features

### 🔎 Retrieval-Augmented Generation

The system uses a RAG-oriented architecture to connect user questions with information stored in the project's knowledge base.

```text
Question
   ↓
Embedding
   ↓
Vector Retrieval
   ↓
Relevant Documents
   ↓
Re-ranking
   ↓
LLM
   ↓
Generated Answer
```

This allows the chatbot to use retrieved domain-specific information instead of depending entirely on the LLM's pretrained knowledge.

---

###  Semantic Routing

The repository contains a dedicated `semantic_router` module for routing incoming questions based on semantic characteristics.

This allows different types of requests to be processed through appropriate parts of the system.

---

###  Philosophy Knowledge Base

The project contains dedicated resources for philosophy materials:

```text
philosophy_books/
data/
insert_data/
```

These resources form the basis for the retrieval pipeline and philosophy-oriented question answering.

---

###  Re-Ranking

The `re_rank` module provides an additional retrieval stage:

```text
Initial Retrieval
       ↓
Candidate Documents
       ↓
Re-Ranking
       ↓
Most Relevant Context
```

Re-ranking helps refine the initial retrieval results before they are passed to the language model.

---

###  Philosophy MCQ Support

The repository contains dedicated tools for philosophy multiple-choice questions, including:

* MCQ extraction
* MCQ dataset creation
* Topic tagging
* MCQ ingestion
* Retrieval verification
* MCQ API testing

Relevant scripts include:

```text
create_philosophy_mcq_data.py
extract_mcq_from_pdf.py
ingest_philosophy_mcq.py
tag_mcq_topics.py
quick_verify_chroma_mcq.py
test_mcq_api.py
```

---

###  Evaluation & Benchmarking

The project includes dedicated benchmarking scripts for evaluating the AI pipeline:

```text
benchmark_inference_time_llms.py
benchmark_inference_time_pipeline.py
```

The repository also includes RAG evaluation dependencies such as **RAGAS** and text-generation evaluation tools.

---

##  Project Structure

```text
Marxist-Leninist-Philosophy-Chatbot/
│
├── data/
│   └── Knowledge-base data
│
├── embeddings/
│   └── Embedding components
│
├── image/
│   └── Project images / assets
│
├── insert_data/
│   └── Data ingestion utilities
│
├── llms/
│   └── LLM integrations
│
├── luyentap/
│   └── Learning / practice components
│
├── philosophy_books/
│   └── Philosophy knowledge resources
│
├── rag/
│   └── Retrieval-Augmented Generation
│
├── re_rank/
│   └── Retrieval re-ranking
│
├── reflection/
│   └── Answer reflection / refinement
│
├── semantic_router/
│   └── Semantic question routing
│
├── test/
│   └── Testing utilities
│
├── benchmark_inference_time_llms.py
├── benchmark_inference_time_pipeline.py
├── create_philosophy_mcq_data.py
├── extract_mcq_from_pdf.py
├── ingest_philosophy_mcq.py
├── quick_retrieve_check.py
├── quick_test.py
├── serve.py
├── tag_mcq_topics.py
├── train_model.py
│
├── requirements.txt
└── README.md
```

The structure above reflects the current public repository layout.

---

##  Technology Stack

| Category            | Technology                           |
| ------------------- | ------------------------------------ |
| Language            | Python                               |
| LLM Framework       | LangChain                            |
| LLM Providers       | Google / OpenAI / other integrations |
| Embeddings          | Sentence Transformers / FastEmbed    |
| Vector Database     | ChromaDB / Qdrant                    |
| RAG Evaluation      | RAGAS                                |
| NLP                 | Transformers                         |
| API                 | Flask                                |
| Validation          | Pydantic                             |
| Data Processing     | Pandas                               |
| Document Processing | PyPDF2 / python-docx                 |
| Runtime             | ONNX Runtime / Accelerate            |

These dependencies are reflected in the repository's current `requirements.txt`.

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ThachDuc123/Chatbot_Triethoc_MacLenin.git
cd Chatbot_Triethoc_MacLenin
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Some LLM integrations require API credentials.

Create a `.env` file and configure the credentials required by the selected backend.

> Do not commit API keys or other secrets to GitHub.

The repository's benchmark code explicitly checks environment variables for online LLM configurations.

### 5. Run the application

The repository provides:

```text
serve.py
run_chatbot.ps1
```

as application-serving / execution entry points.

The exact command may depend on the selected backend and local configuration.

---

##  Research Pipeline

The project can be viewed as several research components rather than a single chatbot script:

```text
                ┌─────────────────┐
                │ Knowledge Base  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Data Processing  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │   Embeddings     │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Vector Database  │
                └────────┬────────┘
                         ↓
User Question → Semantic Router
                         ↓
                     Retriever
                         ↓
                     Re-Ranker
                         ↓
                       LLM
                         ↓
                    Reflection
                         ↓
                      Answer
```

This architecture makes the project suitable not only as a chatbot application, but also as an experimental platform for studying retrieval, semantic routing, re-ranking, and LLM inference.

---

##  Benchmarking

The repository includes separate experiments for:

### LLM inference

```text
benchmark_inference_time_llms.py
```

### End-to-end pipeline

```text
benchmark_inference_time_pipeline.py
```

This separation makes it possible to distinguish:

```text
LLM inference latency
        vs.
Complete RAG pipeline latency
```

which is useful when analyzing system performance.

---

##  Testing

Several verification scripts are included:

```text
quick_test.py
quick_retrieve_check.py
quick_verify_chroma_mcq.py
test_mcq_api.py
test_philosophy_only.py
test_process.py
```

These scripts support checking different parts of the retrieval, MCQ, processing, and API pipeline.

---

##  Future Improvements

Potential future directions include:

* [ ] Improve retrieval quality with hybrid search
* [ ] Add stronger re-ranking models
* [ ] Expand the philosophy knowledge base
* [ ] Add citation/source references to generated answers
* [ ] Build a standardized evaluation dataset
* [ ] Compare multiple embedding models
* [ ] Compare different LLM backends
* [ ] Optimize inference latency
* [ ] Improve conversational memory
* [ ] Add a polished web interface
* [ ] Dockerize the complete system
* [ ] Add automated evaluation through CI/CD

---

##  Academic Context

This project explores the intersection of:

* **Natural Language Processing**
* **Large Language Models**
* **Retrieval-Augmented Generation**
* **Semantic Search**
* **Information Retrieval**
* **Question Answering**
* **AI-assisted Education**

The main idea is to build a domain-specific AI system that can retrieve and use relevant philosophical knowledge when answering questions.

---

##  Author

**ThachDuc123**

GitHub:
https://github.com/ThachDuc123

---

##  License

Add your preferred license here if this project is intended for public redistribution.

---

⭐ If you find this project useful for research or learning, consider giving it a star.
