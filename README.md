
---

# 📄 IFRS 9 Credit Risk RAG Assistant

> **A Retrieval-Augmented Generation (RAG) system for IFRS 9 credit risk analysis with strict evidence grounding and page-level citations.**

This project builds a **finance-grade RAG assistant** that answers IFRS 9 credit risk questions (e.g. SICR, Stage 1–2–3) **strictly based on regulatory documents**, with **transparent evidence and citations**.

---

## 🚀 Key Features

* 🔍 **FAISS-based semantic retrieval**
* 📄 **PDF-grounded answers with page-level citations**
* 🤖 **LLM-generated natural language explanations**
* 🧠 **No hallucination**: answers are grounded in retrieved evidence only
* 🖥️ **Interactive Streamlit UI** (ready for demo & interviews)
* 💼 **Finance-grade use case** (IFRS 9 / Credit Risk / ECL)

---

## 🧠 Example Questions

* *What is Significant Increase in Credit Risk (SICR) under IFRS 9?*
* *How does IFRS 9 define Stage 2 assets?*
* *What qualitative indicators are used to assess SICR?*

Each answer includes:

* ✅ Clear explanation
* ✅ Bullet-point reasoning
* ✅ **Exact PDF source & page reference**

---

## 🏗️ Project Architecture

```
User Question
      ↓
FAISS Vector Search (Top-k Chunks)
      ↓
Evidence Assembly (with PDF + page)
      ↓
LLM Answer Generation
      ↓
Answer + Citations (UI)
```

---

## 📂 Project Structure

```
finance-rag-ifrs9/
│
├── data/
│   ├── raw_pdfs/          # Original IFRS 9 / BCBS documents
│   └── processed/         # pages.jsonl, chunks.jsonl, FAISS index
│
├── src/
│   ├── ingest.py          # PDF → page-level text
│   ├── chunk.py           # Text chunking
│   ├── index.py           # Embedding + FAISS index
│   ├── rag_core.py        # Retrieval & evidence logic
│   └── rag_llm.py         # LLM-based answer generation
│
├── app/
│   └── streamlit_retrieval_demo.py   # Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Create environment & install dependencies

```bash
conda create -n rag python=3.11
conda activate rag
pip install -r requirements.txt
```

---

## 📥 2. Ingest Documents (One-Time Step)

Run this **only once**, or when documents change:

```bash
python src/ingest.py
python src/chunk.py
python src/index.py
```

This will:

* Parse PDFs
* Split text into chunks
* Build FAISS vector index

---

## ▶️ 3. Run the App

```bash
streamlit run app/streamlit_retrieval_demo.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧪 Demo Screenshot

> **IFRS 9 Credit Risk RAG Assistant with evidence citations**

* Left: question input & settings
* Right: retrieved evidence (PDF + page)
* Center: LLM answer grounded in sources

*(You can add your screenshot here in GitHub)*

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## 💡 Why This Project Matters

This project demonstrates:

* ✅ **RAG in a regulated finance context**
* ✅ **Explainability & auditability**
* ✅ **Practical application of LLMs beyond chatbots**
* ✅ **Production-ready design for risk / compliance teams**

Perfect for:

* Credit Risk / IFRS 9 roles
* Risk Modeling / Analytics interviews
* LLM + Finance portfolios

---

## 📌 Future Extensions

* SICR-specific prompts (Stage 1 / 2 / 3)
* Multi-document comparison (IFRS vs BCBS)
* Local LLM backend (offline / cost-free)
* Answer export (Word / PDF for reports)

---

## 👤 Author

**Mengjie**
Background: Statistics / Finance / Credit Risk / LLM Applications

---


