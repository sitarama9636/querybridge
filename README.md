# QueryBridge

**QueryBridge** is a local RAG (Retrieval-Augmented Generation) assistant that connects unstructured documents like Confluence pages and PDFs to a locally running LLM using `Ollama`, `LangChain`, and `ChromaDB`. The project features a FastAPI interface and query rewriting for better information retrieval.

---

## 🔍 Features

- ✅ PDF/Confluence document loader
- ✅ Recursive text chunking with unique chunk ID generation
- ✅ HuggingFace/Bedrock/Ollama embedding support
- ✅ ChromaDB for vector storage and semantic retrieval
- ✅ Query rewriter for vague queries
- ✅ FastAPI endpoints for query and expansion
- ✅ Duplicate document check before DB insert

---

## 🚀 How It Works

1. **Load** PDFs or Confluence content.
2. **Split** into meaningful chunks.
3. **Generate embeddings** using HuggingFace/Ollama.
4. **Store chunks** with unique IDs in ChromaDB.
5. **Query via API**: Query gets expanded and relevant chunks retrieved.
6. **LLM Response** using prompt + context.

---

## 🧪 API Endpoints (FastAPI)

| Method | Endpoint        | Description                    |
|--------|------------------|--------------------------------|
| GET    | `/`              | Health check                   |
| POST   | `/ask`           | Query and get expanded + response |
| POST   | `/expand`        | Only expand the query           |

Test at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🛠️ Tech Stack

- 🧠 Ollama (LLaMA3)
- 🔍 LangChain
- 🧱 ChromaDB
- 🌐 FastAPI
- 📦 HuggingFace Embeddings
- 🧪 Python 3.10+

---

## 📁 Project Structure

```
QueryBridge/
├── app/
│   ├── query_data.py
│   ├── populate_database.py
│   ├── rag_api.py
├── data/
│   └── [PDF files]
├── chroma_vectordb/
├── models/
│   └── [Downloaded embedding models]
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 🧩 Future Enhancements

- [ ] Add LangSmith for prompt tracing and debugging.
- [ ] Fine-tune Ollama model using LoRA/QLoRA.
- [ ] Introduce LangGraph or AutoGen agents.
- [ ] Add model evaluation framework (e.g., relevance scoring).
- [ ] Build a simple frontend dashboard to interact with the API.

---

## 🙌 Author

**Sitarama Bhavirisetty**
