# 🏛️ Multilingual Government Scheme RAG Assistant

A multilingual Retrieval-Augmented Generation (RAG) assistant designed to help citizens inquire about Indian government scheme guidelines in **English, Hindi, Marathi**, and other regional languages.

Built with **Streamlit**, **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **DeepSeek-V3**.

---

## 🌟 Key Features

- **🌐 Multilingual QA**: Ask questions in English, Hindi, or Marathi and receive grounded responses in your chosen target language.
- **📚 Official Scheme Documents Knowledge Base**: Pre-loaded with official guidelines for PM-KISAN, PMFBY, PMJDY, PMJJBY, PMKVY 4.0, PMAY Urban, NAIS, WBCIS, and SDRF/NDRF.
- **🔍 Query Expansion & Reranking**: Intelligently expands queries with scheme-specific domain keywords and filters target PDFs for precise retrieval.
- **📑 Source & Page-Level Citations**: Expandable accordion UI showing exact source document names, page numbers, and retrieved context chunks.
- **📁 Dynamic Document Indexing**: Upload new government scheme PDF files on the fly via the sidebar to update the vector database in real-time.

---

## 🏗️ Project Architecture

```
End_to_End_Project/
├── Documents/                   # Official Government Scheme PDF files
├── faiss_index/                 # Vector index files (index.faiss, index.pkl)
├── Jupyter_Notebook/            # Exploration notebooks
├── rag_pipeline.py              # Core RAG backend (Embeddings, FAISS, Query Expansion, LLM chain)
├── app.py                       # Streamlit Web Application Interface
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SatyajeetGawali04/multilingual-government-rag.git
cd multilingual-government-rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

## ☁️ Free Deployment on Streamlit Community Cloud

1. Fork or push this repository to your GitHub account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a **New App**, select your repository, set the main file to `app.py`, and click **Deploy**.
4. *(Optional)* Add your HuggingFace token under **App Settings -> Secrets**:
   ```toml
   HUGGINGFACEHUB_API_TOKEN = "your_hf_token_here"
   ```

---

## 🛠️ Built With

- **Framework**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM Engine**: `deepseek-ai/DeepSeek-V3` via HuggingFace Hub
