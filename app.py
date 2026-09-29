import os
import streamlit as st
import tempfile
from rag_pipeline import (
    get_embedding_model,
    load_vectorstore,
    process_and_save_pdfs,
    generate_answer
)

st.set_page_config(
    page_title="Multilingual Government Scheme Assistant",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Multilingual Government Scheme RAG Assistant")
st.markdown("Ask questions in **English, Hindi, Marathi**.")

# Sidebar Configuration
st.sidebar.header("⚙️ Settings & Configuration")

# Hugging Face API Token Input
hf_token = st.sidebar.text_input(
    "HuggingFace API Token",
    type="password",
    help="Enter your HuggingFace API Token to enable DeepSeek-V3 LLM inference."
)

# Language Selection
target_language = st.sidebar.selectbox(
    "🌐 Select Target Output Language",
    ["Auto-detect", "English", "Hindi", "Marathi"]
)

# Initialize Embedding Model & FAISS Vectorstore
@st.cache_resource
def init_rag_system():
    embeddings = get_embedding_model()
    vectorstore = load_vectorstore(embeddings)
    return embeddings, vectorstore

embeddings, vectorstore = init_rag_system()

# Dynamic PDF Upload Component
st.sidebar.markdown("---")
st.sidebar.header("📁 Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload new Government Scheme PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.sidebar.button("Index Uploaded PDFs"):
    with st.spinner("Processing & indexing PDF files..."):
        temp_dir = tempfile.mkdtemp()
        pdf_paths = []
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            pdf_paths.append(file_path)
        
        vectorstore = process_and_save_pdfs(pdf_paths, embeddings, vectorstore)
        st.cache_resource.clear()
        st.sidebar.success(f"Successfully indexed {len(uploaded_files)} document(s)!")

# Maintain Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Source Citations & Context"):
                for idx, doc in enumerate(message["sources"], 1):
                    src_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", 0)
                    if isinstance(page, int):
                        page = page + 1
                    st.markdown(f"**Document {idx}:** `{src_name}` (Page {page})")
                    st.caption(doc.page_content[:300] + "...")

# Handle User Input
if user_query := st.chat_input("Ask a question about government schemes (e.g. Who is eligible for PM-KISAN?)..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process query through RAG pipeline
    with st.chat_message("assistant"):
        if vectorstore is None:
            st.error("No FAISS vector store found on disk. Please upload PDF documents in the sidebar first.")
        else:
            with st.spinner("Searching scheme guidelines & generating response..."):
                response_text, retrieved_docs = generate_answer(
                    vectorstore,
                    user_query,
                    target_language=target_language,
                    hf_token=hf_token
                )
                
                st.markdown(response_text)
                
                if retrieved_docs:
                    with st.expander("📚 View Source Citations & Context"):
                        for idx, doc in enumerate(retrieved_docs, 1):
                            src_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page = doc.metadata.get("page", 0)
                            if isinstance(page, int):
                                page = page + 1
                            st.markdown(f"**Document {idx}:** `{src_name}` (Page {page})")
                            st.caption(doc.page_content[:300] + "...")

                # Save assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": retrieved_docs
                })
