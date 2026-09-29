import os
import streamlit as st
import tempfile
from rag_pipeline import (
    get_embedding_model,
    load_vectorstore,
    process_and_save_pdfs,
    generate_answer
)

# Page Setup & Custom CSS Styling
st.set_page_config(
    page_title="Multilingual Government Scheme Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    /* Main Background & Fonts */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-box h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin-bottom: 8px;
    }
    
    .header-box p {
        color: #e0e0e0;
        font-size: 1.05rem;
    }

    /* Badge Pills */
    .scheme-badge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.2);
        color: #fff;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        margin: 2px;
    }

    /* Citation Box */
    .citation-card {
        background-color: #ffffff;
        border-left: 4px solid #2a5298;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Sidebar Header */
    .sidebar-header {
        color: #1e3c72;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=1)

# Header Section
st.markdown("""
<div class="header-box">
    <h1>🏛️ Multilingual Government Scheme RAG Assistant</h1>
    <p>Ask questions in <b>English, Hindi, or Marathi</b> about official Indian government scheme guidelines.</p>
    <div style="margin-top: 10px;">
        <span class="scheme-badge">🌾 PM-KISAN</span>
        <span class="scheme-badge">🛡️ PMFBY</span>
        <span class="scheme-badge">🏦 PMJDY</span>
        <span class="scheme-badge">🏠 PMAY Urban</span>
        <span class="scheme-badge">💡 PMKVY 4.0</span>
        <span class="scheme-badge">🌦️ WBCIS / NAIS</span>
    </div>
</div>
""", unsafe_allow_html=1)

# Sidebar Configuration
st.sidebar.markdown("<h2 class='sidebar-header'>⚙️ Settings & Configuration</h2>", unsafe_allow_html=1)

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
st.sidebar.markdown("<h3 class='sidebar-header'>📁 Document Management</h3>", unsafe_allow_html=1)
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

# Quick Question Suggestions
st.markdown("### 💡 Suggested Questions")
col1, col2, col3 = st.columns(3)

suggested_query = None
with col1:
    if st.button("🌾 Who is eligible for PM-KISAN?"):
        suggested_query = "Who is eligible for PM-KISAN?"
with col2:
    if st.button("🏦 पीएम किसान योजना पात्र कोण आहे?"):
        suggested_query = "पीएम किसान योजनेसाठी कोण पात्र आहे?"
with col3:
    if st.button("🏠 पीएम किसान योजना पात्रता क्या है?"):
        suggested_query = "पीएम किसान योजना के लिए कौन पात्र है?"

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

# Handle Input (Chat Box or Quick Suggestion)
user_query = st.chat_input("Ask a question about government schemes...")
if suggested_query and not user_query:
    user_query = suggested_query

if user_query:
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
