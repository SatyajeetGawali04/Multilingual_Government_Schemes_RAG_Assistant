import os
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

INDEX_FOLDER = r"C:\Users\Admin\OneDrive\Documents\End_to_End_Project\faiss_index"
DOCUMENTS_FOLDER = r"C:\Users\Admin\OneDrive\Documents\End_to_End_Project\Documents"

# Initialize Multilingual Embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# Load existing FAISS vector store or return None if missing
def load_vectorstore(embeddings_model):
    if os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss")) and os.path.exists(os.path.join(INDEX_FOLDER, "index.pkl")):
        return FAISS.load_local(
            INDEX_FOLDER, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
    return None

# Process PDFs and save FAISS index
def process_and_save_pdfs(pdf_paths: List[str], embeddings_model, vectorstore: Optional[FAISS] = None) -> FAISS:
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings_model)
    else:
        vectorstore.add_documents(chunks)
        
    os.makedirs(INDEX_FOLDER, exist_ok=True)
    vectorstore.save_local(INDEX_FOLDER)
    return vectorstore

# Query expansion helper
def expand_query(question: str) -> str:
    q = question.lower().strip()

    if any(k in q for k in ["nais", "राष्ट्रीय कृषि बीमा", "राष्ट्रीय कृषी विमा", "कृषि बीमा योजना", "कृषी विमा योजना"]):
        return (
            question + " NAIS National Agricultural Insurance Scheme "
            "crop insurance agricultural insurance farmers crop loss crop failure yield loss indemnity coverage"
        )
    elif any(k in q for k in ["maharashtra agriculture", "agriculture schemes", "महाराष्ट्र कृषी", "महाराष्ट्र कृषि", "कृषी योजना", "कृषि योजना"]):
        return (
            question + " Maharashtra agriculture schemes agriculture farmer schemes Maharashtra "
            "farmer assistance agricultural support crop agriculture financial assistance"
        )
    elif any(k in q for k in ["pmay", "pmay urban", "पीएमएवाई", "प्रधानमंत्री आवास", "आवास योजना"]):
        return (
            question + " PMAY Urban Pradhan Mantri Awas Yojana Urban Housing for All urban housing beneficiary central assistance"
        )
    elif any(k in q for k in ["pmfby", "पीएमएफबीवाय", "पीएमएफबीवाई", "फसल बीमा", "पीक विमा"]):
        return (
            question + " PMFBY Pradhan Mantri Fasal Bima Yojana crop insurance farmers crop loss crop damage financial support"
        )
    elif any(k in q for k in ["pmjdy", "पीएमजेडीवाई", "प्रधानमंत्री जन धन", "जन धन", "jan dhan"]):
        return (
            question + " PMJDY Pradhan Mantri Jan Dhan Yojana financial inclusion universal access to banking basic banking account RuPay debit card"
        )
    elif any(k in q for k in ["pmjjby", "पीएमजेजेबीवाय", "प्रधानमंत्री जीवन ज्योति", "जीवन ज्योति बीमा", "jeevan jyoti"]):
        return (
            question + " PMJJBY Pradhan Mantri Jeevan Jyoti Bima Yojana life insurance scheme subscriber eligibility life cover death benefit"
        )
    elif any(k in q for k in ["pmkvy", "pmkvy 4.0", "पीएमकेव्हीवाय", "कौशल विकास", "कौशल्य विकास", "skill development"]):
        return (
            question + " PMKVY PMKVY 4.0 Pradhan Mantri Kaushal Vikas Yojana skill development training trainees certification"
        )
    elif any(k in q for k in ["pm-kisan", "pm kisan", "pmकिसान", "पीएम-किसान", "पीएम किसान", "pm-किसान", "pm किसान"]):
        return (
            question + " PM-KISAN Pradhan Mantri Kisan Samman Nidhi financial support income support Rs 6000 per year landholding farmers cultivable land"
        )
    elif any(k in q for k in ["sdrf", "ndrf", "एसडीआरएफ", "एनडीआरएफ", "disaster response", "disaster relief", "आपत्ती", "आपदा"]):
        return (
            question + " SDRF State Disaster Response Fund NDRF National Disaster Response Fund disaster relief assistance natural calamity"
        )
    elif any(k in q for k in ["wbcis", "weather based crop insurance", "हवामान आधारित पीक विमा", "weather based", "हवामान"]):
        return (
            question + " WBCIS Weather Based Crop Insurance Scheme weather parameters rainfall temperature weather triggers crop loss"
        )
    else:
        return question

# Identify target PDF document filters
def get_scheme_sources(question: str) -> Optional[List[str]]:
    q = question.lower().strip()

    if any(k in q for k in ["nais", "राष्ट्रीय कृषि बीमा", "राष्ट्रीय कृषी विमा"]):
        return ["NAIS_Scheme.pdf"]
    if any(k in q for k in ["agriculture schemes", "maharashtra agriculture", "महाराष्ट्र कृषी", "महाराष्ट्र कृषि", "कृषी योजना", "कृषि योजना"]):
        return ["New_Agriculture_Schemes_Maharashtra.pdf"]
    if any(k in q for k in ["pmay", "प्रधानमंत्री आवास", "आवास योजना"]):
        return ["PMAY_Urban_Guidelines.pdf"]
    if any(k in q for k in ["pmfby", "पीएमएफबीवाय", "पीएमएफबीवाई", "फसल बीमा", "पीक विमा"]):
        return ["PMFBY_Revamped_OGs.pdf", "Revamped Operational Guidelines_17th August 2020.pdf"]
    if any(k in q for k in ["pmjdy", "जन धन", "jan dhan", "प्रधानमंत्री जन धन"]):
        return ["PMJDY_Guidelines.pdf"]
    if any(k in q for k in ["pmjjby", "जीवन ज्योति", "jeevan jyoti", "प्रधानमंत्री जीवन ज्योति"]):
        return ["PMJJBY_Rules.pdf"]
    if any(k in q for k in ["pmkvy", "पीएमकेव्हीवाय", "कौशल विकास", "कौशल्य विकास", "skill development"]):
        return ["PMKVY_4_0.pdf"]
    if any(k in q for k in ["pm-kisan", "pm kisan", "pm-किसान", "pm किसान", "पीएम-किसान", "पीएम किसान"]):
        return ["PM_KISAN_Guidelines.pdf", "PM_KISAN_Operational_Guidelines.pdf"]
    if any(k in q for k in ["sdrf", "ndrf", "एसडीआरएफ", "एनडीआरएफ", "disaster response", "disaster relief", "आपत्ती", "आपदा"]):
        return ["SDRFNDRFGuideline_20082024.pdf"]
    if any(k in q for k in ["wbcis", "weather based crop insurance", "हवामान आधारित पीक विमा", "weather based", "हवामान"]):
        if "operational" in q:
            return ["WBCIS_Operational_Guidelines.pdf"]
        return ["WBCIS_Guidelines.pdf"]

    return None

# Retrieve documents from vector store
def retrieve_documents(vectorstore: FAISS, question: str, top_k: int = 4) -> List[Document]:
    expanded_question = expand_query(question)
    allowed_sources = get_scheme_sources(question)

    results = vectorstore.similarity_search_with_score(expanded_question, k=100)

    if allowed_sources:
        filtered_results = []
        for doc, score in results:
            source = os.path.basename(doc.metadata.get("source", ""))
            if source in allowed_sources:
                filtered_results.append((doc, score))
        results = filtered_results

    unique_results = []
    seen = set()
    for doc, score in results:
        text_key = doc.page_content.strip()
        if text_key not in seen:
            seen.add(text_key)
            unique_results.append((doc, score))

    final_results = unique_results[:top_k]
    return [doc for doc, score in final_results]

# Format documents into string context
def format_docs(docs: List[Document]) -> str:
    formatted_docs = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "Unknown")
        if isinstance(page, int):
            page = page + 1
        formatted_docs.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_docs)

# Chat Prompt Template
PROMPT_TEMPLATE = """
You are a multilingual Government Scheme RAG assistant.
Your job is to answer the user's question using ONLY the provided context.

========================
IMPORTANT RULES
========================
1. Use ONLY the information present in the CONTEXT.
2. Do NOT use outside knowledge, memory, assumptions, or general knowledge.
3. If the answer is clearly available in the context, ANSWER THE QUESTION. Do NOT use the fallback response.
4. Keep the answer focused on exactly what the user asked.

========================
TARGET LANGUAGE RULE
========================
Please write your response in the requested language: {target_language}.
If the target language is "Auto-detect", reply in the same language as the question.

========================
SOURCE / CITATION RULES
========================
Every factual answer must include the source and page number from the CONTEXT.
(Source: <document_name>, Page: <page_number>)

========================
FALLBACK RULE
========================
Use the following fallback ONLY when the CONTEXT genuinely does not contain enough information to answer the question:
"I do not have enough information in the provided documents to answer this question. Please consult the relevant government department or authorized person for more clarity."

========================
CONTEXT
========================
{context}

========================
QUESTION
========================
{question}

Now provide the final answer.
"""

def generate_answer(
    vectorstore: FAISS, 
    question: str, 
    target_language: str = "Auto-detect", 
    hf_token: Optional[str] = None
) -> Tuple[str, List[Document]]:
    docs = retrieve_documents(vectorstore, question)
    context_str = format_docs(docs)

    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    try:
        llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-V3",
            max_new_tokens=512,
            temperature=0.2,
        )
        chat_model = ChatHuggingFace(llm=llm)

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | chat_model | StrOutputParser()

        response = chain.invoke({
            "context": context_str,
            "question": question,
            "target_language": target_language
        })
        return response, docs
    except Exception as e:
        # Fallback summary response if API token is missing or offline
        fallback_msg = f"⚠️ LLM Generation Error / API key missing: {str(e)}\n\nHere is the retrieved context from the documents:\n\n{context_str}"
        return fallback_msg, docs
