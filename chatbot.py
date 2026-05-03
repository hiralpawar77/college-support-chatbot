import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Page setup
st.set_page_config(page_title="College Chatbot", page_icon="🎓")
st.title("🎓 College Assistant Chatbot")
st.markdown("*Ask questions about your college based on uploaded documents*")

# Initialize Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Please add your GOOGLE_API_KEY to .env file")
    st.stop()

genai.configure(api_key=api_key)

# Function to find available model
@st.cache_resource
def get_available_model():
    """Find a working Gemini model automatically"""
    try:
        # List all available models
        models = genai.list_models()
        
        # Models to try in order of preference
        preferred_models = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.5-pro',
            'gemini-3-flash-preview',
            'gemini-1.5-pro'
        ]
        
        # Get all model names that support generateContent
        available_models = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name.replace('models/', ''))
        
        # Find first preferred model that's available
        for preferred in preferred_models:
            if preferred in available_models:
                st.success(f"✅ Using model: {preferred}")
                return genai.GenerativeModel(preferred)
        
        # If none of preferred, use first available
        if available_models:
            first_model = available_models[0]
            st.info(f"Using available model: {first_model}")
            return genai.GenerativeModel(first_model)
        
        return None
    except Exception as e:
        st.error(f"Error listing models: {e}")
        return None

# Get working model
model = get_available_model()
if model is None:
    st.error("No available Gemini models found. Please check your API key.")
    st.stop()

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load PDFs
@st.cache_resource
def load_pdfs():
    pdf_folder = "data/pdfs"
    all_text = ""
    
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        st.warning(f"Created '{pdf_folder}' folder. Please add your PDF files there.")
        return [], None
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        st.warning(f"No PDF files found. Please add PDFs to '{pdf_folder}' folder.")
        return [], None
    
    # Read all PDFs
    with st.spinner("Loading PDFs..."):
        for pdf_file in pdf_files:
            try:
                reader = PdfReader(os.path.join(pdf_folder, pdf_file))
                for page in reader.pages:
                    all_text += page.extract_text()
                st.success(f"✓ Loaded: {pdf_file}")
            except Exception as e:
                st.error(f"Error loading {pdf_file}: {e}")
    
    if not all_text:
        st.warning("No text could be extracted from the PDFs.")
        return [], None
    
    # Split into chunks
    words = all_text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= 1000:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Create embeddings
    if chunks:
        embedding_model = load_embedding_model()
        embeddings = embedding_model.encode(chunks)
        return chunks, embeddings
    
    return [], None

# Load documents
with st.spinner("Processing documents..."):
    chunks, embeddings = load_pdfs()

# Sidebar with info
with st.sidebar:
    st.header("📊 Status")
    if chunks:
        st.success(f"✅ {len(chunks)} text chunks loaded")
        st.info(f"📄 PDFs in data/pdfs/ folder")
    else:
        st.warning("⚠️ No documents loaded")
    
    st.header("💡 Sample Questions")
    st.markdown("- What programs are offered?")
    st.markdown("- What are admission requirements?")
    st.markdown("- How much is tuition?")
    st.markdown("- What placements are available?")
    
    st.header("🔧 Model Info")
    st.info("Model auto-detected")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Ask about your college...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            if chunks and embeddings is not None and len(chunks) > 0:
                try:
                    # Find relevant chunks
                    embedding_model = load_embedding_model()
                    query_embedding = embedding_model.encode([user_input])
                    similarities = cosine_similarity(query_embedding, embeddings)[0]
                    top_indices = similarities.argsort()[-3:][::-1]
                    
                    # Get relevant context
                    context_parts = []
                    for idx in top_indices:
                        if similarities[idx] > 0.1:
                            context_parts.append(chunks[idx])
                    
                    if context_parts:
                        context = "\n\n".join(context_parts)
                        
                        prompt = f"""You are a college assistant. Answer the question based ONLY on this information:

INFORMATION FROM COLLEGE DOCUMENTS:
{context}

QUESTION: {user_input}

INSTRUCTIONS:
1. Answer based ONLY on the information above
2. If the answer isn't in the information, say "I don't have that information in the documents"
3. Be specific and accurate
4. Keep the answer concise

ANSWER:"""
                        
                        response = model.generate_content(prompt)
                        answer = response.text
                    else:
                        answer = "I couldn't find relevant information in the documents. Try rephrasing your question."
                except Exception as e:
                    answer = f"Error: {str(e)}"
            else:
                answer = "Please add PDF files to the 'data/pdfs' folder first."
            
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})