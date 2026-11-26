import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGPipeline

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Ask About Rithik (Personal AI)", page_icon="👤", layout="centered")
st.title("👤 Ask About Rithik — Personal AI")
st.caption("Ask questions about Rithik's background, career, skills, and experience.")

DOC_PATH = "sample_document.txt"

# Sidebar: Profile Info & Controls
with st.sidebar:
    st.header("👤 RITHIK")
    st.write("I have developed this AI chatbot to answer questions about me , based on my profile data ")
    
    if st.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Suggested Questions:**")
    st.markdown("- *Tell me about yourself.*")
    st.markdown("- *What are your top skills?*")
    st.markdown("- *What is your work experience?*")
    st.markdown("- *What is your education qualification?*")

# Cache RAG pipeline; re-indexes automatically if sample_document.txt is updated
@st.cache_resource(show_spinner=False)
def load_rag_pipeline(file_mtime: float):
    # Check environment variable (.env) or Streamlit Cloud Secrets
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
        else:
            st.error("GOOGLE_API_KEY not found. Please set it in .env or Streamlit Secrets.")
            return None
    
    if not os.path.exists(DOC_PATH):
        st.error(f"'{DOC_PATH}' not found. Please create it in the project folder.")
        return None

    return RAGPipeline(doc_path=DOC_PATH, api_key=api_key)

# Get file modification time to trigger auto-reload when the file changes
mtime = os.path.getmtime(DOC_PATH) if os.path.exists(DOC_PATH) else 0

# Load pipeline silently without technical developer messages
pipeline = load_rag_pipeline(mtime)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User question input
query = st.chat_input("Ask a question about me...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    if pipeline:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = pipeline.ask(query)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_msg = str(e)
                    if "503" in err_msg or "UNAVAILABLE" in err_msg:
                        st.info("⏳ **Google AI is busy right now:** Temporary server spike. Please wait a few seconds and ask again!")
                    elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        st.warning("⚠️ **Daily limit reached:** You have used up the free daily questions for this API key. Limits reset every 24 hours.")
                    else:
                        st.error("⚠️ Something went wrong while connecting to the AI. Please try again in a moment.")
