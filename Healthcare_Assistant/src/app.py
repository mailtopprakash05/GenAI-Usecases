import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from document_loader import DocumentLoader
from vector_store import VectorStore
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize components
doc_loader = DocumentLoader()
vector_store = VectorStore(api_key=os.getenv('OPENAI_API_KEY'))

# Set up Streamlit page
st.set_page_config(page_title="Healthcare Assistant", page_icon="🏥")
st.title("Healthcare Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load documents if not already loaded
@st.cache_resource
def load_documents():
    # Resolve data directory relative to this file so Streamlit works from any CWD
    data_dir = Path(__file__).resolve().parent.parent / "data"
    texts = doc_loader.load_pdfs(str(data_dir))
    if not texts:
        return False
    vector_store.create_index(texts)
    return True

# Load documents
docs_loaded = load_documents()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("How can I help you with your health-related questions?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get relevant documents (handle case where docs didn't load)
    if not docs_loaded:
        assistant_response = "No documents are loaded into the knowledge base. Please add PDFs into the project's `data` directory."
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    else:
        relevant_docs = vector_store.search(prompt)
        context = "\n".join(relevant_docs)

        # Prepare system message with context
        system_message = f"""You are a knowledgeable healthcare assistant. Use the following information to answer the user's question.
    If you don't find relevant information in the context, say so and provide general medical advice with a disclaimer.
    
    Context:
    {context}
    """

        # Generate response using OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        # Display assistant response
        assistant_response = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Add disclaimer
st.sidebar.markdown("""
## Disclaimer
This AI healthcare assistant provides general information only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")