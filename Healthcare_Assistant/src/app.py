import os
import streamlit as st
from dotenv import load_dotenv
import openai
import requests
from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError
from document_loader import DocumentLoader
from vector_store import VectorStore
from pathlib import Path

# Load environment variables
load_dotenv()

# Read API key from environment (primary source) and allow an optional
# UI override below. We do not persist any UI-provided key to disk.
env_api_key = os.getenv('OPENAI_API_KEY')

# Initialize components (doc loader created first; vector store will be
# initialized after we determine the effective API key so it receives the
# correct credentials or None when no key is provided).
doc_loader = DocumentLoader()
vector_store = None


class MessageModel(BaseModel):
    role: str = Field(..., description="Message role: system/user/assistant")
    content: str = Field(..., min_length=1, description="Message text content")


class ChatRequestModel(BaseModel):
    model: str = Field(..., description="Model name to use for generation")
    messages: List[MessageModel]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(50, ge=1, le=2048)


class ChatResponseModel(BaseModel):
    content: str = Field(..., min_length=0, description="Assistant response text")

# Set up Streamlit page
st.set_page_config(page_title="Healthcare Assistant", page_icon="🏥")
st.title("Healthcare Assistant")

# Sidebar: allow overriding the environment key. If the sidebar field is
# left empty, the app will use the `OPENAI_API_KEY` environment variable.
with st.sidebar:
    st.header("Configuration")
    st.write("Provide an OpenAI API key here to override the OPENAI_API_KEY environment variable (optional). The key is kept only in memory for the session.")
    if "ui_key" not in st.session_state:
        st.session_state.ui_key = ""
    ui_key = st.text_input("OpenAI API key (optional, overrides env)", type="password", value=st.session_state.ui_key)
    if ui_key != st.session_state.ui_key:
        st.session_state.ui_key = ui_key
    # Allow user to choose max tokens for the assistant response (default 50)
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 50
    st.session_state.max_tokens = st.number_input("Max response tokens", min_value=1, max_value=2048, value=st.session_state.max_tokens)

# Determine effective key: UI override wins, otherwise environment key
effective_key = st.session_state.get('ui_key') or env_api_key
if effective_key:
    openai.api_key = effective_key

# Initialize (or reinitialize) the vector store with the effective key
vector_store = VectorStore(api_key=effective_key)

# Show FAISS/index status in the sidebar and provide a rebuild control
with st.sidebar:
    st.markdown("### Index status")
    try:
        idx_path = getattr(vector_store, 'faiss_index_path', None)
        texts_path = getattr(vector_store, 'texts_path', None)
        tfidf_path = getattr(vector_store, 'tfidf_path', None)
        if idx_path and idx_path.exists():
            if getattr(vector_store, 'index', None) is not None:
                st.success("FAISS index: loaded from disk")
            else:
                st.info("FAISS index file exists on disk (not loaded into memory yet)")
        else:
            st.info("FAISS index: not found")
    except Exception:
        st.sidebar.info("Index status: unavailable")

    # Rebuild index button: deletes persisted index files and restarts the app
    if st.button("Rebuild index"):
        try:
            for p in (idx_path, texts_path, tfidf_path):
                try:
                    if p and p.exists():
                        p.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            st.experimental_rerun()
        except Exception:
            # In some Streamlit runtimes calling experimental_rerun can raise
            # when invoked outside the normal script lifecycle. Fall back to
            # asking the user to refresh the page so the app can rebuild the
            # index on the next run.
            st.warning("Index files removed. Please refresh the page to rebuild the index.")

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
        # `relevant_docs` is a list of dicts: {'text','doc_id','score','source'}
        context = "\n".join([d.get('text', '') for d in relevant_docs])
        retrieval_sources = sorted({d.get('source', 'unknown') for d in relevant_docs})

        # Prepare system message with context and an instruction to be precise
        token_limit = st.session_state.get('max_tokens', 50)
        system_message = f"""You are a knowledgeable healthcare assistant. Use the following information to answer the user's question.
If the context contains the answer, prefer using it verbatim (cite briefly). If it does not, say so and provide general medical advice with a clear disclaimer.

Important: be concise and precise. Limit your assistant message to roughly {token_limit} tokens; prioritize essential facts and short, clear recommendations. If you need to ask a clarification question, keep it to one short sentence.

Context:
{context}
"""

        # Prepare messages payload for the model
        messages_payload: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        def call_openai_chat(key: str | None, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 50) -> str:
            """Call OpenAI chat in a resilient way.

            Attempts (in order):
            1) new-style `openai.OpenAI` client
            2) legacy `openai.ChatCompletion` (if present)
            3) direct HTTP POST to the REST API using `requests`
            """
            last_exc = None

            # 1) new-style client (openai.OpenAI)
            try:
                OpenAIClient = getattr(openai, 'OpenAI', None)
                if OpenAIClient is not None:
                    try:
                        client_instance = OpenAIClient(api_key=key) if key else OpenAIClient()
                        resp = client_instance.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                        return resp.choices[0].message.content
                    except Exception as e:
                        last_exc = e
            except Exception as e:
                last_exc = e

            # 2) legacy SDK
            try:
                if hasattr(openai, 'ChatCompletion'):
                    try:
                        if key:
                            openai.api_key = key
                        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                        if isinstance(resp, dict):
                            return resp['choices'][0]['message']['content']
                        return resp.choices[0].message.content
                    except Exception as e:
                        last_exc = e
            except Exception as e:
                last_exc = e

            # 3) direct REST HTTP call
            try:
                if not key:
                    raise RuntimeError('No OpenAI API key available for direct HTTP call')
                url = 'https://api.openai.com/v1/chat/completions'
                headers = {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json'
                }
                payload = {
                    'model': model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                }
                r = requests.post(url, headers=headers, json=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                return data['choices'][0]['message']['content']
            except Exception as e:
                last_exc = e

            # Give up and raise last exception
            raise last_exc

        # Validate outgoing request with Pydantic
        model_name = os.getenv('OPENAI_MODEL', 'gpt-4')
        user_max_tokens = int(st.session_state.get('max_tokens', 50))
        try:
            req_model = ChatRequestModel(
                model=model_name,
                messages=[MessageModel(role=m['role'], content=m['content']) for m in messages_payload],
                temperature=0.7,
                max_tokens=user_max_tokens,
            )
        except ValidationError as ve:
            # Show validation error to user (short form)
            err_text = '; '.join([f"{e['loc'][0]}: {e['msg']}" for e in ve.errors()])
            assistant_response = f"Invalid request parameters: {err_text}"
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
        else:
            # Try calling OpenAI with retries on quota/rate-limit errors.
            short_err = None
            assistant_response = None
            try:
                assistant_response = call_openai_chat(effective_key, req_model.model, messages_payload, temperature=req_model.temperature, max_tokens=req_model.max_tokens)
            except Exception as e:
                short_err = str(e)
                err_lower = short_err.lower()
                is_quota = (
                    'insufficient_quota' in err_lower
                    or ('quota' in err_lower and 'exceed' in err_lower)
                    or '429' in err_lower
                    or 'rate limit' in err_lower
                    or 'rate_limit' in err_lower
                    or 'ratelimit' in err_lower
                )

                # If quota-related, try reducing tokens then switching to a smaller model
                if is_quota:
                    # First attempt: reduce tokens by half (min 10)
                    reduced = max(10, req_model.max_tokens // 2)
                    try:
                        assistant_response = call_openai_chat(effective_key, req_model.model, messages_payload, temperature=req_model.temperature, max_tokens=reduced)
                        assistant_response = f"(Note: response generated with reduced max_tokens={reduced})\n\n" + assistant_response
                    except Exception as e2:
                        short_err = str(e2)
                        # Second attempt: try smaller model (gpt-3.5-turbo)
                        try:
                            assistant_response = call_openai_chat(effective_key, 'gpt-3.5-turbo', messages_payload, temperature=req_model.temperature, max_tokens=reduced)
                            assistant_response = f"(Note: response generated with model=gpt-3.5-turbo and max_tokens={reduced})\n\n" + assistant_response
                        except Exception as e3:
                            short_err = str(e3)
                            # Final fallback: local TF-IDF retrieval
                            try:
                                fallback_docs = vector_store.search(prompt, k=5)
                            except Exception:
                                fallback_docs = []
                            # fallback_docs is list of dicts; extract text
                            fallback_texts = [d.get('text', '') for d in fallback_docs] if fallback_docs else []
                            if fallback_texts:
                                assistant_response = (
                                    "OpenAI generation failed due to quota or rate limits. Below are the top local document excerpts (lower-quality fallback):\n\n"
                                ) + "\n\n---\n\n".join(fallback_texts)
                                response_provenance = 'local_fallback'
                            else:
                                assistant_response = (
                                    "OpenAI generation failed due to quota or rate limits and no local fallbacks are available. "
                                    "Please check your billing, switch to a model you have access to (e.g. gpt-3.5-turbo), reduce max tokens, or paste a different API key in the sidebar."
                                )
                else:
                    # Non-quota error: show friendly message and optional debug
                    assistant_response = (
                        "Failed to generate a response. This may be caused by: an invalid API key, model access restrictions, network problems, or rate limiting. "
                        "Please verify your API key in the sidebar and try again."
                    )

            # Validate assistant response (Pydantic) and display
            try:
                if assistant_response is None:
                    raise RuntimeError(short_err or 'Unknown error')
                ChatResponseModel(content=assistant_response)
            except ValidationError as ve:
                assistant_response = "Received an invalid response from the model."
                if st.session_state.get('show_debug') and short_err:
                    assistant_response += f"\n\nDebug: {short_err}"
            except Exception:
                # keep assistant_response as-is but show debug info if enabled
                if st.session_state.get('show_debug') and short_err:
                    assistant_response += f"\n\nDebug: {short_err}"

            # If not already set by fallback paths, mark provenance as LLM
            if 'response_provenance' not in locals():
                response_provenance = 'llm'

            # Append assistant message and show provenance badge / retrieval sources
            st.session_state.messages.append({"role": "assistant", "content": assistant_response, "meta": {"provenance": response_provenance, "retrieval_sources": retrieval_sources}})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
                # Show a small provenance line so users know whether the answer used FAISS/TF-IDF or was a local fallback
                try:
                    rs = ', '.join(retrieval_sources) if retrieval_sources else 'none'
                    if response_provenance == 'llm':
                        st.caption(f"Answer generated by LLM (retrieval used: {rs})")
                    elif response_provenance == 'local_fallback':
                        st.caption(f"Answer provided from local retrieval only (no LLM). Retrieval source: {rs}")
                    else:
                        st.caption(f"Answer source: {response_provenance}. Retrieval: {rs}")
                except Exception:
                    pass

            # Provide an expander with the top retrieved documents and their metadata
            try:
                if relevant_docs:
                    with st.expander("Show retrieved documents and provenance"):
                        for d in relevant_docs[:5]:
                            src = d.get('source', 'unknown')
                            did = d.get('doc_id', 'n/a')
                            sc = d.get('score', None)
                            sf = d.get('source_file', 'unknown')
                            pg = d.get('page', None)
                            header = f"**Doc {did}** — source: {src} — file: {sf}"
                            if pg:
                                header += f" — page: {pg}"
                            header += f" — score: {sc}"
                            st.markdown(f"{header}\n\n{d.get('text','')[:800]}{'...' if len(d.get('text',''))>800 else ''}")
            except Exception:
                pass

# Add disclaimer
st.sidebar.markdown("""
## Disclaimer
This AI healthcare assistant provides general information only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")

# Footer / credit (centered & muted)
st.markdown(
    "<div style='text-align:center; color:#6c757d; font-size:0.95rem; margin-top:18px; margin-bottom:8px;'>"
    "Built by <strong>Prakash Pujari</strong>, driven by intelligent automation."
    "</div>",
    unsafe_allow_html=True,
)