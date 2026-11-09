import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from config import config

load_dotenv()

# ----------------------------- Settings -----------------------------
MODEL_NAME = "qwen/qwen3.8-27b"  # keep whatever Groq model name already works for you
TOP_K = 3                        # chunks retrieved per question
SCORE_THRESHOLD = 10.0           # Chroma returns a DISTANCE (lower = closer). 10 = effectively off.
MAX_HISTORY = 7                  # last 3 Q&A pairs + current question (keeps the prompt small)
NOT_FOUND = "I couldn't find that in the document."

SYSTEM_PROMPT = (
    "You are a question-answering assistant for the document excerpts given below.\n"
    "Answer the user's question using ONLY the context. "
    "If the context does not contain the answer, say that the document doesn't cover it. "
    "Keep the answer concise: three sentences maximum.\n\n"
    "Context:\n{context}"
)

st.set_page_config(page_title="Embedded QA", layout="centered")
st.title("💬 Embedded QA")

# ----------------------------- API key -----------------------------
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()


# ----------------------------- Cached resources -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Opening vector store...")
def get_vector_store(_embeddings, persist_dir: str):
    """Open Chroma and FAIL LOUDLY if it is empty (Chroma silently creates an empty DB otherwise)."""
    if not os.path.isdir(persist_dir):
        raise RuntimeError(f"Folder does not exist: {persist_dir}")

    # 1) Default LangChain collection name ("langchain")
    store = Chroma(persist_directory=persist_dir, embedding_function=_embeddings)
    if store._collection.count() > 0:
        return store

    # 2) Default collection is empty -> look for any other collection saved in this folder
    found = []
    for c in store._client.list_collections():
        name = getattr(c, "name", c)  # older chromadb returns objects, newer returns names
        count = store._client.get_collection(name).count()
        found.append((name, count))
        if count > 0:
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=_embeddings,
                collection_name=name,
            )

    raise RuntimeError(
        f"No vectors found in '{persist_dir}'. Collections seen: {found or 'none'}. "
        "Run your ingestion script first, or point CHROMA_PERSIST_DIRECTORY to the right folder."
    )


@st.cache_resource
def get_chat_model():
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=1024,  # 400 can be eaten by a reasoning model's <think> block
    )


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks (closed or cut off) some models emit."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


# ----------------------------- LangGraph -----------------------------
class State(MessagesState):
    context: str  # retrieved text for the CURRENT turn only (not stored in chat history)


def call_model(state: State):
    system = SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"]))
    history = state["messages"][-MAX_HISTORY:]  # plain questions and answers only
    response = get_chat_model().invoke([system] + history)
    return {"messages": [AIMessage(content=strip_reasoning(str(response.content)))]}


@st.cache_resource  # cached so the MemorySaver survives Streamlit reruns (this is what gives memory)
def get_langgraph_app():
    workflow = StateGraph(State)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    return workflow.compile(checkpointer=MemorySaver())


# ----------------------------- Load everything -----------------------------
persist_dir = os.path.abspath(config.CHROMA_PERSIST_DIRECTORY)
try:
    vectordb = get_vector_store(get_embeddings_model(), persist_dir)
except Exception as e:
    st.error(f"Could not load the vector store.\n\n{e}")
    st.stop()

app = get_langgraph_app()

# ----------------------------- Session state & sidebar -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    # unique per browser session, because the cached MemorySaver is shared by all sessions
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.subheader("Vector store")
    st.caption(f"Folder: {persist_dir}")
    st.caption(f"Collection: {vectordb._collection.name}")
    st.caption(f"Chunks stored: {vectordb._collection.count()}")
    show_debug = st.checkbox("Show retrieved chunks and scores", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------- Chat -----------------------------
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Retrieve
                results = vectordb.similarity_search_with_score(prompt, k=TOP_K)
                hits = [(d, s) for d, s in results if s <= SCORE_THRESHOLD]

                if show_debug:
                    with st.expander(f"Retrieved {len(results)} chunk(s), {len(hits)} kept"):
                        for d, s in results:
                            st.markdown(
                                f"**score {s:.3f}** · {d.metadata.get('source')} · page {d.metadata.get('page')}"
                            )
                            st.text(d.page_content[:300])

                if not hits:
                    final_response = NOT_FOUND
                else:
                    # 2. Generate: only the plain question goes into memory, context goes in separately
                    context = "\n\n".join(d.page_content for d, _ in hits)
                    result = app.invoke(
                        {"messages": [HumanMessage(content=prompt)], "context": context},
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )
                    ai_response = result["messages"][-1].content or (
                        "The model returned an empty answer (it may have been cut off by max_tokens)."
                    )

                    # 3. Sources (PyPDFLoader pages are 0-based, so +1 for display)
                    sources = list(dict.fromkeys(
                        os.path.basename(str(d.metadata.get("source", "N/A"))) for d, _ in hits
                    ))
                    pages = list(dict.fromkeys(
                        d.metadata.get("page") for d, _ in hits if d.metadata.get("page") is not None
                    ))
                    pages_str = ", ".join(str(p + 1) if isinstance(p, int) else str(p) for p in pages) or "N/A"
                    final_response = (
                        f"{ai_response}\n\n**Source Document**: {', '.join(sources)}\n"
                        f"**Reference Page Numbers**: {pages_str}"
                    )

                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I encountered an error. Please try again."}
                )

# To run: streamlit run chatbot_app.py