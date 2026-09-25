"""Streamlit frontend for the Agentic RAG Harness.

Run with:
    streamlit run frontend.py
Or with uv:
    uv run streamlit run frontend.py

Make sure the FastAPI backend is running:
    uvicorn rag_harness.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
from datetime import datetime
from pathlib import Path
import requests
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG Harness Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .step-card {
        padding: 10px 14px;
        border-radius: 8px;
        background-color: #1E293B;
        border-left: 4px solid #6366F1;
        margin-bottom: 8px;
        font-family: monospace;
        font-size: 0.82rem;
    }
    .step-time {
        color: #94A3B8;
        font-size: 0.72rem;
        margin-bottom: 2px;
    }
    .badge-ok {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-warn {
        background-color: #78350F;
        color: #FDE68A;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-err {
        background-color: #7F1D1D;
        color: #FECACA;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "activity_logs" not in st.session_state:
    st.session_state.activity_logs = []

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def add_log(step_name: str, detail: str, level: str = "INFO"):
    """Record an action step into the sidebar activity log."""
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {
        "timestamp": ts,
        "step": step_name,
        "detail": detail,
        "level": level,
    }
    st.session_state.activity_logs.append(entry)


# ---------------------------------------------------------
# Sidebar: Server Health, Settings & Live Activity Steps
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control & Telemetry")

    backend_url = st.text_input(
        "Backend API URL",
        value=os.getenv("FASTAPI_URL", "http://localhost:8000"),
        help="Base URL of the running FastAPI server",
    ).rstrip("/")

    # Check Health
    col_h1, col_h2 = st.columns([1, 1])
    with col_h1:
        check_now = st.button("🔄 Check Health", use_container_width=True)

    server_online = False
    llm_info = None

    try:
        health_resp = requests.get(f"{backend_url}/v1/health", timeout=2)
        if health_resp.status_code == 200:
            server_online = True
            health_data = health_resp.json()
    except Exception:
        server_online = False

    if server_online:
        st.markdown(
            '<span class="badge-ok">● Backend Online</span>',
            unsafe_allow_html=True,
        )
        try:
            llm_resp = requests.get(f"{backend_url}/v1/health/llm", timeout=2)
            if llm_resp.status_code == 200:
                llm_info = llm_resp.json()
        except Exception:
            pass
    else:
        st.markdown(
            '<span class="badge-err">● Backend Offline</span>',
            unsafe_allow_html=True,
        )
        st.caption("Start with: `uvicorn rag_harness.api.main:app --reload`")

    if llm_info:
        with st.expander("LLM Chain Status", expanded=False):
            st.json(llm_info)

    st.divider()

    # Sidebar Step-by-Step Activity Log
    st.subheader("📋 Live Activity Steps")
    st.caption("Real-time steps executed during ingestion and queries:")

    if st.button("Clear Steps", use_container_width=True):
        st.session_state.activity_logs = []
        st.rerun()

    if not st.session_state.activity_logs:
        st.info("No actions recorded yet. Ingest a file or run a query.")
    else:
        for item in reversed(st.session_state.activity_logs):
            lvl = item["level"]
            badge_class = (
                "badge-ok"
                if lvl == "OK"
                else "badge-warn"
                if lvl == "WARN"
                else "badge-err"
                if lvl == "ERR"
                else "badge-warn"
            )
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-time">[{item['timestamp']}] <span class="{badge_class}">{lvl}</span></div>
                    <strong>{item['step']}</strong><br/>
                    <span style="color: #cbd5e1;">{item['detail']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">Agentic RAG Harness Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">LangGraph Agentic Reasoning & Vector Search Harness with FastAPI & Qdrant</div>',
    unsafe_allow_html=True,
)

tab_ingest, tab_query, tab_graph = st.tabs([
    "📥 Document Ingestion",
    "💬 Query & Agentic Reasoning",
    "🕸️ Agent Graph Architecture",
])

# ---------------------------------------------------------
# Tab 1: Ingestion
# ---------------------------------------------------------
with tab_ingest:
    st.subheader("Upload and Ingest Knowledge Documents")
    st.write(
        "Upload any Markdown, PDF, or text file. The backend will parse, chunk, embed, and store it in Qdrant."
    )

    uploaded_file = st.file_uploader(
        "Choose a document to ingest",
        type=["pdf", "md", "txt"],
        help="Upload files like policy docs, manuals, or research papers.",
    )

    col1, col2 = st.columns(2)
    default_doc_id = (
        Path(uploaded_file.name).stem if uploaded_file else "sample_doc"
    )
    with col1:
        doc_id = st.text_input("Document ID", value=default_doc_id)
    with col2:
        doc_version = st.text_input("Document Version", value="v1")

    if st.button("🚀 Ingest Document", type="primary", disabled=not uploaded_file):
        if not server_online:
            st.error("FastAPI backend is offline! Please start it on http://localhost:8000 first.")
        else:
            upload_dir = Path(".rag_uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved_path = upload_dir / uploaded_file.name

            # Step 1: Save locally
            add_log("Ingest Step 1", f"Saving uploaded file: {uploaded_file.name}", "INFO")
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            add_log("Ingest Step 2", f"File written to disk at {saved_path.resolve()}", "INFO")

            # Step 2: Post to FastAPI
            add_log("Ingest Step 3", f"Calling POST /v1/documents for doc_id='{doc_id}', version='{doc_version}'", "INFO")
            with st.spinner("Processing document (parsing, chunking, and embedding in Qdrant)..."):
                try:
                    payload = {
                        "file_path": str(saved_path.resolve()),
                        "doc_id": doc_id,
                        "doc_version": doc_version,
                    }
                    t0 = time.time()
                    resp = requests.post(f"{backend_url}/v1/documents", json=payload, timeout=120)
                    elapsed = round(time.time() - t0, 2)

                    if resp.status_code == 200:
                        data = resp.json()
                        chunks = data.get("chunks_created", 0)
                        add_log(
                            "Ingest Step 4",
                            f"Completed in {elapsed}s: Created {chunks} chunks for '{doc_id}'",
                            "OK",
                        )
                        st.success(f"✅ Ingestion successful! Created **{chunks}** chunks in {elapsed}s.")
                        st.session_state.ingested_docs.append({
                            "doc_id": doc_id,
                            "version": doc_version,
                            "chunks": chunks,
                            "file": uploaded_file.name,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                    else:
                        err_msg = resp.text
                        add_log("Ingest Failed", f"HTTP {resp.status_code}: {err_msg}", "ERR")
                        st.error(f"❌ Ingestion failed ({resp.status_code}): {err_msg}")
                except Exception as e:
                    add_log("Ingest Error", str(e), "ERR")
                    st.error(f"Error calling backend: {e}")

    # Display list of ingested docs in this session
    if st.session_state.ingested_docs:
        st.markdown("#### Recently Ingested Documents")
        st.dataframe(st.session_state.ingested_docs, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Querying & Reasoning
# ---------------------------------------------------------
with tab_query:
    st.subheader("Ask Questions with Vector Search & Multi-Step Agentic Reasoning")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        mode = st.radio(
            "Reasoning Mode",
            ["🤖 Agentic RAG (LangGraph Multi-Step)", "⚡ Standard RAG (Direct Retrieve & Generate)"],
            horizontal=True,
        )
    with col_q2:
        query_version_key = st.text_input("Version Filter Key", value="v1")

    question = st.text_area(
        "Enter your question",
        placeholder="e.g. What are the key points in the uploaded document?",
        height=100,
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        submit = st.button("🔍 Submit Question", type="primary", disabled=not question.strip())

    if submit:
        if not server_online:
            st.error("FastAPI backend is offline! Please start it on http://localhost:8000 first.")
        else:
            is_agentic = "Agentic" in mode
            endpoint = f"{backend_url}/v1/query/agentic" if is_agentic else f"{backend_url}/v1/query"

            add_log(
                "Query Step 1",
                f"Starting {'Agentic' if is_agentic else 'Standard'} Query: '{question[:50]}...'",
                "INFO",
            )

            with st.spinner("Agent is reasoning and searching knowledge base..."):
                t0 = time.time()
                try:
                    payload = {"question": question, "doc_version_key": query_version_key}
                    resp = requests.post(endpoint, json=payload, timeout=120)
                    elapsed = round(time.time() - t0, 2)

                    if resp.status_code == 200:
                        res_data = resp.json()
                        answer = res_data.get("answer", "")
                        sources = res_data.get("sources_used", [])

                        if is_agentic:
                            action_history = res_data.get("action_history", [])
                            iterations = res_data.get("iterations_used", 1)
                            degraded = res_data.get("degraded", False)
                            degrade_reason = res_data.get("degrade_reason")

                            # Log each agent step to sidebar
                            for idx, step_item in enumerate(action_history, 1):
                                add_log(f"Agent Step {idx}", str(step_item), "INFO")

                            add_log(
                                "Query Complete",
                                f"Agentic answer ready in {elapsed}s ({iterations} iterations, {len(sources)} sources)",
                                "OK",
                            )

                            # Display result
                            st.markdown("### 💡 Agent Answer")
                            st.markdown(answer)

                            # Badges
                            badge_cols = st.columns(4)
                            with badge_cols[0]:
                                st.metric("Reasoning Iterations", iterations)
                            with badge_cols[1]:
                                st.metric("Sources Used", len(sources))
                            with badge_cols[2]:
                                st.metric("Response Time", f"{elapsed}s")
                            with badge_cols[3]:
                                if degraded:
                                    st.warning(f"Degraded: {degrade_reason or 'Fallback'}")
                                else:
                                    st.success("Optimal Response")

                            # Expandable Step-by-Step Reasoner Trace
                            with st.expander("🧠 Step-by-Step Agentic Reasoner Trace", expanded=True):
                                if action_history:
                                    for idx, act in enumerate(action_history, 1):
                                        st.markdown(f"**Step {idx}:** `{act}`")
                                else:
                                    st.write("No intermediate action trace returned.")

                            # Sources
                            if sources:
                                with st.expander("📚 Sources & References", expanded=False):
                                    for s in sources:
                                        st.code(s, language="text")

                        else:
                            # Standard RAG
                            is_insufficient = res_data.get("is_insufficient", False)
                            from_cache = res_data.get("from_cache", False)

                            add_log("Query Step 2", "Vector retrieval completed", "INFO")
                            add_log(
                                "Query Complete",
                                f"Standard answer ready in {elapsed}s (cache={from_cache})",
                                "OK",
                            )

                            st.markdown("### 💡 Answer")
                            st.markdown(answer)

                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Sources Used", len(sources))
                            with col_m2:
                                st.metric("Cache Hit", "Yes" if from_cache else "No")
                            with col_m3:
                                st.metric("Sufficient Context", "No" if is_insufficient else "Yes")

                            if sources:
                                with st.expander("📚 Sources", expanded=False):
                                    for s in sources:
                                        st.code(s, language="text")

                    else:
                        add_log("Query Error", f"HTTP {resp.status_code}: {resp.text}", "ERR")
                        st.error(f"Query failed with status code {resp.status_code}: {resp.text}")

                except Exception as e:
                    add_log("Query Exception", str(e), "ERR")
                    st.error(f"Error during query execution: {e}")

# ---------------------------------------------------------
# Tab 3: Agent Graph Architecture
# ---------------------------------------------------------
with tab_graph:
    st.subheader("LangGraph Multi-Step State Machine")
    st.write(
        "This diagram represents the compiled state machine driving the agent's iterative reasoning, tool-use, evaluation, and synthesis."
    )

    if st.button("Load Graph Architecture"):
        if not server_online:
            st.error("FastAPI backend is offline!")
        else:
            try:
                g_resp = requests.get(f"{backend_url}/v1/query/graph", timeout=10)
                if g_resp.status_code == 200:
                    mermaid_code = g_resp.json().get("mermaid", "")
                    st.code(mermaid_code, language="mermaid")
                    st.info("Tip: Copy the code above and view it visually on [mermaid.live](https://mermaid.live).")
                else:
                    st.error(f"Failed to fetch graph: {g_resp.status_code}")
            except Exception as e:
                st.error(f"Error fetching graph: {e}")
