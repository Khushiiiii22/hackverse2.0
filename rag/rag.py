import streamlit as st
import tempfile
import uuid
import os
from .custom_splitter import load_and_split
from .vectorstore import vectorstore
from .chains import build_chain

def simplify_legal_doc():
    st.set_page_config(page_title="📄 Legal Simplifier", layout="centered")
    st.title("📄 Legal Simplifier")
    st.markdown("Upload a contract, then ask plain-English questions.")

    # State initialization
    if "doc_id" not in st.session_state:
        st.session_state.doc_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar: file upload and processing
    with st.sidebar:
        uploaded_file = st.file_uploader("1. Choose PDF/DOCX", type=["pdf", "docx"])
        if st.button("2. Process document") and uploaded_file:
            with st.spinner("Chunking & storing …"):
                uid = str(uuid.uuid4())
                suffix = f".{uploaded_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                docs = load_and_split(tmp_path)
                for doc in docs:
                    doc.metadata.update({"doc_id": uid, "name": uploaded_file.name})
                vectorstore.add_documents(docs)
                os.remove(tmp_path)
                st.session_state.doc_id = uid
                st.success(f"Done! {len(docs)} chunks stored.")

    # Chat interface
    if st.session_state.doc_id:
        chain = build_chain(st.session_state.doc_id)

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Quick insights styling and buttons
        st.markdown("""
            <style>
            .quick-insight {
                display: inline-block;
                background-color: #262730;
                color: white;
                padding: 12px 16px;
                border-radius: 8px;
                margin: 0 8px 8px 0;
                text-align: center;
                font-weight: bold;
                cursor: pointer;
                transition: background-color 0.2s;
                min-width: 140px;
                max-width: 160px;
                box-sizing: border-box;
                border: 1px solid #444;
            }
            .quick-insight:hover {
                background-color: #333;
            }
            .quick-insight .icon {
                font-size: 1.2em;
                margin-bottom: 4px;
                display: block;
            }
            </style>
        """, unsafe_allow_html=True)

        cols = st.columns(5)
        actions = [
            ("📋", "Executive Summary", "Summarize this document in one paragraph. Focus on purpose, parties, and key obligations.", "exec_sum"),
            ("📌", "Key Clauses", "List the top 5 most important clauses in this document. Include clause numbers or page references if available.", "key_clauses"),
            ("⚠️", "Risk Analysis", "Identify any potential legal or financial risks in this document. Explain why each is a risk.", "risk_analysis"),
            ("✅", "Action Items", "What are the required actions for each party (e.g., landlord, tenant) in this document? List them clearly.", "action_items"),
            ("💡", "Recommendations", "Based on this document, what recommendations would you give to the user to improve clarity, reduce risk, or ensure compliance?", "recommendations"),
        ]

        for col, (icon, label, question, key) in zip(cols, actions):
            col.markdown(f'<div class="quick-insight"><span class="icon">{icon}</span>{label}</div>', unsafe_allow_html=True)
            if col.button("", key=key, help=label):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking …"):
                        result = chain.invoke({"input": question})
                        answer = result["answer"]
                        sources = result.get("context", [])
                    st.write(answer)
                    with st.expander("Sources"):
                        st.json(sources)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        # User chat input
        if question := st.chat_input("Ask about the document …"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking …"):
                    result = chain.invoke({"input": question})
                    answer = result["answer"]
                    sources = result.get("context", [])
                st.write(answer)
                with st.expander("Sources"):
                    st.json(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        st.info("Please upload & process a document first.")
