
import os
import hashlib

import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine


load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF AI Assistant")
st.write("Upload a PDF document and ask questions about its content.")


if "rag" not in st.session_state:
    st.session_state.rag = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_id" not in st.session_state:
    st.session_state.document_id = None


with st.sidebar:
    st.header("Settings")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


if uploaded_file is not None:
    if not API_KEY:
        st.error("GROQ_API_KEY was not found in the environment.")

    else:
        pdf_data = uploaded_file.getvalue()
        document_id = hashlib.sha256(pdf_data).hexdigest()

        if document_id != st.session_state.document_id:
            try:
                with st.spinner("Processing document..."):
                    rag = RAGEngine(API_KEY)
                    chunk_count = rag.process_pdf(pdf_data)

                    st.session_state.rag = rag
                    st.session_state.document_id = document_id
                    st.session_state.messages = []

                st.success(
                    f"Document processed successfully. "
                    f"{chunk_count} chunks created."
                )

            except Exception as error:
                st.session_state.rag = None
                st.session_state.document_id = None
                st.error(f"Document processing failed: {error}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask something about your PDF...")

if question:
    if st.session_state.rag is None:
        st.warning("Please upload a PDF document first.")

    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.rag.ask(question)
                    answer = result["answer"]

                    st.markdown(answer)

                    sources = result.get("sources", [])

                    if sources:
                        pages = ", ".join(map(str, sources))
                        st.caption(f"Source pages: {pages}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:
                    st.error(f"Failed to generate answer: {error}")
