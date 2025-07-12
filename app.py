import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import load_local
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import os


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def get_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks
    return []


def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    return None


def get_conversational_chain(model_name, api_key=None):
    if model_name == "Google AI":
        prompt_template = PromptTemplate(
            template="""Answer the question based on the context provided. If the answer is not in the context, say "I don't know".

Context: {context}
Question: {question}
Answer:""",
            input_variables=["context", "question"]
        )
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
        return chain
    return None


def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please provide API key and upload PDF documents.")
        return

    text_chunks = get_chunks(get_pdf_text(pdf_docs), model_name)
    get_vector_store(text_chunks, model_name, api_key)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_db = load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = vector_db.similarity_search(user_question)
    chain = get_conversational_chain(model_name, api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    user_question_output = user_question
    response_output = response['output_text']
    pdf_names = [pdf.name for pdf in pdf_docs]

    conversation_history.append([
        user_question_output,
        response_output,
        model_name,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ", ".join(pdf_names)
    ])

    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{ background-color: #2b313e; }}
            .chat-message.bot {{ background-color: #475063; }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
        </style>

        <div class="chat-message user">
            <div class="avatar"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"></div>
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar"><img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp"></div>
            <div class="message">{response_output}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display previous conversations
    for question, answer, model_name, time_stamp, pdf_name in reversed(conversation_history[:-1]):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"></div>
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar"><img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp"></div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Export as CSV
    if conversation_history:
        df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download CSV</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
    st.title("📄 Chat with Multiple PDFs")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    st.sidebar.title("Configuration")
    model_name = st.sidebar.selectbox("Select Model", ["Google AI"])
    api_key = st.sidebar.text_input("Google API Key", type="password")

    st.sidebar.markdown("[AIzaSyAmlqWg52M7C0ZgYe4qKUPMe3_yh-AK_zc](https://ai.google.dev/)")

    pdf_docs = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            st.success("PDFs processed successfully.")
        else:
            st.warning("Please upload PDF files.")

    user_question = st.text_input("Ask a question from the PDFs")

    if st.sidebar.button("Reset"):
        st.session_state.conversation_history = []
        st.experimental_rerun()

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""


if __name__ == "__main__":
    main()
