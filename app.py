import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
import os

# Hardcoded Groq API Key
GROQ_API_KEY = "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"

# Set page config
st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")

# Sidebar Instructions
st.sidebar.title("📘 Instructions")
st.sidebar.markdown("""
1. Upload a PDF file.
2. Ask questions in the chat below.
3. Get answers powered by **Groq + Llama3**!
""")

# Title
st.title("💬 RAG Chatbot")
st.markdown("Chat with your PDF document using **Groq + Llama3**.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Store vectorstore in session_state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload a PDF Document", type="pdf", key="pdf_uploader")

# Load PDF and create vector store if file is uploaded
if uploaded_file:
    if st.session_state.vectorstore is None:
        with st.spinner("🔄 Processing PDF..."):
            # Save uploaded file temporarily
            temp_pdf_path = f"temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load and split PDF
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load_and_split()

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

            # Create FAISS vectorstore
            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state.vectorstore = vectorstore

            # Cleanup temp file
            os.remove(temp_pdf_path)

            st.success("✅ PDF processed and indexed!")

    else:
        st.info("📄 PDF already loaded. Ready for chatting!")
else:
    st.info("📂 Please upload a PDF file to begin.")

# Chat input
if st.session_state.vectorstore:
    try:
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            groq_api_key=GROQ_API_KEY  # Using hardcoded key
        )

        prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.also be specific and to the point in answer do not write extra.

{context}

Question: {question}
Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

        user_input = st.chat_input("Ask a question about the PDF...")
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            # Get response from QA chain
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    response = qa_chain.invoke({"query": user_input})
                    answer = response["result"]
                    st.markdown(answer)
            st.session_state.messages.append(AIMessage(content=answer))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")