import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Set page config
st.set_page_config(page_title="📄 RAG PDF Q&A", layout="centered")

# Sidebar Instructions
st.sidebar.title("📘 Instructions")
st.sidebar.markdown("""
1. Upload a PDF file.
2. Enter your Groq API key.
3. Type your question.
4. Get an answer from the document!
""")

# Title
st.title("📄 RAG PDF Question Answering")
st.markdown("Ask questions about your uploaded PDF using **Groq + Llama3**.")

# Input for Groq API Key
groq_api_key = st.text_input("🔑 Enter your Groq API Key", type="password")
os.environ["GROQ_API_KEY"] = groq_api_key  # Optional: set env var

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload a PDF Document", type="pdf")

# Store vectorstore in session_state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

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
        st.info("📄 PDF already loaded. Ready for questions!")

else:
    st.info("📂 Please upload a PDF file to begin.")
    groq_api_key = None

# QA Chain Setup
if groq_api_key and st.session_state.vectorstore:
    try:
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            groq_api_key=groq_api_key
        )

        prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

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

        # Question input
        query = st.text_input("❓ Ask a question about the PDF:")
        if query:
            with st.spinner("🧠 Getting answer from Groq..."):
                response = qa_chain.invoke({"query": query})
                st.markdown("### 💡 Answer:")
                st.success(response["result"])
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    if groq_api_key is None and uploaded_file:
        st.warning("🔑 Please enter your Groq API key to proceed.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io ), [LangChain](https://www.langchain.com ), and [Groq](https://console.groq.com ).")