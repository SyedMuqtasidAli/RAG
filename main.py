# Step 2: Import libraries
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from google.colab import files

# Step 3: Upload PDF
print("Upload your PDF file:")
uploaded = files.upload()
pdf_path = next(iter(uploaded))

# Step 4: Load and split PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    return docs

docs = load_pdf(pdf_path)
print("✅ PDF loaded and split into chunks.")

# Step 5: Create embeddings and FAISS vector store
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

vectorstore = create_vector_store(docs)
print("✅ FAISS vector store created and saved.")

# Step 6: Setup QA Chain using Groq API
def setup_qa_chain(vectorstore, groq_api_key):
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        groq_api_key=groq_api_key
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return qa

# Replace this with your actual Groq API key
GROQ_API_KEY = "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"

qa_chain = setup_qa_chain(vectorstore, GROQ_API_KEY)
print("✅ QA chain ready. You can now ask questions!\n")

# Step 7: Ask Questions in Colab
while True:
    query = input("\n❓ Ask a question about the PDF (type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    response = qa_chain.invoke({"query": query})
    print("\n🧠 Answer:", response["result"])