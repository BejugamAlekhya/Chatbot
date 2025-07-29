import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
 
st.title("Chat with your PDF")
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="PDF")
if uploaded_file is not None:
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks = text_splitter.split_text(text)
 
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
    # Create vector store
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Load local LLaMA2 model via Ollama
    llm = Ollama(model="llama3")  # Make sure llama2 is pulled
 
    # Create QA chain
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type="stuff", return_source_documents=True)
 
    # Ask questions
    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Thinking..."):
            result = qa({"query": question})
            st.markdown("### 🤖 Answer:")
            st.write(result["result"])
            with st.expander("📄 Source Chunks"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:500])
