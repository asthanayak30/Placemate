import streamlit as st
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

st.title("Custom Data Chatbot")

# Upload CSV or Excel file through Streamlit
uploaded_file = st.file_uploader("responses.csv", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the data
    file_extension = uploaded_file.name.split(".")[-1]
    
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension == "xlsx":
        df = pd.read_excel(uploaded_file)

    # Convert dataframe to text format for processing
    data_text = df.to_string()

    # Create embeddings from the data
    embeddings = OllamaEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    final_documents = text_splitter.create_documents([data_text])
    vectors = FAISS.from_documents(final_documents, embeddings)

    # Setup LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama2-7b")

    prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Take user input
    prompt = st.text_input("Input your prompt here")

    if prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # Show retrieved documents for reference
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
