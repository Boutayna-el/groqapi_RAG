import streamlit as st
import os
from langchain_groq import ChatGroq
import time
#from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

#load APIKEK GROQ
groq_api_key = os.getenv("GROQ_API")

st.title("Chatgroq with Llama3")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt= ChatPromptTemplate.from_template(

"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}

"""

)

def vector_embeddings():
    if "vectors" not in st.session_state:

        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)



prompt1= st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("VectorStoreDB is ready")



if prompt1:
    document_chain= create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retieval_chain(retriever, document_chain)
    start=time.process_time()
    response = retrieval_chain.invoke({"input":prompt1})
    print("Response Time:", time.process_time()-start)
    st.write(response["answer"])


