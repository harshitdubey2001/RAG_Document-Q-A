import os
from langchain_community.vectorstores import FAISS
import numpy as np
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

groq_api_key = os.getenv("GROQ_API_KEY")

HF_TOKEN = st.secrets["HF_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(groq_api_key=groq_api_key,model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
   """
   Answer the questions based on the provided context only.
   please provide the most accurate response based on the question
   <context>
   {context}
   <context>
   question:{input}


   """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")  ## Data ingestion step
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embedding)
st.title("RAG Documents Q&A with Groq")        

user_prompt = st.text_input("Enter your query from the research paper")

if "vectors" not in st.session_state:
    st.session_state["vectors"] = None

if "embedding_done" not in st.session_state:
    st.session_state["embedding_done"] = False


if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time 

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({"input":user_prompt})
    print(f"Response time : {time.process_time()-start}")

    st.write(response["answer"])

    ## With a streamlit expander

    with st.expander("Documnets Similarity Seacrh"):
        for i ,doc in enumerate(response['context']):
            st.write(doc.page_content)

            st.write("-------")
