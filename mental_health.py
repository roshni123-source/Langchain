from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain 
# import streamlit as st
pdf_path = "book.pdf"
import os
os.environ["OPENAI_API_KEY"] ="your_api_key"
# st.title("Mental health conversational chatbot")

def main():
  loader = PyPDFLoader(file_path=pdf_path) 
  documents = loader.load() 
  text_splitter = CharacterTextSplitter( chunk_size=100, chunk_overlap=20, separator="\n" ) 
  docs = text_splitter.split_documents(documents)
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_documents(docs, embeddings)    
  vectorstore.save_local("vector_db")
  retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
  llm = ChatOpenAI()
  combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
  combine_docs_chain.invoke({"context": docs, "input": "what is stress adaptation?"})
#   print(combine_docs_chain)
  retriever = FAISS.load_local("vector_db", embeddings).as_retriever()
  retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
  response = retrieval_chain.invoke({"input": "tell me what are the difference between stress and anxiety?"})
  print(response["answer"])

if __name__ == "__main__": 
  main()         

