import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

class RAGEngine:
    def __init__(self, openai_api_key, persist_directory="./chroma_db"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
    
    def ingest_document(self, file_path):
        """Load document and create vector store"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        return f"Processed {len(texts)} text chunks"
    
    def query(self, question, k=4):
        """Query documents and generate answer"""
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k})
        )
        
        return qa_chain.run(question)
