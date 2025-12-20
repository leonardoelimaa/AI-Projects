from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

import os
os.environ['OPENAI_API_KEY'] = "Minha Chave de API"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name= "gpt-4o-mini", max_tokens=500)

#Carregar o PDF

from google.colab import files
uploaded = files.upload()

pdf_link = "DOC-SF238339076816-20230503.pdf"

loader = PyPDFLoader(pdf_link, extract_images= False)
pages = loader.load_and_split()

# Splitter

child_splitter = RecursiveCharacterTextSplitter(chunk_size = 200)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 4000,
    chunk_overlap = 200,
    length_function = len,
    add_start_index = True
)

#Storages
store = InMemoryStore()
vectorstore = Chroma(embedding_function=embeddings, persist_directory='childVectorDB')

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

parent_document_retriever.add_documents(pages, ids=None)

parent_document_retriever.vectorstore.get()

TEMPLATE = """
  Você é um especialista em legislação e tecnologia. Responda a pergunta abaixo utilizando o contexto informado.
  Query:
  {question}

  Context:
  {context}
"""

rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)

setup_retrival = RunnableParallel({"question": RunnablePassthrough(), "context": parent_document_retriever})

output_parser = StrOutputParser()

parent_chain_retrival = setup_retrival | rag_prompt | llm | output_parser

parent_chain_retrival.invoke("Quais os principais riscos do marco legal de ia?")