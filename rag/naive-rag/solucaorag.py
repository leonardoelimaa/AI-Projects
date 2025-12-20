from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.question_answering import load_qa_chain

import os

"""## Carregando modelos de genAI e dados"""

os.environ["OPENAI_API_KEY"] = "Minha Chave de API"

#Load dos modelos (Embedding e LLM)

embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name= "gpt-4o-mini", max_tokens= 200)

from google.colab import files
uploaded = files.upload()

#Carregar o PDF
pdf_link = "DOC-SF238339076816-20230503.pdf"

loader = PyPDFLoader(pdf_link, extract_images= False)
pages = loader.load_and_split()

"""## Manipulando dados"""

#Separando em Chunks
text_splitter = RecursiveCharacterTextSplitter (
    chunk_size = 4000,
    chunk_overlap = 20,
    length_function = len,
    add_start_index = True,
)

chunks = text_splitter.split_documents(pages)

# Salvar no Vector DB - Chroma
db = Chroma.from_documents(chunks, embedding= embeddings_model, persist_directory= "text_index")
db.persist()

# Carregar DB
vectordb = Chroma(persist_directory= "text_index", embedding_function= embeddings_model)

# Load Retriever
retriever = vectordb.as_retriever(search_kwargs= {"k": 3})

# Construcao da cadeia de prompt para chamada do LLM
chain = load_qa_chain(llm, chain_type= "stuff")

"""## Chamada de API com dados manipulados pelo RAG"""

def ask(question):
    context = retriever.invoke(question)
    answer = chain.invoke({
        "input_documents": context,
        "question": question
    })["output_text"]
    return answer

user_question = input("User: ")
answer = ask(user_question)
print("Answer: ", answer)