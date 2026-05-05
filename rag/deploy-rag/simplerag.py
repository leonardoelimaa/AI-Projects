import os
import json
import sys

# --- CORREÇÃO CRÍTICA PARA O CHROMA NO LAMBDA ---
# O Lambda usa uma versão antiga do SQLite. O pysqlite3-binary resolve isto.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("Aviso: pysqlite3-binary não encontrado. O Chroma pode falhar.")

# Imports do LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# --- CACHE GLOBAL (Para evitar recarregar o PDF em cada chamada) ---
llm = None
retriever = None

def init_resources():
    """Inicializa os modelos e o banco de dados vetorial uma única vez por container."""
    global llm, retriever

    if llm is None:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            max_tokens=200
        )

    if retriever is None:
        # Caminho absoluto para o PDF dentro do container Lambda
        pdf_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', ''), "DOC-SF238339076816-20230503.pdf")
        
        loader = PyPDFLoader(pdf_path, extract_images=False)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=20
        )

        chunks = text_splitter.split_documents(pages)
        embeddings = OpenAIEmbeddings()

        # Cria o banco em memória (ephemeral)
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="/tmp/chroma_db"
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# --- HANDLER PRINCIPAL (ALB COMPATÍVEL) ---

def lambda_handler(event, context):
    # Log para diagnóstico no CloudWatch
    print("Evento recebido pelo ALB:", json.dumps(event))

    if event.get("httpMethod") == "GET":
        return {
            "statusCode": 200,
            "statusDescription": "200 OK",
            "isBase64Encoded": False,
            "headers": {"Content-Type": "text/plain"},
            "body": "OK - Servidor RAG Ativo"
        }

    try:
        # 2. Tratamento do corpo da requisição (POST)
        body_raw = event.get("body", "{}")
        
        # O ALB pode enviar o body como string ou já parseado dependendo da config
        if isinstance(body_raw, str):
            body = json.loads(body_raw)
        else:
            body = body_raw
            
        question = body.get("question")

        if not question:
            return {
                "statusCode": 400,
                "statusDescription": "400 Bad Request",
                "isBase64Encoded": False,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Campo 'question' é obrigatório"})
            }

        # 3. Inicialização Lazy (Só roda no primeiro acesso do container)
        init_resources()

        # 4. Execução do RAG
        template = """
        Você é um especialista em legislação e tecnologia. Responda com base no contexto informado.
        Contexto: {context}
        Pergunta: {question}
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        
        # Recupera documentos relevantes
        context_docs = retriever.invoke(question)
        
        # Cria a sequência de execução
        chain = RunnableSequence(prompt | llm)
        
        # Invoca o LLM
        response = chain.invoke({
            "context": context_docs,
            "question": question
        })

        # 5. Resposta Final formatada para o ALB
        return {
            "statusCode": 200,
            "statusDescription": "200 OK",
            "isBase64Encoded": False,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "message": "Tarefa Concluída",
                "details": response.content
            })
        }

    except Exception as e:
        print(f"ERRO CRÍTICO: {str(e)}")
        return {
            "statusCode": 500,
            "statusDescription": "500 Internal Server Error",
            "isBase64Encoded": False,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Erro interno na função Lambda",
                "message": str(e)
            })
        }