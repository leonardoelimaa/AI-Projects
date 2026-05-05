# Use a imagem base da AWS para Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Instala dependências de compilação (necessário para ChromaDB/pysqlite3)
RUN microdnf update -y && microdnf install -y gcc-c++ make

# Copia e instala requirements primeiro (melhora o cache do Docker)
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copia o PDF e o código
COPY DOC-SF238339076816-20230503.pdf ${LAMBDA_TASK_ROOT}/
COPY simplerag.py ${LAMBDA_TASK_ROOT}/

# IMPORTANTE: A Lambda não precisa de chmod +x para o arquivo .py
# pois ela chama o interpretador python diretamente.

# Define o Handler
CMD ["simplerag.lambda_handler"]