# Imagem base
FROM python:3.11-slim

# Variáveis de ambiente para não gerar arquivos .pyc e evitar buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de dependências
COPY requirements.txt .

# Instala as dependências do Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/amaiya/eli5@tfkeras_0_10_1

# Copia o restante do código
COPY . .

# Definindo a variável de ambiente TF_USE_LEGACY_KERAS
ENV TF_USE_LEGACY_KERAS=1

# Definindo variável para GPU (se for usar GPU no host)
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0

CMD ["jupyter", "lab", "--notebook-dir=/app", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
