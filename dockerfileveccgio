FROM nvcr.io/nvidia/pytorch:24.05-py3

ARG USER=standard
ARG USER_ID=1006
ARG USER_GROUP=standard
ARG USER_GROUP_ID=1006
ARG USER_HOME=/home/${USER}

# Crea utente non-root
RUN groupadd --gid $USER_GROUP_ID $USER && \
    useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER

# Installa pacchetti extra
RUN apt-get update && apt-get install -y curl

# Installa librerie Python necessarie
RUN pip install torchgeo

# Imposta utente non-root
USER $USER

# Imposta directory di lavoro
WORKDIR /workspace

# Imposta entrypoint per lanciare comandi dinamici
ENTRYPOINT ["torchrun"]
