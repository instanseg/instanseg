FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

RUN apt-get update -y && \
    apt-get install -y wget unzip git && \
    apt-get clean

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    chmod +x /tmp/miniconda.sh && \
    /tmp/miniconda.sh -b


RUN wget https://github.com/instanseg/instanseg/releases/download/instanseg_models_v1/brightfield_nuclei.zip && \
    unzip brightfield_nuclei.zip && \
    rm brightfield_nuclei.zip
RUN wget https://github.com/instanseg/instanseg/releases/download/instanseg_models_v1/fluorescence_nuclei_and_cells.zip && \
    unzip fluorescence_nuclei_and_cells.zip && \
    rm fluorescence_nuclei_and_cells.zip

RUN conda update conda

ADD . instanseg

RUN conda env create -f ./instanseg/env.yml

RUN conda init && \
    . /root/.bashrc && \
    conda activate instanseg && \
    conda install pip && \
    conda remove pytorch torchvision monai && \
    conda install pytorch==2.1.1 torchvision==0.16.1 monai=1.3.0 pytorch-cuda=12.1 -c conda-forge -c pytorch -c nvidia && \
    pip install cupy-cuda12x && \
    pip install ./instanseg
