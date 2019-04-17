FROM nvidia/cuda:9.0-cudnn7-devel

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev

RUN apt-get -y update && apt-get install -y g++ gcc gfortran build-essential git libopenblas-dev
RUN  rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda

RUN rm ~/miniconda.sh

RUN /opt/conda/bin/conda install conda-build && \
    /opt/conda/bin/conda create -y --name pytorch python=3.7 numpy pyyaml scipy ipython mkl&& \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/pytorch/bin:$PATH

RUN conda install -y --name pytorch pytorch torchvision cudatoolkit=9.0 -c pytorch
RUN conda install -y --name pytorch -c anaconda protobuf

RUN useradd -ms /bin/bash anh


WORKDIR /tmp/
ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -r requirements.txt

RUN conda install --name pytorch -c conda-forge libjpeg-turbo
RUN CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd
RUN conda install -y --name pytorch jpeg libtiff